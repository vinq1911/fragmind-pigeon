"""
Cross-language interop test: Python writes ring + LOA, Go reads.

Creates shared-memory files, writes from Python, then invokes a Go test
program to read and verify. Proves the binary layouts are identical.

Run: python -m pytest test_cross_language.py -v -s
"""

import mmap
import os
import struct
import subprocess
import tempfile
import time
import zlib

import pytest

from fragpigeon import (
    Header,
    LOAPool,
    LOARef,
    Ring,
    FLAG_LOA_PTR,
    HDR_SIZE,
    KIND_PROCESS,
    LOA_REF_SIZE,
    SCHEMA_WEIGHT_SHARD,
    write_loa,
)


def make_ring_file(path, cap_slots=64, slot_size=512):
    """Create a ring shm file on disk (not unlinked, so Go can open it)."""
    size = 64 + cap_slots * slot_size
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o644)
    os.ftruncate(fd, size)
    mm = mmap.mmap(fd, size, access=mmap.ACCESS_WRITE)

    struct.pack_into("<Q", mm, 0, cap_slots)
    struct.pack_into("<Q", mm, 8, 0)
    struct.pack_into("<Q", mm, 16, 0)
    struct.pack_into("<I", mm, 24, slot_size)
    struct.pack_into("<Q", mm, 32, 0xFFFFFFFFFFFFFFFF)
    struct.pack_into("<Q", mm, 40, 0xFFFFFFFFFFFFFFFF)
    mm.close()
    os.close(fd)


def make_payload(size):
    buf = bytearray(size)
    for i in range(size - 4):
        buf[i] = (i * 7 + 13) & 0xFF
    crc = zlib.crc32(buf[:size - 4]) & 0xFFFFFFFF
    struct.pack_into("<I", buf, size - 4, crc)
    return bytes(buf)


GO_VERIFIER = """
package main

import (
    "encoding/binary"
    "fmt"
    "hash/crc32"
    "os"

    "golang.org/x/sys/unix"
)

func main() {
    ringPath := os.Args[1]
    loaPath := os.Args[2]

    // Open and read the ring
    ringFD, err := unix.Open(ringPath, unix.O_RDWR, 0)
    if err != nil { fmt.Fprintf(os.Stderr, "open ring: %v\\n", err); os.Exit(1) }
    var st unix.Stat_t
    unix.Fstat(ringFD, &st)
    mem, err := unix.Mmap(ringFD, 0, int(st.Size), unix.PROT_READ|unix.PROT_WRITE, unix.MAP_SHARED)
    if err != nil { fmt.Fprintf(os.Stderr, "mmap ring: %v\\n", err); os.Exit(1) }

    // Read ring control
    prod := binary.LittleEndian.Uint64(mem[8:])
    cons := binary.LittleEndian.Uint64(mem[16:])
    slotSize := binary.LittleEndian.Uint32(mem[24:])
    capSlots := binary.LittleEndian.Uint64(mem[0:])

    if prod == cons {
        fmt.Fprintln(os.Stderr, "ring empty")
        os.Exit(1)
    }

    // Read slot
    slotOff := 64 + int(cons & (capSlots-1)) * int(slotSize)
    hdr := mem[slotOff:slotOff+64]

    msgLen := binary.LittleEndian.Uint32(hdr[0:])
    kind := binary.LittleEndian.Uint16(hdr[4:])
    flags := binary.LittleEndian.Uint16(hdr[6:])
    conceptID := binary.LittleEndian.Uint64(hdr[16:])
    schemaID := binary.LittleEndian.Uint16(hdr[26:])
    msgID := binary.LittleEndian.Uint32(hdr[32:])

    fmt.Printf("ring: len=%d kind=%d flags=%d concept=%x schema=%d msgID=%d\\n",
        msgLen, kind, flags, conceptID, schemaID, msgID)

    if flags & 0x0010 != 0 {
        // LOA pointer
        payload := mem[slotOff+64:slotOff+64+int(msgLen)]
        poolID := binary.LittleEndian.Uint16(payload[0:])
        slotID := binary.LittleEndian.Uint16(payload[2:])
        offset := binary.LittleEndian.Uint32(payload[4:])
        length := binary.LittleEndian.Uint32(payload[8:])
        fmt.Printf("loa_ref: pool=%d slot=%d offset=%d length=%d\\n", poolID, slotID, offset, length)

        // Open LOA pool and verify data
        loaFD, err := unix.Open(loaPath, unix.O_RDWR, 0)
        if err != nil { fmt.Fprintf(os.Stderr, "open loa: %v\\n", err); os.Exit(1) }
        unix.Fstat(loaFD, &st)
        loaMem, err := unix.Mmap(loaFD, 0, int(st.Size), unix.PROT_READ|unix.PROT_WRITE, unix.MAP_SHARED)
        if err != nil { fmt.Fprintf(os.Stderr, "mmap loa: %v\\n", err); os.Exit(1) }

        magic := binary.LittleEndian.Uint64(loaMem[0:])
        dataBase := binary.LittleEndian.Uint32(loaMem[24:])
        loaSlotSize := binary.LittleEndian.Uint32(loaMem[16:])

        dataOff := int(dataBase) + int(slotID)*int(loaSlotSize) + int(offset)
        data := loaMem[dataOff:dataOff+int(length)]

        // Verify CRC (same as GeneratePayload: last 4 bytes = CRC of rest)
        if int(length) >= 8 {
            expected := binary.LittleEndian.Uint32(data[length-4:])
            actual := crc32.ChecksumIEEE(data[:length-4])
            if expected == actual {
                fmt.Printf("verify: PASS (magic=%x crc=%08x)\\n", magic, actual)
            } else {
                fmt.Printf("verify: FAIL (expected=%08x actual=%08x)\\n", expected, actual)
                os.Exit(1)
            }
        }
    } else {
        // Inline payload
        payload := mem[slotOff+64:slotOff+64+int(msgLen)]
        if int(msgLen) >= 8 {
            expected := binary.LittleEndian.Uint32(payload[msgLen-4:])
            actual := crc32.ChecksumIEEE(payload[:msgLen-4])
            if expected == actual {
                fmt.Printf("verify: PASS (crc=%08x)\\n", actual)
            } else {
                fmt.Printf("verify: FAIL (expected=%08x actual=%08x)\\n", expected, actual)
                os.Exit(1)
            }
        }
    }
}
"""


class TestCrossLanguage:
    def test_python_write_go_read_inline(self, tmp_path):
        """Python writes inline message to ring, Go reads and verifies CRC."""
        ring_path = os.path.join(str(tmp_path), "ring.shm")
        make_ring_file(ring_path, cap_slots=16, slot_size=512)

        # Python writes
        fd = os.open(ring_path, os.O_RDWR)
        ring = Ring.from_fd(fd)
        os.close(fd)

        payload = make_payload(256)
        hdr = Header(
            length=256, kind=KIND_PROCESS, schema_id=SCHEMA_WEIGHT_SHARD,
            concept_id=0xDEADBEEF, concept_bits=24, src_id=42, msg_id=7, ver=1,
        )
        assert ring.try_write(hdr, payload)
        ring.close()

        # Go reads
        go_file = os.path.join(str(tmp_path), "verifier.go")
        with open(go_file, "w") as f:
            f.write(GO_VERIFIER)

        result = subprocess.run(
            ["go", "run", go_file, ring_path, "/dev/null"],
            capture_output=True, text=True, timeout=30,
            cwd=os.path.join(os.path.dirname(__file__), "../.."),
        )
        print(result.stdout)
        if result.stderr:
            print("stderr:", result.stderr)
        assert result.returncode == 0
        assert "verify: PASS" in result.stdout

    def test_python_write_go_read_loa(self, tmp_path):
        """Python writes LOA message, Go reads ring pointer and verifies LOA data."""
        ring_path = os.path.join(str(tmp_path), "ring.shm")
        loa_path = os.path.join(str(tmp_path), "loa.shm")
        make_ring_file(ring_path, cap_slots=16, slot_size=256)

        # Python writes LOA
        fd = os.open(ring_path, os.O_RDWR)
        ring = Ring.from_fd(fd)
        os.close(fd)
        pool = LOAPool.create(loa_path, pool_id=1, num_slots=8, slot_size=4096)

        payload = make_payload(2048)
        hdr = Header(
            length=0, kind=KIND_PROCESS, schema_id=SCHEMA_WEIGHT_SHARD,
            concept_id=0xCAFEBABE, concept_bits=16, src_id=99, msg_id=42, ver=1,
        )
        ref = write_loa(ring, pool, hdr, payload, owner_frag_id=99)
        assert ref is not None
        ring.close()
        pool.close()

        # Go reads
        go_file = os.path.join(str(tmp_path), "verifier.go")
        with open(go_file, "w") as f:
            f.write(GO_VERIFIER)

        result = subprocess.run(
            ["go", "run", go_file, ring_path, loa_path],
            capture_output=True, text=True, timeout=30,
            cwd=os.path.join(os.path.dirname(__file__), "../.."),
        )
        print(result.stdout)
        if result.stderr:
            print("stderr:", result.stderr)
        assert result.returncode == 0
        assert "verify: PASS" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
