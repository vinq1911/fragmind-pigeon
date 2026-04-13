"""
Tests for Python fragment attachment to Go pigeon daemon.

Starts a real Go pigeon process, attaches from Python via SCM_RIGHTS,
registers COIs, exchanges data through rings, and discovers the LOA pool.

Run: python -m pytest test_attach.py -v -s
"""

import os
import signal
import struct
import subprocess
import sys
import time
import zlib

import pytest

from fragpigeon import (
    COI,
    COIClient,
    COIHandle,
    Header,
    LOAPool,
    Ring,
    attach,
    discover_loa_pool,
    write_loa,
    read_loa,
    FLAG_LOA_PTR,
    KIND_PROCESS,
    LOA_REF_SIZE,
    SCHEMA_WEIGHT_SHARD,
    SCHEMA_ACTIVATION,
)


REPO_ROOT = os.path.join(os.path.dirname(__file__), "../..")


def start_pigeon(sock_path, coi_path, loa_path):
    """Start a Go pigeon daemon as a subprocess."""
    env = os.environ.copy()
    env["FM_SITE_ID"] = "1"
    env["FM_PIGEON_SOCK"] = sock_path
    env["FM_COI_SHM_PATH"] = coi_path
    env["FM_LOA_SHM_PATH"] = loa_path

    proc = subprocess.Popen(
        ["go", "run", "./cmd/pigeon"],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Wait for pigeon to start (socket becomes available)
    for _ in range(50):
        if os.path.exists(sock_path):
            return proc
        time.sleep(0.1)
    # Check if process died
    if proc.poll() is not None:
        _, stderr = proc.communicate()
        raise RuntimeError(f"pigeon died: {stderr.decode()}")
    raise RuntimeError("pigeon socket not created in 5s")


def kill_pigeon(proc):
    """Stop the pigeon daemon."""
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def make_payload(size):
    """Deterministic payload with CRC32 (matches Go's GeneratePayload)."""
    buf = bytearray(size)
    for i in range(size - 4):
        buf[i] = (i * 7 + 13) & 0xFF
    crc = zlib.crc32(buf[:size - 4]) & 0xFFFFFFFF
    struct.pack_into("<I", buf, size - 4, crc)
    return bytes(buf)


def verify_payload(data):
    if len(data) < 8:
        return False
    expected = struct.unpack_from("<I", data, len(data) - 4)[0]
    return zlib.crc32(data[:len(data) - 4]) & 0xFFFFFFFF == expected


@pytest.fixture
def pigeon(tmp_path):
    """Start a pigeon daemon for the test, yield paths, stop after."""
    sock = f"/tmp/fp-pytest-{os.getpid()}.sock"
    coi = str(tmp_path / "coi.shm")
    loa = str(tmp_path / "loa.shm")

    proc = start_pigeon(sock, coi, loa)
    yield sock, coi, loa
    kill_pigeon(proc)
    try:
        os.unlink(sock)
    except FileNotFoundError:
        pass


class TestAttach:
    def test_basic_attach(self, pigeon):
        """Python attaches to Go pigeon, gets ring FDs and fragID."""
        sock, _, _ = pigeon
        att = attach(sock)
        assert att.frag_id > 0
        assert att.in_ring is not None
        assert att.out_ring is not None
        print(f"attached: frag_id={att.frag_id}")
        att.close()

    def test_write_to_outbound_ring(self, pigeon):
        """Python writes to outbound ring (pigeon reads for routing)."""
        sock, _, _ = pigeon
        att = attach(sock)

        payload = make_payload(128)
        hdr = Header(
            length=len(payload),
            kind=KIND_PROCESS,
            concept_id=0x0001000000000000,
            concept_bits=16,
            schema_id=SCHEMA_ACTIVATION,
            src_id=att.frag_id,
            msg_id=1,
            ver=1,
        )
        assert att.out_ring.try_write(hdr, payload)
        print("outbound write: OK")
        att.close()

    def test_pigeon_writes_to_inbound_ring(self, pigeon):
        """Verify pigeon can write to our inbound ring (via routing)."""
        sock, _, _ = pigeon

        # Attach two fragments
        att_a = attach(sock)
        att_b = attach(sock)

        # Register B's COI so A's messages route to B
        concept_id = 0x00DD000000000000
        with COIClient(sock, att_b.frag_id) as coi:
            coi.register([COI(concept_id=concept_id, bits=16, schema_id=SCHEMA_WEIGHT_SHARD)])
        time.sleep(0.2)  # let registration propagate

        # A sends a message targeting B's COI
        payload = make_payload(64)
        hdr = Header(
            length=len(payload),
            kind=KIND_PROCESS,
            concept_id=concept_id,
            concept_bits=16,
            schema_id=SCHEMA_WEIGHT_SHARD,
            src_id=att_a.frag_id,
            msg_id=42,
            ver=1,
        )
        assert att_a.out_ring.try_write(hdr, payload)

        # B should receive it on inbound ring
        msg = att_b.in_ring.read_within(3.0)
        assert msg is not None, "B did not receive message"
        assert msg.header.msg_id == 42
        assert msg.header.src_id == att_a.frag_id
        assert verify_payload(msg.payload)
        print(f"routed: A(frag={att_a.frag_id}) → pigeon → B(frag={att_b.frag_id}): OK")

        att_a.close()
        att_b.close()


class TestCOI:
    def test_register_and_loa_info(self, pigeon):
        """Register COIs and query LOA pool path."""
        sock, _, loa = pigeon
        att = attach(sock)

        with COIClient(sock, att.frag_id) as coi:
            coi.register([
                COI(concept_id=0xAAAA000000000000, bits=16, schema_id=1),
                COI(concept_id=0xBBBB000000000000, bits=24, schema_id=2),
            ])
            path = coi.loa_info()
            assert path == loa
            print(f"LOA path: {path}")

        att.close()

    def test_coi_handle_auto_renew(self, pigeon):
        """COIHandle auto-renews in background."""
        sock, _, _ = pigeon
        att = attach(sock)

        handle = COIHandle.start(
            socket_path=sock,
            frag_id=att.frag_id,
            cois=[COI(concept_id=0xCCCC000000000000, bits=16, schema_id=1)],
            renew_interval=0.5,
        )
        time.sleep(1.5)  # let at least 2 renewals happen
        handle.close()
        print("COIHandle auto-renew: OK")
        att.close()

    def test_discover_loa_pool(self, pigeon):
        """Discover and open LOA pool from pigeon."""
        sock, _, _ = pigeon
        pool = discover_loa_pool(sock)
        assert pool.num_slots > 0
        assert pool.slot_size > 0

        # Verify we can alloc from the discovered pool
        buf, ref = pool.alloc(256, owner_frag_id=1)
        buf[:] = b"\xAB" * 256
        pool.commit(ref)

        data = pool.deref(ref)
        assert data[:1] == b"\xAB"
        pool.release(ref)
        pool.close()
        print("LOA pool discovered and functional: OK")


class TestLOARouting:
    def test_loa_tensor_via_pigeon(self, pigeon):
        """Python A writes tensor via LOA, pigeon routes to Python B."""
        sock, _, _ = pigeon

        att_a = attach(sock)
        att_b = attach(sock)
        pool = discover_loa_pool(sock)

        # B subscribes to a concept
        concept_id = 0x00EE000000000000
        with COIClient(sock, att_b.frag_id) as coi:
            coi.register([COI(concept_id=concept_id, bits=16, schema_id=SCHEMA_WEIGHT_SHARD)])
        time.sleep(0.2)

        # A writes a "tensor" via LOA
        tensor_data = make_payload(4096)
        hdr = Header(
            kind=KIND_PROCESS,
            concept_id=concept_id,
            concept_bits=16,
            schema_id=SCHEMA_WEIGHT_SHARD,
            src_id=att_a.frag_id,
            msg_id=99,
            ver=1,
        )
        ref = write_loa(att_a.out_ring, pool, hdr, tensor_data, owner_frag_id=att_a.frag_id)
        assert ref is not None

        # B receives the LOA pointer via pigeon routing
        result = read_loa(att_b.in_ring, pool, timeout_s=3.0)
        assert result is not None
        msg, data, got_ref = result
        assert msg.header.msg_id == 99
        assert msg.is_loa
        assert verify_payload(data)

        if got_ref:
            pool.release(got_ref)

        print(f"LOA tensor routed: A → pigeon → B ({len(tensor_data)} bytes): OK")

        pool.close()
        att_a.close()
        att_b.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
