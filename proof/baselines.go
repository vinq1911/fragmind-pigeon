package proof

import (
	"encoding/binary"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"time"
)

// BenchUDS benchmarks Unix Domain Socket transport (end-to-end).
func BenchUDS(payloadSize, count int) (BenchResult, error) {
	dir := os.TempDir()
	sock := filepath.Join(dir, fmt.Sprintf("proof-uds-%d.sock", time.Now().UnixNano()))
	defer os.Remove(sock)

	return benchStream("uds", payloadSize, count, func() (net.Conn, net.Conn, func(), error) {
		ln, err := net.Listen("unix", sock)
		if err != nil {
			return nil, nil, nil, err
		}
		acceptCh := make(chan net.Conn, 1)
		go func() {
			c, _ := ln.Accept()
			acceptCh <- c
		}()
		client, err := net.Dial("unix", sock)
		if err != nil {
			ln.Close()
			return nil, nil, nil, err
		}
		server := <-acceptCh
		cleanup := func() {
			client.Close()
			server.Close()
			ln.Close()
		}
		return client, server, cleanup, nil
	})
}

// BenchTCP benchmarks TCP localhost transport (end-to-end).
func BenchTCP(payloadSize, count int) (BenchResult, error) {
	return benchStream("tcp", payloadSize, count, func() (net.Conn, net.Conn, func(), error) {
		ln, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			return nil, nil, nil, err
		}
		acceptCh := make(chan net.Conn, 1)
		go func() {
			c, _ := ln.Accept()
			acceptCh <- c
		}()
		client, err := net.Dial("tcp", ln.Addr().String())
		if err != nil {
			ln.Close()
			return nil, nil, nil, err
		}
		server := <-acceptCh
		cleanup := func() {
			client.Close()
			server.Close()
			ln.Close()
		}
		return client, server, cleanup, nil
	})
}

// BenchPipe benchmarks os.Pipe() transport (end-to-end).
func BenchPipe(payloadSize, count int) (BenchResult, error) {
	r, w, err := os.Pipe()
	if err != nil {
		return BenchResult{}, err
	}
	defer r.Close()
	defer w.Close()

	payload := GeneratePayload(payloadSize)
	lr := NewLatencyRecorder(count)
	verified := true

	lenBuf := make([]byte, 4)

	// Per-message ack channel: reader signals completion so writer
	// measures the full end-to-end latency (write + kernel copy + read).
	ack := make(chan bool, 1)

	// Reader goroutine — reads full message, verifies CRC, acks.
	done := make(chan error, 1)
	go func() {
		readLenBuf := make([]byte, 4)
		readBuf := make([]byte, payloadSize)
		for i := 0; i < count; i++ {
			if _, err := io.ReadFull(r, readLenBuf); err != nil {
				done <- err
				return
			}
			plen := int(binary.LittleEndian.Uint32(readLenBuf))
			if plen > len(readBuf) {
				readBuf = make([]byte, plen)
			}
			if _, err := io.ReadFull(r, readBuf[:plen]); err != nil {
				done <- err
				return
			}
			ack <- VerifyPayload(readBuf[:plen])
		}
		done <- nil
	}()

	start := time.Now()
	for i := 0; i < count; i++ {
		binary.LittleEndian.PutUint32(lenBuf, uint32(payloadSize))

		t0 := time.Now()
		if _, err := w.Write(lenBuf); err != nil {
			return BenchResult{}, err
		}
		if _, err := w.Write(payload); err != nil {
			return BenchResult{}, err
		}
		// Wait for reader to fully receive and verify
		ok := <-ack
		lr.Record(time.Since(t0))
		if !ok {
			verified = false
		}
	}

	if err := <-done; err != nil {
		return BenchResult{}, fmt.Errorf("pipe reader: %w", err)
	}
	wall := time.Since(start)

	stats := lr.Stats()
	totalBytes := int64(count) * int64(payloadSize)

	return BenchResult{
		Transport:     "pipe",
		PayloadName:   payloadName(payloadSize),
		PayloadSize:   payloadSize,
		Ops:           count,
		TotalBytes:    totalBytes,
		WallTime:      wall,
		ThroughputMBs: float64(totalBytes) / wall.Seconds() / 1e6,
		MsgsPerSec:    float64(count) / wall.Seconds(),
		LatencyP50:    stats.P50.Nanoseconds(),
		LatencyP95:    stats.P95.Nanoseconds(),
		LatencyP99:    stats.P99.Nanoseconds(),
		LatencyMin:    stats.Min.Nanoseconds(),
		LatencyMax:    stats.Max.Nanoseconds(),
		Verified:      verified,
	}, nil
}

// benchStream is shared logic for UDS and TCP.
// End-to-end: writer sends, waits for reader ack, so both copies are timed.
// Reader also verifies CRC32 integrity on every message.
func benchStream(
	transport string,
	payloadSize, count int,
	setup func() (writer net.Conn, reader net.Conn, cleanup func(), err error),
) (BenchResult, error) {
	writer, reader, cleanup, err := setup()
	if err != nil {
		return BenchResult{}, err
	}
	defer cleanup()

	payload := GeneratePayload(payloadSize)
	lr := NewLatencyRecorder(count)
	verified := true

	lenBuf := make([]byte, 4)

	// Per-message ack: reader signals it has fully received + verified.
	ack := make(chan bool, 1)

	done := make(chan error, 1)
	go func() {
		readLenBuf := make([]byte, 4)
		readBuf := make([]byte, payloadSize)
		for i := 0; i < count; i++ {
			if _, err := io.ReadFull(reader, readLenBuf); err != nil {
				done <- err
				return
			}
			plen := int(binary.LittleEndian.Uint32(readLenBuf))
			if plen > len(readBuf) {
				readBuf = make([]byte, plen)
			}
			if _, err := io.ReadFull(reader, readBuf[:plen]); err != nil {
				done <- err
				return
			}
			ack <- VerifyPayload(readBuf[:plen])
		}
		done <- nil
	}()

	start := time.Now()
	for i := 0; i < count; i++ {
		binary.LittleEndian.PutUint32(lenBuf, uint32(payloadSize))

		t0 := time.Now()
		if _, err := writer.Write(lenBuf); err != nil {
			return BenchResult{}, err
		}
		if _, err := writer.Write(payload); err != nil {
			return BenchResult{}, err
		}
		// Block until reader has fully received + verified this message
		ok := <-ack
		lr.Record(time.Since(t0))
		if !ok {
			verified = false
		}
	}

	if err := <-done; err != nil {
		return BenchResult{}, fmt.Errorf("%s reader: %w", transport, err)
	}
	wall := time.Since(start)

	stats := lr.Stats()
	totalBytes := int64(count) * int64(payloadSize)

	return BenchResult{
		Transport:     transport,
		PayloadName:   payloadName(payloadSize),
		PayloadSize:   payloadSize,
		Ops:           count,
		TotalBytes:    totalBytes,
		WallTime:      wall,
		ThroughputMBs: float64(totalBytes) / wall.Seconds() / 1e6,
		MsgsPerSec:    float64(count) / wall.Seconds(),
		LatencyP50:    stats.P50.Nanoseconds(),
		LatencyP95:    stats.P95.Nanoseconds(),
		LatencyP99:    stats.P99.Nanoseconds(),
		LatencyMin:    stats.Min.Nanoseconds(),
		LatencyMax:    stats.Max.Nanoseconds(),
		Verified:      verified,
	}, nil
}
