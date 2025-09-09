package main

import (
	"encoding/binary"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
	"time"

	"golang.org/x/sys/unix"
)

func putU64(b []byte, v uint64) { binary.LittleEndian.PutUint64(b, v) }
func putU32(b []byte, v uint32) { binary.LittleEndian.PutUint32(b, v) }

// cross-platform shm ring (temp file + ftruncate + mmap init)
func mkRing(name string, capSlots, slotSize int) (*os.File, error) {
	size := 64 + capSlots*slotSize
	path := filepath.Join(os.TempDir(), name+".shm")
	fd, err := unix.Open(path, unix.O_CREAT|unix.O_EXCL|unix.O_RDWR, 0600)
	if err != nil {
		// reuse if exists
		fd, err = unix.Open(path, unix.O_RDWR, 0600)
		if err != nil {
			return nil, err
		}
	}
	if err := unix.Ftruncate(fd, int64(size)); err != nil {
		unix.Close(fd)
		_ = os.Remove(path)
		return nil, err
	}

	mem, err := unix.Mmap(fd, 0, size, unix.PROT_READ|unix.PROT_WRITE, unix.MAP_SHARED)
	if err != nil {
		unix.Close(fd)
		return nil, err
	}
	putU64(mem[0:], uint64(capSlots))  // CapSlots
	putU64(mem[8:], 0)                 // ProdIdx
	putU64(mem[16:], 0)                // ConsIdx
	putU32(mem[24:], uint32(slotSize)) // SlotSize
	putU64(mem[32:], ^uint64(0))       // ProdEvtFD=-1
	putU64(mem[40:], ^uint64(0))       // ConsEvtFD=-1
	_ = unix.Munmap(mem)
	_ = os.Remove(path) // unlink path

	return os.NewFile(uintptr(fd), name), nil
}

func spawn(absBin string, env map[string]string, fds ...*os.File) *exec.Cmd {
	cmd := exec.Command(absBin)
	cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
	cmd.Env = os.Environ()
	for k, v := range env {
		cmd.Env = append(cmd.Env, k+"="+v)
	}
	cmd.ExtraFiles = fds // FDs will be numbered 3,4,...
	return cmd
}

func mustBuild(absOut, pkg string) {
	// Build the package to an absolute path so coord (running in a temp dir) can exec it.
	cmd := exec.Command("go", "build", "-o", absOut, pkg)
	cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
	if err := cmd.Run(); err != nil {
		log.Fatalf("build %s: %v", pkg, err)
	}
	// Make sure it’s executable (usually already is)
	_ = os.Chmod(absOut, 0o755)
}

func main() {
	// Absolute bin dir for child executables
	binDir, err := os.MkdirTemp("", "fragmind-demo-bin-*")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(binDir)
	recvBin := filepath.Join(binDir, "receiver")
	sendBin := filepath.Join(binDir, "sender")

	// Build the two fragments by package DIRECTORY (not .go file)
	mustBuild(recvBin, "./examples/local_ring_demo/receiver")
	mustBuild(sendBin, "./examples/local_ring_demo/sender")

	// Create two rings
	const slots, slot = 1024, 1024
	ringSR, err := mkRing("ringSR", slots, slot)
	if err != nil {
		log.Fatal(err)
	} // sender->receiver
	ringRS, err := mkRing("ringRS", slots, slot)
	if err != nil {
		log.Fatal(err)
	} // receiver->sender
	defer ringSR.Close()
	defer ringRS.Close()

	// Receiver: IN=ringSR (fd3), OUT=ringRS (fd4)
	rcmd := spawn(recvBin, map[string]string{
		"FM_IN_FD":       "3",
		"FM_OUT_FD":      "4",
		"FM_PIGEON_SOCK": "/tmp/pigeon.sock", // optional; ok if pigeon not running
	}, ringSR, ringRS)
	if err := rcmd.Start(); err != nil {
		log.Fatal(err)
	}
	time.Sleep(200 * time.Millisecond)

	// Sender: IN=ringRS (fd3), OUT=ringSR (fd4)
	scmd := spawn(sendBin, map[string]string{
		"FM_IN_FD":       "3",
		"FM_OUT_FD":      "4",
		"FM_PIGEON_SOCK": "/tmp/pigeon.sock",
	}, ringRS, ringSR)
	if err := scmd.Start(); err != nil {
		log.Fatal(err)
	}

	_ = rcmd.Wait()
	_ = scmd.Wait()

	// End the process group (optional)
	_ = syscall.Kill(0, syscall.SIGTERM)
}
