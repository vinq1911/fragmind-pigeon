//go:build darwin || freebsd || netbsd || openbsd || dragonfly || solaris

package internal

import (
	"fmt"
	"os"
	"path/filepath"

	"golang.org/x/sys/unix"
)

// CreateAnonymousFile emulates memfd by creating a temp file, ftruncate, mmap-safe.
// We return the path so caller can unlink it after mapping (keeps it anonymous).
func CreateAnonymousFile(name string, size int) (int, string, error) {
	dir := os.TempDir() // on macOS there is no /dev/shm
	path := filepath.Join(dir, name)
	// Create with O_EXCL to avoid races; we will remove the directory entry later.
	fd, err := unix.Open(path, unix.O_CREAT|unix.O_EXCL|unix.O_RDWR, 0600)
	if err != nil {
		return -1, "", fmt.Errorf("open temp: %w", err)
	}
	if err := unix.Ftruncate(fd, int64(size)); err != nil {
		unix.Close(fd)
		_ = os.Remove(path)
		return -1, "", fmt.Errorf("ftruncate: %w", err)
	}
	return fd, path, nil
}
