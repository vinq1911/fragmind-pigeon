//go:build linux

package internal

import (
	"fmt"

	"golang.org/x/sys/unix"
)

// CreateAnonymousFile returns (fd, unlinkPath, error).
// unlinkPath is empty on Linux memfd (nothing to unlink).
func CreateAnonymousFile(name string, size int) (int, string, error) {
	fd, err := unix.MemfdCreate(name, unix.MFD_CLOEXEC)
	if err != nil {
		return -1, "", fmt.Errorf("memfd_create: %w", err)
	}
	if err := unix.Ftruncate(fd, int64(size)); err != nil {
		unix.Close(fd)
		return -1, "", fmt.Errorf("ftruncate: %w", err)
	}
	return fd, "", nil
}
