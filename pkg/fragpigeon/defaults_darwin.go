//go:build darwin || freebsd || netbsd || openbsd || dragonfly || solaris

package fragpigeon

const DefaultCOIShmPath = "/tmp/fragmind.coi.local"

func defaultLOADir() string { return "/tmp" }
