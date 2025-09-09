package fragpigeon

import "sync"

type Destinations struct {
	LocalFragIDs []uint16
	RemoteSites  []uint16
}

type Router struct {
	mu     sync.RWMutex
	byBits map[uint16]map[uint64]*Destinations // bits -> concept -> dests
}

func NewRouter() *Router {
	return &Router{
		byBits: make(map[uint16]map[uint64]*Destinations),
	}
}

func (r *Router) Add(bits uint16, concept uint64, locals []uint16, remotes []uint16) {
	r.mu.Lock()
	defer r.mu.Unlock()

	// lazy init
	if r.byBits == nil {
		r.byBits = make(map[uint16]map[uint64]*Destinations)
	}
	row := r.byBits[bits]
	if row == nil {
		row = make(map[uint64]*Destinations)
		r.byBits[bits] = row
	}
	d := row[concept]
	if d == nil {
		d = &Destinations{}
		row[concept] = d
	}
	if len(locals) > 0 {
		d.LocalFragIDs = appendUniqueU16(d.LocalFragIDs, locals...)
	}
	if len(remotes) > 0 {
		d.RemoteSites = appendUniqueU16(d.RemoteSites, remotes...)
	}
}

func (r *Router) Remove(bits uint16, concept uint64) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.byBits == nil {
		return
	}
	if row, ok := r.byBits[bits]; ok {
		delete(row, concept)
		if len(row) == 0 {
			delete(r.byBits, bits)
		}
	}
}

func (r *Router) DestinationsBy(bits uint16, concept uint64) *Destinations {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if r.byBits == nil {
		return nil
	}
	if row, ok := r.byBits[bits]; ok {
		return row[concept]
	}
	return nil
}

func (r *Router) Destinations(h Header) *Destinations {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if r.byBits == nil {
		return nil
	}
	if row, ok := r.byBits[h.ConceptBits]; ok {
		return row[h.ConceptID]
	}
	return nil
}

func appendUniqueU16(dst []uint16, src ...uint16) []uint16 {
	seen := make(map[uint16]struct{}, len(dst))
	for _, v := range dst {
		seen[v] = struct{}{}
	}
	for _, v := range src {
		if _, ok := seen[v]; !ok {
			dst = append(dst, v)
			seen[v] = struct{}{}
		}
	}
	return dst
}
