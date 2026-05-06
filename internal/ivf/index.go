package ivf

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"unsafe"

	"golang.org/x/sys/unix"

	"rinha-backend-golang/m/v2/internal/index"
	"rinha-backend-golang/m/v2/internal/refdata"
)

// Index is the runtime IVF searcher: mmap'd centroids + reordered refs +
// per-cell offsets. All three blobs are kernel-page-cache-shared between
// API replicas (the very thing that lets us fit two instances in 350 MB).
type Index struct {
	Centroids *refdata.Vectors
	Vectors   *refdata.Vectors
	Labels    *refdata.Labels

	K       int
	Dim     int
	Count   uint64
	Offsets []uint32 // length K+1; aliases mmap region

	rawOffsets []byte // owned mmap to munmap on Close
}

// Open mmaps the four artifact files written by Build under dir.
func Open(dir string, dim uint32) (*Index, error) {
	cents, err := refdata.OpenVectors(filepath.Join(dir, "centroids.f32"), dim)
	if err != nil {
		return nil, fmt.Errorf("open centroids: %w", err)
	}
	vecs, err := refdata.OpenVectors(filepath.Join(dir, "refs.f32"), dim)
	if err != nil {
		_ = cents.Close()
		return nil, fmt.Errorf("open refs: %w", err)
	}
	lbls, err := refdata.OpenLabels(filepath.Join(dir, "labels.bits"))
	if err != nil {
		_ = cents.Close()
		_ = vecs.Close()
		return nil, fmt.Errorf("open labels: %w", err)
	}

	rawOff, offsets, k, count, err := openOffsets(filepath.Join(dir, "cell_offsets.u32"))
	if err != nil {
		_ = cents.Close()
		_ = vecs.Close()
		_ = lbls.Close()
		return nil, err
	}

	if uint64(k) != cents.Header.Count {
		_ = unix.Munmap(rawOff)
		_ = cents.Close()
		_ = vecs.Close()
		_ = lbls.Close()
		return nil, fmt.Errorf("%w: offsets K=%d != centroids count=%d", ErrBadOffsets, k, cents.Header.Count)
	}
	if count != vecs.Header.Count || count != lbls.Header.Count {
		_ = unix.Munmap(rawOff)
		_ = cents.Close()
		_ = vecs.Close()
		_ = lbls.Close()
		return nil, fmt.Errorf("%w: count mismatch offsets=%d vectors=%d labels=%d",
			ErrBadOffsets, count, vecs.Header.Count, lbls.Header.Count)
	}
	if uint64(offsets[k]) != count {
		_ = unix.Munmap(rawOff)
		_ = cents.Close()
		_ = vecs.Close()
		_ = lbls.Close()
		return nil, fmt.Errorf("%w: last offset %d != count %d", ErrBadOffsets, offsets[k], count)
	}

	return &Index{
		Centroids:  cents,
		Vectors:    vecs,
		Labels:     lbls,
		K:          int(k),
		Dim:        int(dim),
		Count:      count,
		Offsets:    offsets,
		rawOffsets: rawOff,
	}, nil
}

// Close releases all mmaps.
func (ix *Index) Close() error {
	var firstErr error
	if err := ix.Centroids.Close(); err != nil && firstErr == nil {
		firstErr = err
	}
	if err := ix.Vectors.Close(); err != nil && firstErr == nil {
		firstErr = err
	}
	if err := ix.Labels.Close(); err != nil && firstErr == nil {
		firstErr = err
	}
	if ix.rawOffsets != nil {
		if err := unix.Munmap(ix.rawOffsets); err != nil && firstErr == nil {
			firstErr = err
		}
		ix.rawOffsets = nil
		ix.Offsets = nil
	}
	return firstErr
}

// openOffsets mmaps cell_offsets.u32 and returns the raw mmap, an aliasing
// []uint32 view, and the parsed K + count.
func openOffsets(path string) ([]byte, []uint32, uint32, uint64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, 0, 0, fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return nil, nil, 0, 0, fmt.Errorf("stat %s: %w", path, err)
	}
	if st.Size() < HeaderSize {
		return nil, nil, 0, 0, fmt.Errorf("%w: %s is %d bytes", ErrShortHeader, path, st.Size())
	}
	raw, err := unix.Mmap(int(f.Fd()), 0, int(st.Size()), unix.PROT_READ, unix.MAP_SHARED)
	if err != nil {
		return nil, nil, 0, 0, fmt.Errorf("mmap %s: %w", path, err)
	}
	h, err := parseOffsetsHeader(raw)
	if err != nil {
		_ = unix.Munmap(raw)
		return nil, nil, 0, 0, fmt.Errorf("%s: %w", path, err)
	}
	wantBytes := uint64(HeaderSize) + uint64(h.K+1)*4
	if uint64(len(raw)) < wantBytes {
		_ = unix.Munmap(raw)
		return nil, nil, 0, 0, fmt.Errorf("%w: file %d < want %d", ErrBadSize, len(raw), wantBytes)
	}
	payload := raw[HeaderSize:]
	view := unsafe.Slice((*uint32)(unsafe.Pointer(&payload[0])), int(h.K)+1)
	return raw, view, h.K, h.Count, nil
}

// SearchParams configures TopK / FraudScore. NProbe is the number of cells
// the searcher visits per query; higher = more recall, lower = lower latency.
type SearchParams struct {
	NProbe int
}

// Searcher exposes the TopK / FraudScore interface that mirrors index.Brute
// so server.Handler can hold either one.
type Searcher struct {
	ix *Index
	p  SearchParams
}

// NewSearcher binds an Index to a fixed nprobe.
func NewSearcher(ix *Index, p SearchParams) (*Searcher, error) {
	if ix == nil {
		return nil, fmt.Errorf("ivf: nil index")
	}
	if p.NProbe <= 0 || p.NProbe > ix.K {
		return nil, fmt.Errorf("ivf: NProbe must be in (0, %d] (got %d)", ix.K, p.NProbe)
	}
	return &Searcher{ix: ix, p: p}, nil
}

// TopK fills out[:k] with the k nearest neighbors of query, sorted ascending
// by distance. Algorithm:
//  1. Compute distances to all K centroids with the AVX2 kernel.
//  2. Pick the NProbe closest cells via a small max-tracker.
//  3. For each chosen cell, run the AVX2 kernel over its members and merge
//     into the running k-element max-tracker.
func (s *Searcher) TopK(query []float32, out []index.Neighbor) ([]index.Neighbor, error) {
	if len(query) != s.ix.Dim {
		return nil, fmt.Errorf("%w: query has %d dims want %d",
			refdata.ErrBadDim, len(query), s.ix.Dim)
	}
	if len(out) == 0 {
		return out[:0], index.ErrShortOut
	}
	k := len(out)
	if uint64(k) > s.ix.Count {
		k = int(s.ix.Count)
	}

	for i := range out {
		out[i] = index.Neighbor{Distance: float32(math.MaxFloat32)}
	}

	var qbuf [index.QueryStride]float32
	copy(qbuf[:], query)

	// Centroid distances.
	centDists := make([]float32, s.ix.K)
	index.DistancesBlock(qbuf[:], s.ix.Centroids.Data, s.ix.K, centDists)

	// Top-NProbe cells (smallest distances).
	probes := pickTopCells(centDists, s.p.NProbe)

	// Scratch for cell scans. Largest cell is bounded by Count so we cap at
	// a generous block; if any cell is larger, we chunk.
	const blockSize = 1024
	var dists [blockSize]float32

	maxIdx := 0
	for _, c := range probes {
		from := s.ix.Offsets[c]
		to := s.ix.Offsets[c+1]
		for off := from; off < to; off += blockSize {
			n := int(to - off)
			if n > blockSize {
				n = blockSize
			}
			index.DistancesBlock(qbuf[:], s.ix.Vectors.Data[int(off)*s.ix.Dim:], n, dists[:n])
			for i := 0; i < n; i++ {
				d := dists[i]
				if d >= out[maxIdx].Distance {
					continue
				}
				idx := off + uint32(i)
				out[maxIdx] = index.Neighbor{
					Distance: d,
					Index:    idx,
					Fraud:    s.ix.Labels.Get(uint64(idx)),
				}
				maxIdx = 0
				for j := 1; j < k; j++ {
					if out[j].Distance > out[maxIdx].Distance {
						maxIdx = j
					}
				}
			}
		}
	}

	insertionSortByDistance(out[:k])
	return out[:k], nil
}

// FraudScore is a thin wrapper that returns frauds/k after TopK.
func (s *Searcher) FraudScore(query []float32, scratch []index.Neighbor) (float32, error) {
	hits, err := s.TopK(query, scratch)
	if err != nil {
		return 0, err
	}
	frauds := 0
	for _, n := range hits {
		if n.Fraud {
			frauds++
		}
	}
	return float32(frauds) / float32(len(hits)), nil
}

// pickTopCells returns the indices of the nprobe smallest entries in dists.
// Returned slice is unsorted; the search loop doesn't care about probe
// order. nprobe is small (8-16 typical), so an O(K*nprobe) max-tracker is
// faster than a heap.
func pickTopCells(dists []float32, nprobe int) []uint32 {
	if nprobe > len(dists) {
		nprobe = len(dists)
	}
	out := make([]uint32, nprobe)
	tops := make([]float32, nprobe)
	for i := range tops {
		tops[i] = float32(math.MaxFloat32)
	}
	maxIdx := 0
	for i, d := range dists {
		if d >= tops[maxIdx] {
			continue
		}
		tops[maxIdx] = d
		out[maxIdx] = uint32(i)
		maxIdx = 0
		for j := 1; j < nprobe; j++ {
			if tops[j] > tops[maxIdx] {
				maxIdx = j
			}
		}
	}
	return out
}

// insertionSortByDistance sorts in-place ascending. k is small (<= ~16).
func insertionSortByDistance(s []index.Neighbor) {
	for i := 1; i < len(s); i++ {
		x := s[i]
		j := i - 1
		for j >= 0 && s[j].Distance > x.Distance {
			s[j+1] = s[j]
			j--
		}
		s[j+1] = x
	}
}

