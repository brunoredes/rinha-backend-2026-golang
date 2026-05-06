// Package index implements vector similarity search over the preprocessed
// reference dataset. The Brute searcher is the baseline brute-force k-NN
// using squared Euclidean distance; it exists to validate correctness and
// give us a measurable starting point before SIMD/IVF land.
package index

import (
	"errors"
	"fmt"
	"math"

	"rinha-backend-golang/m/v2/internal/refdata"
)

// ErrShortOut is returned when TopK's out slice is smaller than k.
var ErrShortOut = errors.New("index: out slice shorter than k")

// Neighbor describes one search hit. Distance is squared L2 (no sqrt — we
// only need ordering).
type Neighbor struct {
	Distance float32
	Index    uint32
	Fraud    bool
}

// blockSize is how many distances we ask the SIMD kernel to compute per
// call before merging into the running top-K. 1024 floats fit easily in
// L1 (4 KiB) so the post-block top-K scan reads from cache.
const blockSize = 1024

// Brute scans every reference vector for each query. O(N*dim) per query.
type Brute struct {
	vecs *refdata.Vectors
	lbls *refdata.Labels
	dim  int
}

// NewBrute validates that vectors and labels agree on count.
func NewBrute(vecs *refdata.Vectors, lbls *refdata.Labels) (*Brute, error) {
	if vecs == nil || lbls == nil {
		return nil, errors.New("index: nil vectors or labels")
	}
	if vecs.Header.Count != lbls.Header.Count {
		return nil, fmt.Errorf("index: count mismatch vectors=%d labels=%d",
			vecs.Header.Count, lbls.Header.Count)
	}
	return &Brute{vecs: vecs, lbls: lbls, dim: int(vecs.Header.Dim)}, nil
}

// Count returns the number of indexed vectors.
func (b *Brute) Count() uint64 { return b.vecs.Header.Count }

// TopK fills out[:k] with the k nearest neighbors of query, sorted ascending
// by distance. The returned slice aliases out and has length k.
//
// Algorithm: pad the query into a 16-float buffer, hand blocks of refs to
// the SIMD kernel (or scalar fallback), then merge each block's distances
// into a k-element max-tracker. The early-exit `d >= currentMax` check
// skips the vast majority of refs once the tracker warms up; the post-
// replace max scan is O(k), negligible at k=5.
func (b *Brute) TopK(query []float32, out []Neighbor) ([]Neighbor, error) {
	if len(query) != b.dim {
		return nil, fmt.Errorf("%w: query has %d dims want %d",
			refdata.ErrBadDim, len(query), b.dim)
	}
	if len(out) == 0 {
		return out[:0], ErrShortOut
	}
	k := len(out)
	if uint64(k) > b.vecs.Header.Count {
		k = int(b.vecs.Header.Count)
	}

	for i := range out {
		out[i] = Neighbor{Distance: float32(math.MaxFloat32)}
	}

	// Pad the query so the SIMD kernel can load two YMMs.
	var qbuf [QueryStride]float32
	copy(qbuf[:], query)

	var dists [blockSize]float32
	data := b.vecs.Data
	count := int(b.vecs.Header.Count)
	maxIdx := 0

	for start := 0; start < count; start += blockSize {
		n := blockSize
		if start+n > count {
			n = count - start
		}
		distancesBlock(qbuf[:], data[start*b.dim:], n, dists[:n])
		for i := 0; i < n; i++ {
			d := dists[i]
			if d >= out[maxIdx].Distance {
				continue
			}
			out[maxIdx] = Neighbor{
				Distance: d,
				Index:    uint32(start + i),
				Fraud:    b.lbls.Get(uint64(start + i)),
			}
			maxIdx = 0
			for j := 1; j < k; j++ {
				if out[j].Distance > out[maxIdx].Distance {
					maxIdx = j
				}
			}
		}
	}

	insertionSortByDistance(out[:k])
	return out[:k], nil
}

// FraudScore returns the fraction of fraud-labeled neighbors among the
// k-nearest, exactly as required by the spec. The caller supplies a
// scratch slice of length >= k to avoid allocations on the hot path.
func (b *Brute) FraudScore(query []float32, scratch []Neighbor) (float32, error) {
	hits, err := b.TopK(query, scratch)
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

// insertionSortByDistance sorts in-place ascending. k is small (<= ~16).
func insertionSortByDistance(s []Neighbor) {
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
