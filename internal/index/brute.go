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
// Algorithm: maintain an unsorted array of k candidates plus the index of
// its current max. For each scanned reference, the early-exit check
// `dist < topMax` skips the vast majority once the heap warms up; the
// post-replace max scan is O(k), which is negligible at k=5.
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

	data := b.vecs.Data
	dim := b.dim
	maxIdx := 0

	for i := 0; i < int(b.vecs.Header.Count); i++ {
		d := squaredL2(query, data[i*dim:(i+1)*dim])
		if d >= out[maxIdx].Distance {
			continue
		}
		out[maxIdx] = Neighbor{
			Distance: d,
			Index:    uint32(i),
			Fraud:    b.lbls.Get(uint64(i)),
		}
		// Recompute the index of the new max.
		maxIdx = 0
		for j := 1; j < k; j++ {
			if out[j].Distance > out[maxIdx].Distance {
				maxIdx = j
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

// squaredL2 computes the squared Euclidean distance for two equal-length
// vectors. Manual unroll into 4 accumulators helps the compiler keep the
// FMA chain wide; for 14-dim queries this is ~5 ns on Haswell.
func squaredL2(a, b []float32) float32 {
	// len(a) == len(b) == dim is guaranteed by callers.
	var s0, s1, s2, s3 float32
	i := 0
	n := len(a)
	for ; i+4 <= n; i += 4 {
		d0 := a[i] - b[i]
		d1 := a[i+1] - b[i+1]
		d2 := a[i+2] - b[i+2]
		d3 := a[i+3] - b[i+3]
		s0 += d0 * d0
		s1 += d1 * d1
		s2 += d2 * d2
		s3 += d3 * d3
	}
	s := s0 + s1 + s2 + s3
	for ; i < n; i++ {
		d := a[i] - b[i]
		s += d * d
	}
	return s
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
