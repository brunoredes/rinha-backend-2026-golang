package ivf

import (
	"errors"
	"fmt"
	"log/slog"
	"math"
	"math/rand/v2"

	"rinha-backend-golang/m/v2/internal/index"
)

// TrainParams configures KMeans. Declared before the consumer per CS-6.
type TrainParams struct {
	K        int
	Dim      int
	Iters    int
	Seed     uint64
	Logger   *slog.Logger
}

// KMeans runs Lloyd's algorithm with k-means++ initialization on samples.
// samples is a flat row-major buffer of len = nSamples * Dim. Returns
// the centroids as a flat row-major buffer of len = K * Dim.
//
// The implementation is straightforward: per iteration we (a) assign each
// sample to the nearest centroid and (b) recompute centroids as the mean
// of their assigned samples. Empty clusters are re-seeded from a random
// sample to keep K constant.
func KMeans(samples []float32, p TrainParams) ([]float32, error) {
	if p.K <= 0 {
		return nil, fmt.Errorf("kmeans: K must be > 0 (got %d)", p.K)
	}
	if p.Dim <= 0 || p.Dim > index.VectorDim {
		return nil, fmt.Errorf("kmeans: Dim must be in (0, %d] (got %d)", index.VectorDim, p.Dim)
	}
	if len(samples)%p.Dim != 0 {
		return nil, errors.New("kmeans: samples length not multiple of dim")
	}
	n := len(samples) / p.Dim
	if n < p.K {
		return nil, fmt.Errorf("kmeans: need at least K=%d samples (got %d)", p.K, n)
	}
	if p.Iters <= 0 {
		p.Iters = 10
	}
	if p.Logger == nil {
		p.Logger = slog.Default()
	}

	rng := rand.New(rand.NewPCG(p.Seed, p.Seed^0x9E3779B97F4A7C15))

	centroids := initKMeansPP(samples, n, p.K, p.Dim, rng)

	// Persistent buffers — k-means is allocation-free on the hot path.
	assign := make([]uint32, n)
	sums := make([]float32, p.K*p.Dim)
	counts := make([]uint32, p.K)
	dists := make([]float32, p.K)

	var qbuf [index.QueryStride]float32

	for iter := 0; iter < p.Iters; iter++ {
		// (a) Assign each sample to its nearest centroid.
		var totalDist float64
		for i := 0; i < n; i++ {
			copyToQueryBuf(qbuf[:], samples[i*p.Dim:(i+1)*p.Dim], p.Dim)
			index.DistancesBlock(qbuf[:], centroids, p.K, dists)
			best := 0
			bestD := dists[0]
			for c := 1; c < p.K; c++ {
				if dists[c] < bestD {
					bestD = dists[c]
					best = c
				}
			}
			assign[i] = uint32(best)
			totalDist += float64(bestD)
		}

		// (b) Recompute centroids as the mean of assigned samples.
		for i := range sums {
			sums[i] = 0
		}
		for i := range counts {
			counts[i] = 0
		}
		for i := 0; i < n; i++ {
			c := assign[i]
			counts[c]++
			base := int(c) * p.Dim
			for d := 0; d < p.Dim; d++ {
				sums[base+d] += samples[i*p.Dim+d]
			}
		}
		empty := 0
		for c := 0; c < p.K; c++ {
			if counts[c] == 0 {
				// Re-seed from a random sample to keep K populated.
				src := rng.IntN(n)
				copy(centroids[c*p.Dim:(c+1)*p.Dim], samples[src*p.Dim:(src+1)*p.Dim])
				empty++
				continue
			}
			inv := 1.0 / float32(counts[c])
			base := c * p.Dim
			for d := 0; d < p.Dim; d++ {
				centroids[base+d] = sums[base+d] * inv
			}
		}

		p.Logger.Info("kmeans iter",
			"iter", iter,
			"avg_dist", totalDist/float64(n),
			"empty_cells", empty,
		)
	}

	return centroids, nil
}

// initKMeansPP picks K centroids using k-means++: the first center is a
// uniformly random sample; each subsequent center is sampled with weight
// proportional to D(x)^2 where D(x) is the squared distance from x to its
// nearest already-chosen center.
func initKMeansPP(samples []float32, n, k, dim int, rng *rand.Rand) []float32 {
	centroids := make([]float32, k*dim)
	first := rng.IntN(n)
	copy(centroids[0:dim], samples[first*dim:(first+1)*dim])

	// minD2[i] = squared distance from sample i to its nearest existing center.
	minD2 := make([]float32, n)
	for i := 0; i < n; i++ {
		minD2[i] = squaredL2Slice(samples[i*dim:(i+1)*dim], centroids[0:dim])
	}

	for c := 1; c < k; c++ {
		// Sample index proportional to minD2.
		var sum float64
		for _, d := range minD2 {
			sum += float64(d)
		}
		if sum == 0 {
			// All samples coincide with chosen centers — degenerate; pick random.
			pick := rng.IntN(n)
			copy(centroids[c*dim:(c+1)*dim], samples[pick*dim:(pick+1)*dim])
			continue
		}
		target := rng.Float64() * sum
		var acc float64
		pick := n - 1
		for i, d := range minD2 {
			acc += float64(d)
			if acc >= target {
				pick = i
				break
			}
		}
		copy(centroids[c*dim:(c+1)*dim], samples[pick*dim:(pick+1)*dim])

		// Update minD2 against the new center.
		newCenter := centroids[c*dim : (c+1)*dim]
		for i := 0; i < n; i++ {
			d := squaredL2Slice(samples[i*dim:(i+1)*dim], newCenter)
			if d < minD2[i] {
				minD2[i] = d
			}
		}
	}
	return centroids
}

// copyToQueryBuf copies up to QueryStride lanes from src into dst, zeroing
// any trailing lanes so the SIMD kernel sees a clean over-read region.
func copyToQueryBuf(dst, src []float32, dim int) {
	copy(dst, src[:dim])
	for i := dim; i < len(dst); i++ {
		dst[i] = 0
	}
}

// squaredL2Slice is a small portable scalar helper for k-means init only —
// not on the per-query hot path, so SIMD isn't worth the indirection.
func squaredL2Slice(a, b []float32) float32 {
	var s float32
	for i := range a {
		d := a[i] - b[i]
		s += d * d
	}
	if math.IsNaN(float64(s)) {
		return 0
	}
	return s
}
