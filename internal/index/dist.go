package index

// VectorDim is the fixed vector dimensionality for this codebase. Mirrored
// here (rather than imported from domain) to keep the index package free
// of domain-layer dependencies on its hot path.
const VectorDim = 14

// QueryStride is the per-query buffer length in float32 lanes used by the
// SIMD kernel. The first VectorDim lanes hold the query; the remainder is
// zero-padded so the asm kernel can load two YMM registers without
// over-reading random memory.
const QueryStride = 16

// distancesBlockScalar is the reference implementation: portable, used as
// the fallback on non-amd64 / non-AVX2 hosts and as the oracle for tests.
// q is a QueryStride-long buffer with q[VectorDim:] == 0.
func distancesBlockScalar(q []float32, refs []float32, n int, out []float32) {
	for i := 0; i < n; i++ {
		base := i * VectorDim
		var s0, s1, s2, s3 float32
		// Manual 4-way unroll of 14-dim subtract+square. Same shape as
		// the original Brute.squaredL2 so behavior matches lockstep.
		j := 0
		for ; j+4 <= VectorDim; j += 4 {
			d0 := q[j] - refs[base+j]
			d1 := q[j+1] - refs[base+j+1]
			d2 := q[j+2] - refs[base+j+2]
			d3 := q[j+3] - refs[base+j+3]
			s0 += d0 * d0
			s1 += d1 * d1
			s2 += d2 * d2
			s3 += d3 * d3
		}
		s := s0 + s1 + s2 + s3
		for ; j < VectorDim; j++ {
			d := q[j] - refs[base+j]
			s += d * d
		}
		out[i] = s
	}
}
