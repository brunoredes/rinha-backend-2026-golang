package index

import (
	"math"
	"math/rand/v2"
	"testing"
)

const benchDim = VectorDim

// padQuery copies a 14-float query into a 16-float SIMD-ready buffer.
func padQuery(q [VectorDim]float32) []float32 {
	out := make([]float32, QueryStride)
	copy(out, q[:])
	return out
}

func randomVectors(t testing.TB, n int, seed uint64) []float32 {
	t.Helper()
	rng := rand.New(rand.NewPCG(seed, seed+1))
	// Allocate count*dim + tail pad lanes so the SIMD kernel's 16-float
	// load on the last vector stays in-bounds (mirrors the file format).
	v := make([]float32, n*VectorDim+QueryStride)
	for i := 0; i < n*VectorDim; i++ {
		v[i] = float32(rng.Float64())
	}
	return v
}

func TestDistancesBlock_AVX2MatchesScalar(t *testing.T) {
	t.Parallel()
	if !hasAVX2FMA {
		t.Skip("AVX2+FMA not available on this host")
	}

	const n = 257 // exercise the non-multiple-of-block tail
	refs := randomVectors(t, n, 1)
	var rawQ [VectorDim]float32
	rng := rand.New(rand.NewPCG(7, 11))
	for i := range rawQ {
		rawQ[i] = float32(rng.Float64())
	}
	q := padQuery(rawQ)

	asmOut := make([]float32, n)
	scOut := make([]float32, n)
	dist14SquaredAVX2Block(&q[0], &refs[0], n, &asmOut[0])
	distancesBlockScalar(q, refs, n, scOut)

	const eps = 1e-5
	for i := 0; i < n; i++ {
		diff := float64(asmOut[i] - scOut[i])
		if math.Abs(diff) > eps {
			t.Fatalf("ref %d: asm=%v scalar=%v", i, asmOut[i], scOut[i])
		}
	}
}

func TestDistancesBlock_KnownVectors(t *testing.T) {
	t.Parallel()

	// Two refs: one identical to the query (distance 0), one offset by 1
	// in every lane (distance == 14).
	q := make([]float32, QueryStride)
	for i := 0; i < VectorDim; i++ {
		q[i] = 0.5
	}
	refs := make([]float32, 2*VectorDim+QueryStride)
	copy(refs, q[:VectorDim])
	for i := 0; i < VectorDim; i++ {
		refs[VectorDim+i] = q[i] + 1
	}

	out := make([]float32, 2)
	distancesBlock(q, refs, 2, out)

	if out[0] != 0 {
		t.Fatalf("identical: got %v want 0", out[0])
	}
	if math.Abs(float64(out[1]-float32(VectorDim))) > 1e-5 {
		t.Fatalf("offset-1: got %v want %v", out[1], float32(VectorDim))
	}
}

func BenchmarkDistancesBlock_3M(b *testing.B) {
	const n = 3_000_000
	refs := randomVectors(b, n, 42)
	var rawQ [VectorDim]float32
	for i := range rawQ {
		rawQ[i] = 0.5
	}
	q := padQuery(rawQ)
	out := make([]float32, n)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		distancesBlock(q, refs, n, out)
	}
}

func BenchmarkDistancesBlock_3M_Scalar(b *testing.B) {
	const n = 3_000_000
	refs := randomVectors(b, n, 42)
	var rawQ [VectorDim]float32
	for i := range rawQ {
		rawQ[i] = 0.5
	}
	q := padQuery(rawQ)
	out := make([]float32, n)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		distancesBlockScalar(q, refs, n, out)
	}
}
