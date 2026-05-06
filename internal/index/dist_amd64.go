//go:build amd64

package index

import "golang.org/x/sys/cpu"

// dist14SquaredAVX2Block is the assembly kernel. q must point to a buffer
// of 16 float32 with the last two lanes zeroed; refs points to the start
// of the (count * 14)-float reference blob; out receives one float32 per
// reference. Defined in dist_amd64.s.
func dist14SquaredAVX2Block(q, refs *float32, n int, out *float32)

// hasAVX2FMA is set at init time. We require both AVX2 (for VMOVUPS/
// VEXTRACTF128 on YMM) and FMA (for VFMADD231PS).
var hasAVX2FMA = cpu.X86.HasAVX2 && cpu.X86.HasFMA

// distancesBlock fills out[:n] with squared L2 distances between q and
// the n consecutive 14-float vectors at refs[0:n*14]. q must be a 16-
// float buffer (last two lanes zeroed). The function dispatches to the
// AVX2+FMA kernel when available and falls back to the scalar reference
// implementation otherwise.
func distancesBlock(q []float32, refs []float32, n int, out []float32) {
	if hasAVX2FMA {
		dist14SquaredAVX2Block(&q[0], &refs[0], n, &out[0])
		return
	}
	distancesBlockScalar(q, refs, n, out)
}
