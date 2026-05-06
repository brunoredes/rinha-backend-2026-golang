//go:build !amd64

package index

func distancesBlock(q []float32, refs []float32, n int, out []float32) {
	distancesBlockScalar(q, refs, n, out)
}
