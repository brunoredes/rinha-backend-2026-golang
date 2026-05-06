//go:build !amd64

package index

func distancesBlock(q []float32, refs []float32, n int, out []float32) {
	distancesBlockScalar(q, refs, n, out)
}

// DistancesBlock is the exported entry point. See dist_amd64.go.
func DistancesBlock(q []float32, refs []float32, n int, out []float32) {
	distancesBlock(q, refs, n, out)
}
