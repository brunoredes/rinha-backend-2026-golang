package ivf_test

import (
	"math/rand/v2"
	"os"
	"testing"

	"rinha-backend-golang/m/v2/internal/index"
	"rinha-backend-golang/m/v2/internal/ivf"
	"rinha-backend-golang/m/v2/internal/refdata"
)

// BenchmarkIVF_Search_3M measures pure search latency against the real
// pre-built IVF index in /tmp/ivf. Skipped when the index isn't present
// so unit-test runs don't require it.
func BenchmarkIVF_Search_3M(b *testing.B) {
	const indexDir = "/tmp/ivf"
	if _, err := os.Stat(indexDir); err != nil {
		b.Skipf("ivf index missing at %s; build with cmd/buildindex first", indexDir)
	}

	ix, err := ivf.Open(indexDir, dim)
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { _ = ix.Close() })

	s, err := ivf.NewSearcher(ix, ivf.SearchParams{NProbe: 8})
	if err != nil {
		b.Fatal(err)
	}

	rng := rand.New(rand.NewPCG(1, 2))
	q := make([]float32, dim)
	for j := range q {
		q[j] = float32(rng.Float64())
	}
	out := make([]index.Neighbor, 5)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := s.TopK(q, out); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkBrute_Search_3M is the apples-to-apples comparison.
func BenchmarkBrute_Search_3M(b *testing.B) {
	const refsPath = "/tmp/refs.f32"
	if _, err := os.Stat(refsPath); err != nil {
		b.Skipf("refs missing at %s", refsPath)
	}

	vecs, err := refdata.OpenVectors(refsPath, dim)
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { _ = vecs.Close() })
	lbls, err := refdata.OpenLabels("/tmp/labels.bits")
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { _ = lbls.Close() })

	bsr, err := index.NewBrute(vecs, lbls)
	if err != nil {
		b.Fatal(err)
	}

	rng := rand.New(rand.NewPCG(1, 2))
	q := make([]float32, dim)
	for j := range q {
		q[j] = float32(rng.Float64())
	}
	out := make([]index.Neighbor, 5)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := bsr.TopK(q, out); err != nil {
			b.Fatal(err)
		}
	}
}
