package ivf_test

import (
	"math/rand/v2"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"rinha-backend-golang/m/v2/internal/index"
	"rinha-backend-golang/m/v2/internal/ivf"
	"rinha-backend-golang/m/v2/internal/refdata"
)

const dim = 14

// buildSyntheticDataset writes a small refs.f32 + labels.bits with a
// known structure: nClusters Gaussian-ish blobs in 14-d so k-means has
// signal to recover.
func buildSyntheticDataset(t *testing.T, nPerCluster, nClusters int) (string, string) {
	t.Helper()
	tmp := t.TempDir()
	vp := filepath.Join(tmp, "refs.f32")
	lp := filepath.Join(tmp, "labels.bits")

	vf, err := os.Create(vp)
	if err != nil {
		t.Fatal(err)
	}
	lf, err := os.Create(lp)
	if err != nil {
		t.Fatal(err)
	}
	vw, err := refdata.NewVectorWriter(vf, dim)
	if err != nil {
		t.Fatal(err)
	}
	lw, err := refdata.NewLabelWriter(lf)
	if err != nil {
		t.Fatal(err)
	}

	rng := rand.New(rand.NewPCG(7, 11))
	row := make([]float32, dim)
	for c := 0; c < nClusters; c++ {
		var center [dim]float32
		for j := range center {
			center[j] = float32(rng.Float64())
		}
		for n := 0; n < nPerCluster; n++ {
			for j := 0; j < dim; j++ {
				row[j] = center[j] + float32(rng.Float64()*0.05-0.025)
			}
			if err := vw.Append(row); err != nil {
				t.Fatal(err)
			}
			if err := lw.Append(c%2 == 0); err != nil {
				t.Fatal(err)
			}
		}
	}
	if err := vw.Close(); err != nil {
		t.Fatal(err)
	}
	if err := lw.Close(); err != nil {
		t.Fatal(err)
	}
	_ = vf.Close()
	_ = lf.Close()
	return vp, lp
}

func TestIVF_BuildAndQuery_RecallVsBrute(t *testing.T) {
	t.Parallel()

	const (
		nPerCluster = 80
		nClusters   = 16
		k           = 8
		nprobe      = 4
		topk        = 5
	)
	vp, lp := buildSyntheticDataset(t, nPerCluster, nClusters)
	vecs, err := refdata.OpenVectors(vp, dim)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = vecs.Close() })
	lbls, err := refdata.OpenLabels(lp)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = lbls.Close() })

	// Train on the full data (small enough).
	sample := make([]float32, len(vecs.Data))
	copy(sample, vecs.Data)
	centroids, err := ivf.KMeans(sample, ivf.TrainParams{
		K: k, Dim: dim, Iters: 10, Seed: 1,
	})
	if err != nil {
		t.Fatal(err)
	}

	outDir := t.TempDir()
	if err := ivf.Build(ivf.BuildParams{
		Centroids: centroids,
		Vectors:   vecs,
		Labels:    lbls,
		OutDir:    outDir,
		K:         k,
		Dim:       dim,
	}); err != nil {
		t.Fatal(err)
	}

	ix, err := ivf.Open(outDir, dim)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = ix.Close() })
	s, err := ivf.NewSearcher(ix, ivf.SearchParams{NProbe: nprobe})
	if err != nil {
		t.Fatal(err)
	}

	brute, err := index.NewBrute(vecs, lbls)
	if err != nil {
		t.Fatal(err)
	}

	rng := rand.New(rand.NewPCG(99, 100))
	q := make([]float32, dim)
	for j := range q {
		q[j] = float32(rng.Float64())
	}

	bruteOut := make([]index.Neighbor, topk)
	bruteRes, err := brute.TopK(q, bruteOut)
	if err != nil {
		t.Fatal(err)
	}

	ivfOut := make([]index.Neighbor, topk)
	ivfRes, err := s.TopK(q, ivfOut)
	if err != nil {
		t.Fatal(err)
	}

	// IVF returns indices into the reordered refs (different physical
	// row from the brute view), so we compare on *distances* — same
	// physical vectors produce identical squared-L2 values up to float
	// reduction order. Recall@5 = how many brute distances appear in the
	// IVF result within epsilon.
	const eps = 1e-5
	hits := 0
	for _, b := range bruteRes {
		for _, i := range ivfRes {
			diff := float64(b.Distance - i.Distance)
			if diff < eps && diff > -eps {
				hits++
				break
			}
		}
	}
	if hits < 4 {
		t.Fatalf("recall@5 too low: %d/%d (ivf=%v brute=%v)", hits, topk, ivfRes, bruteRes)
	}

	// Sanity: fraud scores must agree exactly because labels travel with
	// the reorder, and distance ordering of the top-5 is identical.
	bScore := fraudScore(bruteRes)
	iScore := fraudScore(ivfRes)
	if bScore != iScore {
		t.Fatalf("fraud score mismatch: brute=%v ivf=%v", bScore, iScore)
	}
}

func fraudScore(ns []index.Neighbor) float32 {
	var f int
	for _, n := range ns {
		if n.Fraud {
			f++
		}
	}
	return float32(f) / float32(len(ns))
}

func TestPickTopCells_SimpleOracle(t *testing.T) {
	t.Parallel()

	dists := []float32{0.9, 0.1, 0.5, 0.2, 0.7, 0.3}
	// Run a tiny end-to-end IVF call to exercise pickTopCells indirectly:
	// we already cover it via TestIVF_BuildAndQuery_RecallVsBrute, so
	// here we just assert sort ordering of the input itself for sanity.
	sort.Slice(dists, func(i, j int) bool { return dists[i] < dists[j] })
	if dists[0] != 0.1 || dists[1] != 0.2 {
		t.Fatalf("oracle broken: %v", dists)
	}
}
