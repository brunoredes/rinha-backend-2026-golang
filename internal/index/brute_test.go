package index_test

import (
	"math/rand/v2"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"rinha-backend-golang/m/v2/internal/index"
	"rinha-backend-golang/m/v2/internal/refdata"
)

const dim = 14

// buildFixture creates a small refs.f32 + labels.bits with deterministic
// content so tests can assert exact ordering.
func buildFixture(t *testing.T, n int) (*refdata.Vectors, *refdata.Labels) {
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

	rng := rand.New(rand.NewPCG(1, 2))
	row := make([]float32, dim)
	for i := 0; i < n; i++ {
		for j := 0; j < dim; j++ {
			row[j] = float32(rng.Float64())
		}
		if err := vw.Append(row); err != nil {
			t.Fatal(err)
		}
		if err := lw.Append(i%3 == 0); err != nil {
			t.Fatal(err)
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

	v, err := refdata.OpenVectors(vp, dim)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = v.Close() })
	l, err := refdata.OpenLabels(lp)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = l.Close() })
	return v, l
}

// naiveTopK is the obvious O(N log N) implementation used as the oracle.
func naiveTopK(query []float32, vecs *refdata.Vectors, lbls *refdata.Labels, k int) []index.Neighbor {
	n := int(vecs.Header.Count)
	all := make([]index.Neighbor, n)
	for i := 0; i < n; i++ {
		row := vecs.Row(uint64(i))
		var s float32
		for j := range row {
			d := query[j] - row[j]
			s += d * d
		}
		all[i] = index.Neighbor{Distance: s, Index: uint32(i), Fraud: lbls.Get(uint64(i))}
	}
	sort.Slice(all, func(i, j int) bool { return all[i].Distance < all[j].Distance })
	return all[:k]
}

func TestBrute_TopK_MatchesNaive(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		n, k int
	}{
		{"tiny_k1", 50, 1},
		{"tiny_k5", 50, 5},
		{"k_eq_n", 5, 5},
		{"k_gt_n", 3, 5},
		{"medium", 5000, 5},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			vecs, lbls := buildFixture(t, tc.n)
			b, err := index.NewBrute(vecs, lbls)
			if err != nil {
				t.Fatal(err)
			}
			rng := rand.New(rand.NewPCG(42, 99))
			q := make([]float32, dim)
			for j := range q {
				q[j] = float32(rng.Float64())
			}

			out := make([]index.Neighbor, tc.k)
			got, err := b.TopK(q, out)
			if err != nil {
				t.Fatalf("topk: %v", err)
			}

			want := naiveTopK(q, vecs, lbls, min(tc.k, tc.n))
			if len(got) != len(want) {
				t.Fatalf("len: got %d want %d", len(got), len(want))
			}
			// Distance reduction order differs (unrolled vs linear), so we
			// compare indices for ranking correctness and require distances
			// match within a small epsilon.
			const eps = 1e-5
			for i := range got {
				if got[i].Index != want[i].Index {
					t.Errorf("rank %d: got idx %d want %d", i, got[i].Index, want[i].Index)
				}
				diff := got[i].Distance - want[i].Distance
				if diff < -eps || diff > eps {
					t.Errorf("rank %d: dist drift got %v want %v", i, got[i].Distance, want[i].Distance)
				}
			}
		})
	}
}

func TestBrute_FraudScore(t *testing.T) {
	t.Parallel()
	vecs, lbls := buildFixture(t, 200)
	b, err := index.NewBrute(vecs, lbls)
	if err != nil {
		t.Fatal(err)
	}
	q := make([]float32, dim)
	for j := range q {
		q[j] = 0.5
	}
	scratch := make([]index.Neighbor, 5)
	score, err := b.FraudScore(q, scratch)
	if err != nil {
		t.Fatal(err)
	}
	if score < 0 || score > 1 {
		t.Fatalf("score out of range: %v", score)
	}
}

func TestBrute_DimMismatch(t *testing.T) {
	t.Parallel()
	vecs, lbls := buildFixture(t, 10)
	b, _ := index.NewBrute(vecs, lbls)
	out := make([]index.Neighbor, 5)
	if _, err := b.TopK(make([]float32, dim+1), out); err == nil {
		t.Fatal("expected error")
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
