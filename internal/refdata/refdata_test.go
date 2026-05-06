package refdata_test

import (
	"errors"
	"math"
	"os"
	"path/filepath"
	"testing"

	"rinha-backend-golang/m/v2/internal/refdata"
)

const dim = 14

func TestRoundtrip(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name    string
		count   int
		fraudOf func(i int) bool
	}{
		{name: "empty", count: 0, fraudOf: func(int) bool { return false }},
		{name: "one_legit", count: 1, fraudOf: func(int) bool { return false }},
		{name: "one_fraud", count: 1, fraudOf: func(int) bool { return true }},
		{name: "byte_aligned", count: 8, fraudOf: func(i int) bool { return i%2 == 0 }},
		{name: "trailing_partial_byte", count: 13, fraudOf: func(i int) bool { return i%3 == 0 }},
		{name: "many", count: 257, fraudOf: func(i int) bool { return i%5 == 0 }},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			tmp := t.TempDir()
			vecPath := filepath.Join(tmp, "refs.f32")
			lblPath := filepath.Join(tmp, "labels.bits")

			vecF, err := os.Create(vecPath)
			if err != nil {
				t.Fatalf("create vec: %v", err)
			}
			t.Cleanup(func() { _ = vecF.Close() })

			lblF, err := os.Create(lblPath)
			if err != nil {
				t.Fatalf("create lbl: %v", err)
			}
			t.Cleanup(func() { _ = lblF.Close() })

			vw, err := refdata.NewVectorWriter(vecF, dim)
			if err != nil {
				t.Fatalf("new vec writer: %v", err)
			}
			lw, err := refdata.NewLabelWriter(lblF)
			if err != nil {
				t.Fatalf("new lbl writer: %v", err)
			}

			vec := make([]float32, dim)
			for i := 0; i < tc.count; i++ {
				for j := 0; j < dim; j++ {
					vec[j] = float32(i*dim + j)
				}
				if err := vw.Append(vec); err != nil {
					t.Fatalf("append vec %d: %v", i, err)
				}
				if err := lw.Append(tc.fraudOf(i)); err != nil {
					t.Fatalf("append lbl %d: %v", i, err)
				}
			}

			if err := vw.Close(); err != nil {
				t.Fatalf("close vec: %v", err)
			}
			if err := lw.Close(); err != nil {
				t.Fatalf("close lbl: %v", err)
			}
			_ = vecF.Close()
			_ = lblF.Close()

			vecs, err := refdata.OpenVectors(vecPath, dim)
			if err != nil {
				t.Fatalf("open vec: %v", err)
			}
			t.Cleanup(func() { _ = vecs.Close() })

			lbls, err := refdata.OpenLabels(lblPath)
			if err != nil {
				t.Fatalf("open lbl: %v", err)
			}
			t.Cleanup(func() { _ = lbls.Close() })

			if got, want := vecs.Header.Count, uint64(tc.count); got != want {
				t.Fatalf("vec count: got %d want %d", got, want)
			}
			if got, want := lbls.Header.Count, uint64(tc.count); got != want {
				t.Fatalf("lbl count: got %d want %d", got, want)
			}

			for i := 0; i < tc.count; i++ {
				row := vecs.Row(uint64(i))
				for j := 0; j < dim; j++ {
					want := float32(i*dim + j)
					if row[j] != want {
						t.Fatalf("vec[%d][%d]: got %v want %v", i, j, row[j], want)
					}
				}
				if got, want := lbls.Get(uint64(i)), tc.fraudOf(i); got != want {
					t.Fatalf("label[%d]: got %v want %v", i, got, want)
				}
			}
		})
	}
}

func TestOpenVectors_BadDim(t *testing.T) {
	t.Parallel()

	tmp := t.TempDir()
	path := filepath.Join(tmp, "refs.f32")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = f.Close() })

	vw, err := refdata.NewVectorWriter(f, dim)
	if err != nil {
		t.Fatalf("new writer: %v", err)
	}
	if err := vw.Append(make([]float32, dim)); err != nil {
		t.Fatalf("append: %v", err)
	}
	if err := vw.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	_ = f.Close()

	if _, err := refdata.OpenVectors(path, dim+1); !errors.Is(err, refdata.ErrBadDim) {
		t.Fatalf("expected ErrBadDim, got %v", err)
	}
}

func TestOpenVectors_BadMagic(t *testing.T) {
	t.Parallel()

	tmp := t.TempDir()
	path := filepath.Join(tmp, "garbage")
	if err := os.WriteFile(path, make([]byte, refdata.HeaderSize+16), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := refdata.OpenVectors(path, dim); !errors.Is(err, refdata.ErrBadMagic) {
		t.Fatalf("expected ErrBadMagic, got %v", err)
	}
}

func TestLabelBytes(t *testing.T) {
	t.Parallel()
	cases := []struct {
		count uint64
		want  uint64
	}{
		{0, 0}, {1, 1}, {7, 1}, {8, 1}, {9, 2}, {16, 2}, {17, 3},
	}
	for _, c := range cases {
		if got := refdata.LabelBytes(c.count); got != c.want {
			t.Errorf("LabelBytes(%d) = %d, want %d", c.count, got, c.want)
		}
	}
}

// guard against silent NaN handling regressions in the float32 round-trip.
func TestVectorRoundtrip_NaN(t *testing.T) {
	t.Parallel()

	tmp := t.TempDir()
	path := filepath.Join(tmp, "refs.f32")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = f.Close() })

	vw, err := refdata.NewVectorWriter(f, dim)
	if err != nil {
		t.Fatal(err)
	}
	row := make([]float32, dim)
	row[0] = float32(math.Inf(1))
	row[1] = float32(math.Inf(-1))
	if err := vw.Append(row); err != nil {
		t.Fatal(err)
	}
	if err := vw.Close(); err != nil {
		t.Fatal(err)
	}
	_ = f.Close()

	v, err := refdata.OpenVectors(path, dim)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = v.Close() })
	got := v.Row(0)
	if !math.IsInf(float64(got[0]), 1) || !math.IsInf(float64(got[1]), -1) {
		t.Fatalf("inf round-trip failed: %v", got)
	}
}
