// Command buildindex turns the preprocessed refs.f32 + labels.bits into an
// IVF index (centroids + reordered refs/labels + cell offsets) under -out.
// Run once at image build time.
package main

import (
	"flag"
	"fmt"
	"log/slog"
	"math/rand/v2"
	"os"
	"time"

	"rinha-backend-golang/m/v2/internal/domain"
	"rinha-backend-golang/m/v2/internal/ivf"
	"rinha-backend-golang/m/v2/internal/refdata"
)

type config struct {
	inVectors  string
	inLabels   string
	out        string
	k          int
	iters      int
	sampleSize int
	seed       uint64
}

func parseFlags() config {
	var c config
	flag.StringVar(&c.inVectors, "in-vectors", "data/refs.f32", "preprocessed refs.f32")
	flag.StringVar(&c.inLabels, "in-labels", "data/labels.bits", "preprocessed labels.bits")
	flag.StringVar(&c.out, "out", "data/ivf", "output directory for IVF artifacts")
	flag.IntVar(&c.k, "k", 2048, "number of cells")
	flag.IntVar(&c.iters, "iters", 10, "k-means iterations")
	flag.IntVar(&c.sampleSize, "sample", 200_000, "training sample size")
	flag.Uint64Var(&c.seed, "seed", 1, "rng seed")
	flag.Parse()
	return c
}

func main() {
	cfg := parseFlags()
	logger := slog.New(slog.NewJSONHandler(os.Stderr, nil))
	if err := run(cfg, logger); err != nil {
		logger.Error("buildindex failed", "err", err)
		os.Exit(1)
	}
}

func run(cfg config, logger *slog.Logger) error {
	start := time.Now()

	vecs, err := refdata.OpenVectors(cfg.inVectors, domain.VectorDimensions)
	if err != nil {
		return fmt.Errorf("open vectors: %w", err)
	}
	defer vecs.Close()
	lbls, err := refdata.OpenLabels(cfg.inLabels)
	if err != nil {
		return fmt.Errorf("open labels: %w", err)
	}
	defer lbls.Close()

	count := int(vecs.Header.Count)
	if cfg.sampleSize > count {
		cfg.sampleSize = count
	}
	dim := int(vecs.Header.Dim)

	logger.Info("training",
		"k", cfg.k,
		"iters", cfg.iters,
		"sample", cfg.sampleSize,
		"total_refs", count,
	)

	sample := drawSample(vecs.Data, count, cfg.sampleSize, dim, cfg.seed)

	centroids, err := ivf.KMeans(sample, ivf.TrainParams{
		K:      cfg.k,
		Dim:    dim,
		Iters:  cfg.iters,
		Seed:   cfg.seed,
		Logger: logger,
	})
	if err != nil {
		return fmt.Errorf("kmeans: %w", err)
	}
	logger.Info("training done", "elapsed", time.Since(start).String())

	if err := ivf.Build(ivf.BuildParams{
		Centroids: centroids,
		Vectors:   vecs,
		Labels:    lbls,
		OutDir:    cfg.out,
		K:         cfg.k,
		Dim:       dim,
		Logger:    logger,
	}); err != nil {
		return fmt.Errorf("build: %w", err)
	}

	logger.Info("buildindex complete",
		"k", cfg.k,
		"out", cfg.out,
		"elapsed", time.Since(start).String(),
	)
	return nil
}

// drawSample copies sampleSize random rows from src (count rows of dim
// floats) into a fresh flat buffer. Reservoir sampling would be more
// efficient if we were streaming, but src is mmap'd so random access is
// fine and the simpler index-based approach is clearer.
func drawSample(src []float32, count, sampleSize, dim int, seed uint64) []float32 {
	rng := rand.New(rand.NewPCG(seed, seed^0xA5A5A5A5A5A5A5A5))
	out := make([]float32, sampleSize*dim)
	for i := 0; i < sampleSize; i++ {
		j := rng.IntN(count)
		copy(out[i*dim:(i+1)*dim], src[j*dim:(j+1)*dim])
	}
	return out
}
