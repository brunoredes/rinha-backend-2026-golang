// Command api is the fraud-detection HTTP server.
//
// On startup it loads the preprocessed reference dataset (mmap), reads the
// normalization + MCC files, builds the brute-force searcher, and serves
// /ready and /fraud-score on the configured address.
package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"rinha-backend-golang/m/v2/internal/config"
	"rinha-backend-golang/m/v2/internal/detection"
	"rinha-backend-golang/m/v2/internal/domain"
	"rinha-backend-golang/m/v2/internal/index"
	"rinha-backend-golang/m/v2/internal/ivf"
	"rinha-backend-golang/m/v2/internal/refdata"
	"rinha-backend-golang/m/v2/internal/server"
)

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stderr, nil))
	if err := run(logger); err != nil {
		logger.Error("api fatal", "err", err)
		os.Exit(1)
	}
}

func run(logger *slog.Logger) error {
	cfg, err := config.Load()
	if err != nil {
		return err
	}

	startLoad := time.Now()

	searcher, indexCount, closer, err := openSearcher(cfg, logger)
	if err != nil {
		return err
	}
	defer closer()

	norm, err := detection.LoadNormalization(cfg.NormalizationPath)
	if err != nil {
		return err
	}
	mcc, err := detection.LoadMccRisk(cfg.MCCRiskPath)
	if err != nil {
		return err
	}
	vec, err := detection.NewVectorizer(norm, mcc)
	if err != nil {
		return err
	}

	h, err := server.New(server.Deps{
		Vectorizer: vec,
		Searcher:   searcher,
		K:          cfg.K,
		Threshold:  cfg.Threshold,
		Logger:     logger,
	})
	if err != nil {
		return err
	}
	h.SetReady(true)

	logger.Info("refdata loaded",
		"count", indexCount,
		"elapsed", time.Since(startLoad).String(),
	)

	srv := &http.Server{
		Addr:         cfg.Addr,
		Handler:      h.Routes(),
		ReadTimeout:  cfg.ReadTimeout,
		WriteTimeout: cfg.WriteTimeout,
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	errCh := make(chan error, 1)
	go func() {
		logger.Info("listening", "addr", cfg.Addr)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			errCh <- err
			return
		}
		errCh <- nil
	}()

	select {
	case <-ctx.Done():
		logger.Info("shutdown signal received")
	case err := <-errCh:
		return err
	}

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	h.SetReady(false)
	if err := srv.Shutdown(shutdownCtx); err != nil {
		return fmt.Errorf("shutdown: %w", err)
	}
	return nil
}

// openSearcher tries the IVF index first; if its directory is missing or
// any file inside it is malformed, it falls back to brute-force scanning
// the flat refs.f32 + labels.bits. The returned closer must be invoked
// before process exit.
func openSearcher(cfg config.API, logger *slog.Logger) (server.Searcher, uint64, func(), error) {
	if cfg.IVFDir != "" {
		if _, err := os.Stat(cfg.IVFDir); err == nil {
			ix, err := ivf.Open(cfg.IVFDir, domain.VectorDimensions)
			if err == nil {
				s, err := ivf.NewSearcher(ix, ivf.SearchParams{NProbe: cfg.NProbe})
				if err != nil {
					_ = ix.Close()
					return nil, 0, nil, fmt.Errorf("ivf searcher: %w", err)
				}
				logger.Info("using ivf searcher",
					"dir", cfg.IVFDir, "k", ix.K, "nprobe", cfg.NProbe, "count", ix.Count)
				return s, ix.Count, func() { _ = ix.Close() }, nil
			}
			logger.Warn("ivf open failed; falling back to brute", "err", err)
		} else {
			logger.Info("ivf dir not present; using brute-force", "dir", cfg.IVFDir)
		}
	}

	vecs, err := refdata.OpenVectors(cfg.VectorsPath, domain.VectorDimensions)
	if err != nil {
		return nil, 0, nil, fmt.Errorf("open vectors: %w", err)
	}
	lbls, err := refdata.OpenLabels(cfg.LabelsPath)
	if err != nil {
		_ = vecs.Close()
		return nil, 0, nil, fmt.Errorf("open labels: %w", err)
	}
	b, err := index.NewBrute(vecs, lbls)
	if err != nil {
		_ = vecs.Close()
		_ = lbls.Close()
		return nil, 0, nil, fmt.Errorf("brute searcher: %w", err)
	}
	logger.Info("using brute searcher", "count", b.Count())
	return b, b.Count(), func() {
		_ = vecs.Close()
		_ = lbls.Close()
	}, nil
}
