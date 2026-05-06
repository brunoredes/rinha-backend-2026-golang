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

	logger.Info("loading refdata", "vectors", cfg.VectorsPath, "labels", cfg.LabelsPath)
	startLoad := time.Now()

	vecs, err := refdata.OpenVectors(cfg.VectorsPath, domain.VectorDimensions)
	if err != nil {
		return fmt.Errorf("open vectors: %w", err)
	}
	defer vecs.Close()

	lbls, err := refdata.OpenLabels(cfg.LabelsPath)
	if err != nil {
		return fmt.Errorf("open labels: %w", err)
	}
	defer lbls.Close()

	searcher, err := index.NewBrute(vecs, lbls)
	if err != nil {
		return fmt.Errorf("build searcher: %w", err)
	}

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
		"count", searcher.Count(),
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
