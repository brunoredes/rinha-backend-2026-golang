// Package server wires the HTTP layer for the fraud-detection API.
// The handlers expose:
//
//	GET  /ready        — 200 once the searcher is loaded.
//	POST /fraud-score  — vectorize, top-K, score, respond.
//
// The handlers are deliberately small: business logic lives in the
// vectorizer (internal/detection) and searcher (internal/index).
package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"sync"
	"sync/atomic"

	"rinha-backend-golang/m/v2/internal/detection"
	"rinha-backend-golang/m/v2/internal/domain"
	"rinha-backend-golang/m/v2/internal/index"
)

// Searcher is the small slice of *index.Brute (or future IVF) that the
// handler depends on. Defined near the consumer per CS-3.
type Searcher interface {
	FraudScore(query []float32, scratch []index.Neighbor) (float32, error)
}

// Deps groups the wiring required to build a Handler. Declared before the
// constructor per CS-6.
type Deps struct {
	Vectorizer *detection.Vectorizer
	Searcher   Searcher
	K          int
	Threshold  float32
	Logger     *slog.Logger
}

// Handler implements http.Handler for the API.
type Handler struct {
	vec       *detection.Vectorizer
	search    Searcher
	k         int
	threshold float32
	logger    *slog.Logger
	ready     atomic.Bool
	scratch   sync.Pool
}

// New builds a Handler. The handler is not Ready until SetReady(true) is
// called by the bootstrapper after refdata mmap succeeds.
func New(d Deps) (*Handler, error) {
	if d.Vectorizer == nil {
		return nil, errors.New("server: nil vectorizer")
	}
	if d.Searcher == nil {
		return nil, errors.New("server: nil searcher")
	}
	if d.K <= 0 {
		return nil, fmt.Errorf("server: K must be > 0 (got %d)", d.K)
	}
	if d.Logger == nil {
		d.Logger = slog.Default()
	}
	h := &Handler{
		vec:       d.Vectorizer,
		search:    d.Searcher,
		k:         d.K,
		threshold: d.Threshold,
		logger:    d.Logger,
	}
	h.scratch.New = func() any {
		s := make([]index.Neighbor, d.K)
		return &s
	}
	return h, nil
}

// SetReady toggles the readiness flag.
func (h *Handler) SetReady(ready bool) { h.ready.Store(ready) }

// Routes registers the endpoints on a fresh ServeMux.
func (h *Handler) Routes() *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /ready", h.handleReady)
	mux.HandleFunc("POST /fraud-score", h.handleFraudScore)
	return mux
}

func (h *Handler) handleReady(w http.ResponseWriter, _ *http.Request) {
	if !h.ready.Load() {
		http.Error(w, "not ready", http.StatusServiceUnavailable)
		return
	}
	w.WriteHeader(http.StatusOK)
}

func (h *Handler) handleFraudScore(w http.ResponseWriter, r *http.Request) {
	var req domain.FraudScoreRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid json: "+err.Error(), http.StatusBadRequest)
		return
	}
	if err := req.Validate(); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	vec, err := h.vec.Build(req)
	if err != nil {
		http.Error(w, "vectorize: "+err.Error(), http.StatusBadRequest)
		return
	}

	scratchPtr := h.scratch.Get().(*[]index.Neighbor)
	defer h.scratch.Put(scratchPtr)

	score, err := h.search.FraudScore(vec[:], *scratchPtr)
	if err != nil {
		h.logger.Error("search failed", "id", req.ID, "err", err)
		http.Error(w, "search failed", http.StatusInternalServerError)
		return
	}

	resp := domain.FraudScoreResponse{
		Approved:   score < h.threshold,
		FraudScore: float64(score),
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(resp); err != nil {
		h.logger.Warn("encode response", "id", req.ID, "err", err)
	}
}
