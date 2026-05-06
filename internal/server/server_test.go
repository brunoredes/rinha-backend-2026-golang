package server_test

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"rinha-backend-golang/m/v2/internal/detection"
	"rinha-backend-golang/m/v2/internal/domain"
	"rinha-backend-golang/m/v2/internal/index"
	"rinha-backend-golang/m/v2/internal/server"
)

type fakeSearcher struct {
	score float32
	err   error
}

func (f fakeSearcher) FraudScore(_ []float32, _ []index.Neighbor) (float32, error) {
	return f.score, f.err
}

func newHandler(t *testing.T, fs fakeSearcher) *server.Handler {
	t.Helper()
	norm := detection.Normalization{
		MaxAmount:            10000,
		MaxInstallments:      12,
		AmountVsAvgRatio:     10,
		MaxMinutes:           1440,
		MaxKm:                1000,
		MaxTxCount24h:        20,
		MaxMerchantAvgAmount: 10000,
	}
	v, err := detection.NewVectorizer(norm, map[string]float64{"5411": 0.15})
	if err != nil {
		t.Fatal(err)
	}
	h, err := server.New(server.Deps{
		Vectorizer: v,
		Searcher:   fs,
		K:          5,
		Threshold:  0.6,
	})
	if err != nil {
		t.Fatal(err)
	}
	h.SetReady(true)
	return h
}

func validReqJSON() string {
	return `{
	  "id":"tx-1",
	  "transaction":{"amount":41.12,"installments":2,"requested_at":"2026-03-11T18:45:53Z"},
	  "customer":{"avg_amount":82.24,"tx_count_24h":3,"known_merchants":["MERC-016"]},
	  "merchant":{"id":"MERC-016","mcc":"5411","avg_amount":60.25},
	  "terminal":{"is_online":false,"card_present":true,"km_from_home":29.23},
	  "last_transaction":null
	}`
}

func TestReady(t *testing.T) {
	t.Parallel()
	h := newHandler(t, fakeSearcher{})
	srv := httptest.NewServer(h.Routes())
	t.Cleanup(srv.Close)

	resp, err := http.Get(srv.URL + "/ready")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("got %d", resp.StatusCode)
	}
}

func TestFraudScore_Approved(t *testing.T) {
	t.Parallel()
	h := newHandler(t, fakeSearcher{score: 0.0})
	srv := httptest.NewServer(h.Routes())
	t.Cleanup(srv.Close)

	resp, err := http.Post(srv.URL+"/fraud-score", "application/json", strings.NewReader(validReqJSON()))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("status %d body %s", resp.StatusCode, body)
	}
	var got domain.FraudScoreResponse
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		t.Fatal(err)
	}
	if !got.Approved || got.FraudScore != 0 {
		t.Fatalf("got %+v", got)
	}
}

func TestFraudScore_Denied(t *testing.T) {
	t.Parallel()
	h := newHandler(t, fakeSearcher{score: 0.8})
	srv := httptest.NewServer(h.Routes())
	t.Cleanup(srv.Close)

	resp, err := http.Post(srv.URL+"/fraud-score", "application/json", strings.NewReader(validReqJSON()))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var got domain.FraudScoreResponse
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		t.Fatal(err)
	}
	if got.Approved || got.FraudScore < 0.79 || got.FraudScore > 0.81 {
		t.Fatalf("got %+v", got)
	}
}

func TestFraudScore_BadJSON(t *testing.T) {
	t.Parallel()
	h := newHandler(t, fakeSearcher{})
	srv := httptest.NewServer(h.Routes())
	t.Cleanup(srv.Close)

	resp, err := http.Post(srv.URL+"/fraud-score", "application/json", bytes.NewReader([]byte("{bad}")))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("got %d", resp.StatusCode)
	}
}

func TestFraudScore_SearcherError(t *testing.T) {
	t.Parallel()
	h := newHandler(t, fakeSearcher{err: errors.New("boom")})
	srv := httptest.NewServer(h.Routes())
	t.Cleanup(srv.Close)

	resp, err := http.Post(srv.URL+"/fraud-score", "application/json", strings.NewReader(validReqJSON()))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusInternalServerError {
		t.Fatalf("got %d", resp.StatusCode)
	}
}
