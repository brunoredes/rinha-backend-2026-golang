// Package config loads runtime configuration from environment variables.
// Validation is fail-fast at startup (CFG-1) and the returned struct is
// treated as immutable thereafter (CFG-2).
package config

import (
	"errors"
	"fmt"
	"os"
	"strconv"
	"time"
)

// API holds everything the fraud-score server needs at boot.
type API struct {
	Addr            string
	VectorsPath     string
	LabelsPath      string
	NormalizationPath string
	MCCRiskPath     string
	K               int
	Threshold       float32
	ReadTimeout     time.Duration
	WriteTimeout    time.Duration
}

// Load reads environment variables, applies defaults, and validates.
func Load() (API, error) {
	cfg := API{
		Addr:              getEnv("ADDR", ":9999"),
		VectorsPath:       getEnv("DATA_VECTORS", "data/refs.f32"),
		LabelsPath:        getEnv("DATA_LABELS", "data/labels.bits"),
		NormalizationPath: getEnv("DATA_NORMALIZATION", "resources/normalization.json"),
		MCCRiskPath:       getEnv("DATA_MCC_RISK", "resources/mcc_risk.json"),
		ReadTimeout:       2 * time.Second,
		WriteTimeout:      2 * time.Second,
	}

	k, err := getInt("K", 5)
	if err != nil {
		return API{}, err
	}
	cfg.K = k

	thr, err := getFloat32("THRESHOLD", 0.6)
	if err != nil {
		return API{}, err
	}
	cfg.Threshold = thr

	if err := cfg.validate(); err != nil {
		return API{}, err
	}
	return cfg, nil
}

func (c API) validate() error {
	if c.Addr == "" {
		return errors.New("config: ADDR must not be empty")
	}
	if c.K <= 0 {
		return fmt.Errorf("config: K must be > 0 (got %d)", c.K)
	}
	if c.Threshold < 0 || c.Threshold > 1 {
		return fmt.Errorf("config: THRESHOLD must be in [0,1] (got %v)", c.Threshold)
	}
	for _, p := range [...]struct{ name, path string }{
		{"DATA_VECTORS", c.VectorsPath},
		{"DATA_LABELS", c.LabelsPath},
	} {
		if p.path == "" {
			return fmt.Errorf("config: %s must not be empty", p.name)
		}
	}
	return nil
}

func getEnv(key, def string) string {
	if v, ok := os.LookupEnv(key); ok && v != "" {
		return v
	}
	return def
}

func getInt(key string, def int) (int, error) {
	v, ok := os.LookupEnv(key)
	if !ok || v == "" {
		return def, nil
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return 0, fmt.Errorf("config: %s: %w", key, err)
	}
	return n, nil
}

func getFloat32(key string, def float32) (float32, error) {
	v, ok := os.LookupEnv(key)
	if !ok || v == "" {
		return def, nil
	}
	f, err := strconv.ParseFloat(v, 32)
	if err != nil {
		return 0, fmt.Errorf("config: %s: %w", key, err)
	}
	return float32(f), nil
}
