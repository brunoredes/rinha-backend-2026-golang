// Command preprocess converts references.json.gz into the binary refs.f32 +
// labels.bits files consumed at runtime. Run at image build time so the
// 3M-vector dataset is never re-parsed at container startup.
//
// Usage:
//
//	preprocess -in resources/references.json.gz \
//	    -out-vectors data/refs.f32 \
//	    -out-labels  data/labels.bits
package main

import (
	"compress/gzip"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"time"

	"rinha-backend-golang/m/v2/internal/domain"
	"rinha-backend-golang/m/v2/internal/refdata"
)

type config struct {
	in         string
	outVectors string
	outLabels  string
}

func parseFlags() config {
	var c config
	flag.StringVar(&c.in, "in", "resources/references.json.gz", "input references.json.gz")
	flag.StringVar(&c.outVectors, "out-vectors", "data/refs.f32", "output vector blob path")
	flag.StringVar(&c.outLabels, "out-labels", "data/labels.bits", "output label bitmap path")
	flag.Parse()
	return c
}

// referenceRecord matches one element of references.json.
type referenceRecord struct {
	Vector []float32 `json:"vector"`
	Label  string    `json:"label"`
}

func main() {
	cfg := parseFlags()
	logger := slog.New(slog.NewJSONHandler(os.Stderr, nil))

	if err := run(cfg, logger); err != nil {
		logger.Error("preprocess failed", "err", err)
		os.Exit(1)
	}
}

func run(cfg config, logger *slog.Logger) error {
	start := time.Now()

	if err := os.MkdirAll(filepath.Dir(cfg.outVectors), 0o755); err != nil {
		return fmt.Errorf("mkdir vectors out: %w", err)
	}
	if err := os.MkdirAll(filepath.Dir(cfg.outLabels), 0o755); err != nil {
		return fmt.Errorf("mkdir labels out: %w", err)
	}

	in, err := os.Open(cfg.in)
	if err != nil {
		return fmt.Errorf("open %s: %w", cfg.in, err)
	}
	defer in.Close()

	gz, err := gzip.NewReader(in)
	if err != nil {
		return fmt.Errorf("gzip reader %s: %w", cfg.in, err)
	}
	defer gz.Close()

	vecOut, err := os.Create(cfg.outVectors)
	if err != nil {
		return fmt.Errorf("create %s: %w", cfg.outVectors, err)
	}
	defer vecOut.Close()

	lblOut, err := os.Create(cfg.outLabels)
	if err != nil {
		return fmt.Errorf("create %s: %w", cfg.outLabels, err)
	}
	defer lblOut.Close()

	vw, err := refdata.NewVectorWriter(vecOut, domain.VectorDimensions)
	if err != nil {
		return err
	}
	lw, err := refdata.NewLabelWriter(lblOut)
	if err != nil {
		return err
	}

	count, frauds, err := streamRecords(gz, vw, lw)
	if err != nil {
		return err
	}

	if err := lw.Close(); err != nil {
		return fmt.Errorf("close labels: %w", err)
	}
	if err := vw.Close(); err != nil {
		return fmt.Errorf("close vectors: %w", err)
	}

	logger.Info("preprocess complete",
		"count", count,
		"frauds", frauds,
		"legit", count-frauds,
		"vector_bytes", uint64(refdata.HeaderSize)+count*uint64(domain.VectorDimensions)*4,
		"label_bytes", uint64(refdata.HeaderSize)+refdata.LabelBytes(count),
		"elapsed", time.Since(start).String(),
	)
	return nil
}

// streamRecords reads a JSON array from r, appending each record to vw and lw.
// Returns total count and number of fraud-labeled records.
func streamRecords(r io.Reader, vw *refdata.VectorWriter, lw *refdata.LabelWriter) (count, frauds uint64, err error) {
	dec := json.NewDecoder(r)

	if err = expectDelim(dec, '['); err != nil {
		return 0, 0, fmt.Errorf("expected JSON array open: %w", err)
	}

	for dec.More() {
		var rec referenceRecord
		if err = dec.Decode(&rec); err != nil {
			return count, frauds, fmt.Errorf("decode record %d: %w", count, err)
		}
		if len(rec.Vector) != domain.VectorDimensions {
			return count, frauds, fmt.Errorf("record %d: got %d dims want %d", count, len(rec.Vector), domain.VectorDimensions)
		}

		isFraud, err := parseLabel(rec.Label)
		if err != nil {
			return count, frauds, fmt.Errorf("record %d: %w", count, err)
		}

		if err = vw.Append(rec.Vector); err != nil {
			return count, frauds, err
		}
		if err = lw.Append(isFraud); err != nil {
			return count, frauds, err
		}

		if isFraud {
			frauds++
		}
		count++
	}

	if err = expectDelim(dec, ']'); err != nil {
		return count, frauds, fmt.Errorf("expected JSON array close: %w", err)
	}
	return count, frauds, nil
}

func parseLabel(s string) (bool, error) {
	switch s {
	case "fraud":
		return true, nil
	case "legit":
		return false, nil
	default:
		return false, fmt.Errorf("unknown label %q", s)
	}
}

func expectDelim(dec *json.Decoder, want json.Delim) error {
	tok, err := dec.Token()
	if err != nil {
		return err
	}
	d, ok := tok.(json.Delim)
	if !ok || d != want {
		return fmt.Errorf("got %v want %v", tok, want)
	}
	return nil
}
