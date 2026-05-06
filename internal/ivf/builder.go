package ivf

import (
	"encoding/binary"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"time"

	"rinha-backend-golang/m/v2/internal/index"
	"rinha-backend-golang/m/v2/internal/refdata"
)

// BuildParams describes the inputs for Build. Declared before consumer (CS-6).
type BuildParams struct {
	Centroids []float32 // K * Dim, flat
	Vectors   *refdata.Vectors
	Labels    *refdata.Labels
	OutDir    string
	K         int
	Dim       int
	Logger    *slog.Logger
}

// Build assigns every reference vector to its nearest centroid, sorts the
// dataset so that all members of cell c sit at refs[off[c]..off[c+1]),
// and writes the four artifact files into OutDir:
//
//	centroids.f32, refs.f32, labels.bits, cell_offsets.u32
//
// Memory: builds a 168 MB output buffer in-process. Acceptable for the
// build-time tool; the runtime never allocates this much.
func Build(p BuildParams) error {
	if p.Vectors == nil || p.Labels == nil {
		return fmt.Errorf("ivf.Build: nil vectors or labels")
	}
	if p.Vectors.Header.Count != p.Labels.Header.Count {
		return fmt.Errorf("ivf.Build: count mismatch vectors=%d labels=%d",
			p.Vectors.Header.Count, p.Labels.Header.Count)
	}
	if p.K <= 0 || p.Dim <= 0 {
		return fmt.Errorf("ivf.Build: bad K=%d or Dim=%d", p.K, p.Dim)
	}
	if len(p.Centroids) != p.K*p.Dim {
		return fmt.Errorf("ivf.Build: centroids len %d want %d", len(p.Centroids), p.K*p.Dim)
	}
	if p.Logger == nil {
		p.Logger = slog.Default()
	}
	if err := os.MkdirAll(p.OutDir, 0o755); err != nil {
		return fmt.Errorf("mkdir %s: %w", p.OutDir, err)
	}

	count := int(p.Vectors.Header.Count)
	dim := p.Dim

	start := time.Now()

	// Pass 1: assign every ref to its nearest centroid + count per cell.
	assignments := make([]uint32, count)
	counts := make([]uint32, p.K)
	dists := make([]float32, p.K)
	var qbuf [index.QueryStride]float32

	src := p.Vectors.Data
	for i := 0; i < count; i++ {
		copyToQueryBuf(qbuf[:], src[i*dim:(i+1)*dim], dim)
		index.DistancesBlock(qbuf[:], p.Centroids, p.K, dists)
		best := 0
		bestD := dists[0]
		for c := 1; c < p.K; c++ {
			if dists[c] < bestD {
				bestD = dists[c]
				best = c
			}
		}
		assignments[i] = uint32(best)
		counts[best]++
		if i > 0 && i%500_000 == 0 {
			p.Logger.Info("assigning", "done", i, "elapsed", time.Since(start).String())
		}
	}
	p.Logger.Info("assigned all refs", "count", count, "elapsed", time.Since(start).String())

	// Compute prefix-sum offsets and per-cell write cursors.
	offsets := make([]uint32, p.K+1)
	for c := 0; c < p.K; c++ {
		offsets[c+1] = offsets[c] + counts[c]
	}
	cursors := make([]uint32, p.K)
	copy(cursors, offsets[:p.K])

	// Pass 2: scatter ref bytes + label bits into reorder buffers.
	reorderedVecs := make([]float32, count*dim)
	reorderedLabels := make([]byte, refdata.LabelBytes(uint64(count)))
	for i := 0; i < count; i++ {
		c := assignments[i]
		dst := cursors[c]
		cursors[c]++
		copy(reorderedVecs[int(dst)*dim:(int(dst)+1)*dim], src[i*dim:(i+1)*dim])
		if p.Labels.Get(uint64(i)) {
			reorderedLabels[dst>>3] |= 1 << (dst & 7)
		}
	}
	p.Logger.Info("reorder complete", "elapsed", time.Since(start).String())

	if err := writeVectorBlob(filepath.Join(p.OutDir, "refs.f32"), reorderedVecs, dim); err != nil {
		return err
	}
	if err := writeLabelBlob(filepath.Join(p.OutDir, "labels.bits"), reorderedLabels, uint64(count)); err != nil {
		return err
	}
	if err := writeVectorBlobFromFlat(filepath.Join(p.OutDir, "centroids.f32"), p.Centroids, dim); err != nil {
		return err
	}
	if err := writeOffsets(filepath.Join(p.OutDir, "cell_offsets.u32"), offsets, uint64(count)); err != nil {
		return err
	}

	p.Logger.Info("ivf build complete",
		"k", p.K,
		"count", count,
		"out", p.OutDir,
		"elapsed", time.Since(start).String(),
	)
	return nil
}

func writeVectorBlob(path string, flat []float32, dim int) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create %s: %w", path, err)
	}
	defer f.Close()
	w, err := refdata.NewVectorWriter(f, uint32(dim))
	if err != nil {
		return err
	}
	n := len(flat) / dim
	for i := 0; i < n; i++ {
		if err := w.Append(flat[i*dim : (i+1)*dim]); err != nil {
			return err
		}
	}
	return w.Close()
}

func writeVectorBlobFromFlat(path string, flat []float32, dim int) error {
	return writeVectorBlob(path, flat, dim)
}

func writeLabelBlob(path string, packed []byte, count uint64) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create %s: %w", path, err)
	}
	defer f.Close()
	w, err := refdata.NewLabelWriter(f)
	if err != nil {
		return err
	}
	for i := uint64(0); i < count; i++ {
		fraud := packed[i>>3]&(1<<(i&7)) != 0
		if err := w.Append(fraud); err != nil {
			return err
		}
	}
	return w.Close()
}

func writeOffsets(path string, offsets []uint32, count uint64) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create %s: %w", path, err)
	}
	defer f.Close()
	if err := writeOffsetsHeader(f, OffsetsHeader{
		Version: FormatVersion,
		K:       uint32(len(offsets) - 1),
		Count:   count,
	}); err != nil {
		return err
	}
	buf := make([]byte, len(offsets)*4)
	for i, off := range offsets {
		binary.LittleEndian.PutUint32(buf[i*4:i*4+4], off)
	}
	if _, err := f.Write(buf); err != nil {
		return fmt.Errorf("write offsets: %w", err)
	}
	return nil
}
