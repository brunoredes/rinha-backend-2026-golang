package refdata

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
)

// VectorWriter streams float32 vectors into an *os.File, writing a
// placeholder header up-front and patching the final count on Close.
// The underlying file is left open and positioned at end-of-file.
type VectorWriter struct {
	f     *os.File
	bw    *bufio.Writer
	dim   uint32
	count uint64
	row   []byte
}

// NewVectorWriter wraps f. f must be a writable *os.File (Seek-able).
func NewVectorWriter(f *os.File, dim uint32) (*VectorWriter, error) {
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return nil, fmt.Errorf("seek %s: %w", f.Name(), err)
	}
	bw := bufio.NewWriterSize(f, 1<<20)
	if err := writeVectorHeader(bw, VectorHeader{Version: FormatVersion, Dim: dim, Count: 0}); err != nil {
		return nil, err
	}
	return &VectorWriter{f: f, bw: bw, dim: dim, row: make([]byte, int(dim)*4)}, nil
}

// Append writes one vector. Length must equal the writer's dim.
func (w *VectorWriter) Append(vec []float32) error {
	if uint32(len(vec)) != w.dim {
		return fmt.Errorf("%w: got %d want %d", ErrBadDim, len(vec), w.dim)
	}
	for i, v := range vec {
		binary.LittleEndian.PutUint32(w.row[i*4:i*4+4], math.Float32bits(v))
	}
	if _, err := w.bw.Write(w.row); err != nil {
		return fmt.Errorf("write vector %d: %w", w.count, err)
	}
	w.count++
	return nil
}

// Count returns how many vectors have been appended.
func (w *VectorWriter) Count() uint64 { return w.count }

// Close flushes and patches the header with the final count.
func (w *VectorWriter) Close() error {
	if err := w.bw.Flush(); err != nil {
		return fmt.Errorf("flush: %w", err)
	}
	return patchUint64At(w.f, 16, w.count)
}

// LabelWriter streams a packed fraud bitmap into an *os.File.
type LabelWriter struct {
	f     *os.File
	bw    *bufio.Writer
	count uint64
	cur   byte
	bit   uint
}

// NewLabelWriter wraps f and writes a placeholder header.
func NewLabelWriter(f *os.File) (*LabelWriter, error) {
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return nil, fmt.Errorf("seek %s: %w", f.Name(), err)
	}
	bw := bufio.NewWriterSize(f, 1<<20)
	if err := writeLabelHeader(bw, LabelHeader{Version: FormatVersion, Count: 0}); err != nil {
		return nil, err
	}
	return &LabelWriter{f: f, bw: bw}, nil
}

// Append records one label (true = fraud).
func (w *LabelWriter) Append(fraud bool) error {
	if fraud {
		w.cur |= 1 << w.bit
	}
	w.bit++
	w.count++
	if w.bit == 8 {
		if err := w.bw.WriteByte(w.cur); err != nil {
			return fmt.Errorf("write label byte: %w", err)
		}
		w.cur = 0
		w.bit = 0
	}
	return nil
}

// Count returns how many labels have been appended.
func (w *LabelWriter) Count() uint64 { return w.count }

// Close flushes the trailing partial byte (if any) and patches the count.
func (w *LabelWriter) Close() error {
	if w.bit != 0 {
		if err := w.bw.WriteByte(w.cur); err != nil {
			return fmt.Errorf("flush trailing byte: %w", err)
		}
	}
	if err := w.bw.Flush(); err != nil {
		return fmt.Errorf("flush: %w", err)
	}
	return patchUint64At(w.f, 12, w.count)
}

// patchUint64At writes a little-endian uint64 at the given byte offset.
func patchUint64At(f *os.File, off int64, v uint64) error {
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], v)
	if _, err := f.WriteAt(buf[:], off); err != nil {
		return fmt.Errorf("patch header at %d: %w", off, err)
	}
	return nil
}
