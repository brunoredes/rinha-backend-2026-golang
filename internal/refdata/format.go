// Package refdata defines the on-disk binary format for the preprocessed
// reference dataset and provides reader/writer helpers.
//
// The format is intentionally tiny and version-tagged so the preprocess
// step (build time) and the runtime can evolve independently.
//
//	refs.f32   — vector blob.
//	  bytes  0.. 7  magic           "RNHA_VEC"
//	  bytes  8..11  version u32     LE
//	  bytes 12..15  dim u32         LE   (== domain.VectorDimensions)
//	  bytes 16..23  count u64       LE
//	  bytes 24..27  flags u32       LE   (reserved; 0 today)
//	  bytes 28..63  reserved (zero)
//	  bytes 64..    count*dim float32 LE, 64-byte aligned start so AVX
//	                loads can be aligned.
//
//	labels.bits — packed fraud bitmap, one bit per vector (1 = fraud).
//	  bytes  0.. 7  magic           "RNHA_LBL"
//	  bytes  8..11  version u32     LE
//	  bytes 12..19  count u64       LE
//	  bytes 20..63  reserved (zero)
//	  bytes 64..    ceil(count/8) bytes, bit (i%8) of byte (i/8).
package refdata

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
)

const (
	HeaderSize    = 64
	FormatVersion = uint32(1)

	// TailPadFloats is the number of zero float32 lanes appended after the
	// vector payload so a SIMD kernel reading two YMM registers (16 floats)
	// can over-read past the last vector without segfaulting or pulling
	// adjacent vector data into the high lanes. 8 is enough for a single
	// trailing YMM; we use 16 for headroom.
	TailPadFloats = 16
)

var (
	magicVectors = [8]byte{'R', 'N', 'H', 'A', '_', 'V', 'E', 'C'}
	magicLabels  = [8]byte{'R', 'N', 'H', 'A', '_', 'L', 'B', 'L'}
)

// Sentinel errors so callers can branch on cause without string matching (ERR-2).
var (
	ErrBadMagic       = errors.New("refdata: bad magic")
	ErrBadVersion     = errors.New("refdata: unsupported version")
	ErrBadDim         = errors.New("refdata: dim mismatch")
	ErrBadSize        = errors.New("refdata: payload size mismatch")
	ErrShortHeader    = errors.New("refdata: short header")
)

// VectorHeader is the 64-byte fixed header of refs.f32.
type VectorHeader struct {
	Version uint32
	Dim     uint32
	Count   uint64
	Flags   uint32
}

// LabelHeader is the 64-byte fixed header of labels.bits.
type LabelHeader struct {
	Version uint32
	Count   uint64
}

// writeVectorHeader emits a 64-byte header for the vector blob.
func writeVectorHeader(w io.Writer, h VectorHeader) error {
	var buf [HeaderSize]byte
	copy(buf[0:8], magicVectors[:])
	binary.LittleEndian.PutUint32(buf[8:12], h.Version)
	binary.LittleEndian.PutUint32(buf[12:16], h.Dim)
	binary.LittleEndian.PutUint64(buf[16:24], h.Count)
	binary.LittleEndian.PutUint32(buf[24:28], h.Flags)
	if _, err := w.Write(buf[:]); err != nil {
		return fmt.Errorf("write vector header: %w", err)
	}
	return nil
}

// writeLabelHeader emits a 64-byte header for the label bitmap.
func writeLabelHeader(w io.Writer, h LabelHeader) error {
	var buf [HeaderSize]byte
	copy(buf[0:8], magicLabels[:])
	binary.LittleEndian.PutUint32(buf[8:12], h.Version)
	binary.LittleEndian.PutUint64(buf[12:20], h.Count)
	if _, err := w.Write(buf[:]); err != nil {
		return fmt.Errorf("write label header: %w", err)
	}
	return nil
}

// parseVectorHeader validates magic/version and returns header fields.
func parseVectorHeader(buf []byte) (VectorHeader, error) {
	if len(buf) < HeaderSize {
		return VectorHeader{}, ErrShortHeader
	}
	for i, b := range magicVectors {
		if buf[i] != b {
			return VectorHeader{}, ErrBadMagic
		}
	}
	h := VectorHeader{
		Version: binary.LittleEndian.Uint32(buf[8:12]),
		Dim:     binary.LittleEndian.Uint32(buf[12:16]),
		Count:   binary.LittleEndian.Uint64(buf[16:24]),
		Flags:   binary.LittleEndian.Uint32(buf[24:28]),
	}
	if h.Version != FormatVersion {
		return h, fmt.Errorf("%w: got %d want %d", ErrBadVersion, h.Version, FormatVersion)
	}
	return h, nil
}

// parseLabelHeader validates magic/version and returns header fields.
func parseLabelHeader(buf []byte) (LabelHeader, error) {
	if len(buf) < HeaderSize {
		return LabelHeader{}, ErrShortHeader
	}
	for i, b := range magicLabels {
		if buf[i] != b {
			return LabelHeader{}, ErrBadMagic
		}
	}
	h := LabelHeader{
		Version: binary.LittleEndian.Uint32(buf[8:12]),
		Count:   binary.LittleEndian.Uint64(buf[12:20]),
	}
	if h.Version != FormatVersion {
		return h, fmt.Errorf("%w: got %d want %d", ErrBadVersion, h.Version, FormatVersion)
	}
	return h, nil
}

// LabelBytes returns the number of bytes needed to pack count bits.
func LabelBytes(count uint64) uint64 {
	return (count + 7) / 8
}
