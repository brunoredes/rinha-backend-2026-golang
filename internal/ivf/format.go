// Package ivf implements an inverted-file (IVF) index over the preprocessed
// reference dataset. The index is built offline by cmd/buildindex and read
// at runtime via Open / Search.
//
// Files written in the index directory:
//
//	centroids.f32     — refdata vector blob, K rows of dim float32
//	                    (the cell centers).
//	refs.f32          — refdata vector blob, count rows reordered so all
//	                    members of cell c are contiguous in [off[c]..off[c+1]).
//	labels.bits       — refdata bitmap aligned 1:1 with the reordered refs.
//	cell_offsets.u32  — prefix-sum offsets per cell (see format below).
//
// cell_offsets.u32 layout (64-byte header + (k+1) * uint32):
//
//	bytes  0.. 7  magic   "RNHA_OFF"
//	bytes  8..11  version u32          (== FormatVersion)
//	bytes 12..15  k       u32          (number of cells)
//	bytes 16..23  count   u64          (total reordered vectors)
//	bytes 24..63  reserved (zero)
//	bytes 64..    (k+1) * uint32 LE    (off[0]=0, off[k]=count)
package ivf

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
)

const (
	HeaderSize    = 64
	FormatVersion = uint32(1)
)

var magicOffsets = [8]byte{'R', 'N', 'H', 'A', '_', 'O', 'F', 'F'}

// Sentinel errors so callers can branch via errors.Is (ERR-2).
var (
	ErrBadMagic    = errors.New("ivf: bad magic")
	ErrBadVersion  = errors.New("ivf: unsupported version")
	ErrBadSize     = errors.New("ivf: payload size mismatch")
	ErrShortHeader = errors.New("ivf: short header")
	ErrBadOffsets  = errors.New("ivf: malformed cell offsets")
)

// OffsetsHeader is the fixed header of cell_offsets.u32.
type OffsetsHeader struct {
	Version uint32
	K       uint32
	Count   uint64
}

func writeOffsetsHeader(w io.Writer, h OffsetsHeader) error {
	var buf [HeaderSize]byte
	copy(buf[0:8], magicOffsets[:])
	binary.LittleEndian.PutUint32(buf[8:12], h.Version)
	binary.LittleEndian.PutUint32(buf[12:16], h.K)
	binary.LittleEndian.PutUint64(buf[16:24], h.Count)
	if _, err := w.Write(buf[:]); err != nil {
		return fmt.Errorf("write offsets header: %w", err)
	}
	return nil
}

func parseOffsetsHeader(buf []byte) (OffsetsHeader, error) {
	if len(buf) < HeaderSize {
		return OffsetsHeader{}, ErrShortHeader
	}
	for i, b := range magicOffsets {
		if buf[i] != b {
			return OffsetsHeader{}, ErrBadMagic
		}
	}
	h := OffsetsHeader{
		Version: binary.LittleEndian.Uint32(buf[8:12]),
		K:       binary.LittleEndian.Uint32(buf[12:16]),
		Count:   binary.LittleEndian.Uint64(buf[16:24]),
	}
	if h.Version != FormatVersion {
		return h, fmt.Errorf("%w: got %d want %d", ErrBadVersion, h.Version, FormatVersion)
	}
	return h, nil
}
