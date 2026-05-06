package refdata

import (
	"fmt"
	"os"
	"unsafe"

	"golang.org/x/sys/unix"
)

// Vectors is a read-only mmap of refs.f32. The Data slice is a view into
// the file's payload region (starting at offset HeaderSize) reinterpreted
// as []float32. The backing memory is not garbage collected; callers must
// invoke Close to release it.
type Vectors struct {
	Header VectorHeader
	Data   []float32
	raw    []byte
}

// Labels is a read-only mmap of labels.bits. Use Get to read a single bit.
type Labels struct {
	Header LabelHeader
	Bits   []byte
	raw    []byte
}

// OpenVectors mmaps refs.f32, validates the header, and exposes the payload
// as a []float32 view. The view aliases page-cache memory so multiple
// processes share the same physical pages.
func OpenVectors(path string, expectDim uint32) (*Vectors, error) {
	raw, err := mmapReadOnly(path)
	if err != nil {
		return nil, err
	}
	h, err := parseVectorHeader(raw)
	if err != nil {
		_ = unix.Munmap(raw)
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	if h.Dim != expectDim {
		_ = unix.Munmap(raw)
		return nil, fmt.Errorf("%w: got %d want %d", ErrBadDim, h.Dim, expectDim)
	}
	want := uint64(h.Dim) * h.Count * 4
	if uint64(len(raw))-HeaderSize < want {
		_ = unix.Munmap(raw)
		return nil, fmt.Errorf("%w: file payload %d < expected %d", ErrBadSize, uint64(len(raw))-HeaderSize, want)
	}
	payload := raw[HeaderSize : HeaderSize+int(want)]
	var view []float32
	if h.Count > 0 {
		// Reinterpret as []float32. Safe: payload is 64-byte aligned (header is
		// exactly HeaderSize bytes which is a multiple of 4).
		view = unsafe.Slice((*float32)(unsafe.Pointer(&payload[0])), int(h.Count)*int(h.Dim))
	}
	return &Vectors{Header: h, Data: view, raw: raw}, nil
}

// Row returns the i-th vector as a slice aliasing the mmap region.
func (v *Vectors) Row(i uint64) []float32 {
	off := int(i) * int(v.Header.Dim)
	return v.Data[off : off+int(v.Header.Dim)]
}

// Close releases the mmap.
func (v *Vectors) Close() error {
	if v.raw == nil {
		return nil
	}
	err := unix.Munmap(v.raw)
	v.raw = nil
	v.Data = nil
	return err
}

// OpenLabels mmaps labels.bits and validates the header.
func OpenLabels(path string) (*Labels, error) {
	raw, err := mmapReadOnly(path)
	if err != nil {
		return nil, err
	}
	h, err := parseLabelHeader(raw)
	if err != nil {
		_ = unix.Munmap(raw)
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	want := LabelBytes(h.Count)
	if uint64(len(raw))-HeaderSize < want {
		_ = unix.Munmap(raw)
		return nil, fmt.Errorf("%w: file payload %d < expected %d", ErrBadSize, uint64(len(raw))-HeaderSize, want)
	}
	bits := raw[HeaderSize : HeaderSize+int(want)]
	return &Labels{Header: h, Bits: bits, raw: raw}, nil
}

// Get returns true if vector i is labeled fraud.
func (l *Labels) Get(i uint64) bool {
	return l.Bits[i>>3]&(1<<(i&7)) != 0
}

// Close releases the mmap.
func (l *Labels) Close() error {
	if l.raw == nil {
		return nil
	}
	err := unix.Munmap(l.raw)
	l.raw = nil
	l.Bits = nil
	return err
}

// mmapReadOnly opens the file shared, read-only. MADV_RANDOM is hinted
// because IVF probing accesses cells out of order; switch to MADV_WILLNEED
// during warmup if cold-start matters.
func mmapReadOnly(path string) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat %s: %w", path, err)
	}
	size := st.Size()
	if size < HeaderSize {
		return nil, fmt.Errorf("%w: %s is %d bytes", ErrShortHeader, path, size)
	}
	data, err := unix.Mmap(int(f.Fd()), 0, int(size), unix.PROT_READ, unix.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("mmap %s: %w", path, err)
	}
	_ = unix.Madvise(data, unix.MADV_RANDOM)
	return data, nil
}
