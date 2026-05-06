# syntax=docker/dockerfile:1.7

# ---------- 1. Build the binaries and pre-process the dataset. -------------
# Doing the heavy work at image-build time means the runtime container
# never parses 3M JSON records, and the IVF index is shipped ready to mmap.
FROM golang:1.25-bookworm AS build

WORKDIR /src

ENV CGO_ENABLED=0 \
    GOOS=linux \
    GOARCH=amd64 \
    GOFLAGS=-trimpath

COPY go.mod go.sum ./
RUN go mod download

COPY . .

# Build the three binaries we actually use during the rest of the build.
RUN go build -ldflags='-s -w' -o /out/api          ./cmd/api      \
 && go build -ldflags='-s -w' -o /out/preprocess   ./cmd/preprocess \
 && go build -ldflags='-s -w' -o /out/buildindex   ./cmd/buildindex

# ---------- 2. Bake the dataset. -------------------------------------------
# Runs preprocess + buildindex against the bundled resources, producing
# /out/data/{refs.f32,labels.bits,ivf/*,normalization.json,mcc_risk.json}.
RUN mkdir -p /out/data \
 && /out/preprocess  -in resources/references.json.gz \
                     -out-vectors /out/data/refs.f32 \
                     -out-labels  /out/data/labels.bits \
 && /out/buildindex  -in-vectors /out/data/refs.f32 \
                     -in-labels  /out/data/labels.bits \
                     -out        /out/data/ivf \
                     -k 2048 -iters 10 -sample 200000 \
 && cp resources/normalization.json /out/data/normalization.json \
 && cp resources/mcc_risk.json      /out/data/mcc_risk.json

# ---------- 3. Runtime image. ----------------------------------------------
# Distroless static is ~2 MB and has no shell; the Go binary is statically
# linked (CGO disabled). Data lives at /data so a docker volume mounted at
# the same path will (a) be auto-populated from the image on first start,
# and (b) be shared between API replicas for one-copy mmap page cache.
FROM gcr.io/distroless/static-debian12:nonroot AS runtime

ENV ADDR=:9999 \
    DATA_VECTORS=/data/refs.f32 \
    DATA_LABELS=/data/labels.bits \
    DATA_IVF=/data/ivf \
    DATA_NORMALIZATION=/data/normalization.json \
    DATA_MCC_RISK=/data/mcc_risk.json \
    NPROBE=8 \
    K=5 \
    THRESHOLD=0.6 \
    GOMEMLIMIT=140MiB \
    GOGC=50 \
    GOMAXPROCS=1

COPY --from=build /out/api /api
COPY --from=build /out/data /data

EXPOSE 9999
USER nonroot:nonroot
ENTRYPOINT ["/api"]
