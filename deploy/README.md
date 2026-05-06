# Deployment

The project ships as a 3-service `docker compose` stack: one nginx LB plus
two identical API replicas. The full resource budget is exactly the spec
ceiling — **1.00 CPU / 350 MB across all services**.

| service | cpus | memory |
|---------|-----:|-------:|
| nginx   | 0.10 | 20 M   |
| api1    | 0.45 | 165 M  |
| api2    | 0.45 | 165 M  |
| **total** | **1.00** | **350 M** |

## Build the image locally

The image is self-contained: at build time it pre-processes
`resources/references.json.gz` into the binary `refs.f32` + `labels.bits`
files and trains the IVF index, so the running container does no JSON
parsing of the dataset.

```bash
docker compose build
```

The build step takes ~70 s on a modern laptop (k-means dominates). The
resulting image is ~200 MB (≈170 MB of which is the IVF blobs).

## Run

```bash
docker compose up -d
curl -s http://localhost:9999/ready                      # 2xx
curl -s -X POST http://localhost:9999/fraud-score \
  -H 'Content-Type: application/json' -d @payload.json
```

## How the memory budget actually works

The 168 MB IVF blob would not fit twice — each API replica carrying its
own copy would blow the 165 MB per-container limit. To avoid that, both
API replicas mount the **same named volume** at `/data`:

```yaml
volumes:
  appdata:
services:
  api1: { volumes: [appdata:/data] }
  api2: { volumes: [appdata:/data], depends_on: [api1] }
```

Docker auto-populates `appdata` from `api1`'s image content the first
time it's mounted. `api2` then mounts the same inode — both processes
`mmap(2)` the file and the **kernel page cache holds exactly one set
of physical pages** under both processes.

## Publishing the image (for submission)

The submission branch's `docker-compose.yml` must reference public
images compatible with `linux-amd64`. To publish:

```bash
docker buildx build --platform linux/amd64 \
  -t ghcr.io/<your-username>/rinha-fraud:<tag> \
  --push .
```

Then update the `image:` field in `docker-compose.yml` (or set
`IMAGE=ghcr.io/...` so the env-substitution kicks in).

## Tunables (env vars on the API container)

| var       | default | meaning                                 |
|-----------|--------:|-----------------------------------------|
| `NPROBE`  | 8       | IVF cells visited per query             |
| `K`       | 5       | top-K neighbors for the score           |
| `THRESHOLD` | 0.6   | `approved = fraud_score < THRESHOLD`    |
| `GOMEMLIMIT` | 140MiB | soft cap on Go heap                   |
| `GOGC`    | 50      | GC trigger ratio                        |
| `GOMAXPROCS` | 1    | one OS thread per replica (matches cgroup CPU share) |
