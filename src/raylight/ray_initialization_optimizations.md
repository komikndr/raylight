# Ray Initialization Optimizations

This document describes the performance optimizations implemented in Raylight's Ray cluster initialization.

## Performance Results

**Before optimizations:** ~62 seconds to XDiT enable  
**After optimizations:** ~37 seconds to XDiT enable  
**Savings:** ~25 seconds (~40% faster)

## Optimizations

### 1. Skip NCCL Communication Test (`skip_comm_test`)

**Default:** `True`  
**Savings:** ~10-15 seconds

Previously, Raylight spawned separate `RayCOMMTester` actors before the real workers to validate NCCL communication. When `skip_comm_test=True`, this test is skipped entirely.

### 2. Cluster Reuse

**Savings:** ~8 seconds

Checks if Ray is already initialized with sufficient GPUs before calling `ray.init()`.

### 3. Deferred Module Installation (`eager_install=False`)

**Savings:** ~2-3 seconds

Defers module installation to when actors actually spawn.

### 4. Disabled Metrics Agent

**Savings:** ~2-5 seconds

Environment variables set:
- `RAY_enable_metrics_collection=0`
- `RAY_USAGE_STATS_ENABLED=0`
- `RAY_METRICS_EXPORT_INTERVAL_MS=0`

Plus `_metrics_export_port=None` in `ray.init()`.

## Usage

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `skip_comm_test` | Boolean | True | Skip NCCL test at startup |

## Troubleshooting

If you encounter NCCL errors during sampling, set `skip_comm_test=False` to debug.
