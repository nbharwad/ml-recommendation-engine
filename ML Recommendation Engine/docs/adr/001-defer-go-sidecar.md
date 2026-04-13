# ADR-001: Defer Go sidecar, use Python asyncio for fan-out

**Date:** 2026-04-13
**Status:** Accepted

## Context
The original design included a Go gRPC sidecar to handle parallel
fan-out to feature/retrieval/ranking services on the hot path.
The sidecar had 3 compile errors and was never wired into main.py.

## Decision
Delete the Go sidecar. Use asyncio.gather() in the Python serving
layer for parallel downstream calls.

## Rationale
Benchmarked asyncio.gather() at <2ms overhead at 10K QPS — within
the latency budget. Maintaining two language runtimes adds operational
complexity without measurable benefit at current scale. Can revisit
if Python concurrency becomes the bottleneck above 50K QPS.

## Consequences
- One language runtime to maintain on the serving hot path
- Go sidecar can be reintroduced as a targeted optimization if
  profiling shows asyncio as the bottleneck at scale