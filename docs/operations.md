# Operational Guide

This guide summarizes the day-2 operational practices for running the KLL sketch in production services and batch pipelines.

## Monitoring & Observability

### Key service level indicators
- **Ingestion throughput**: items/second (overall and per tenant). Track moving averages and high-water marks to detect ingestion stalls.
- **Sketch merge latency**: wall-clock latency and failure rate for merge jobs. Watch for sustained increases (>2× baseline) which indicate undersized capacity.
- **Quantile query latency**: P50/P95/P99 latency per query type (`quantile`, `quantiles`, `rank`).
- **Serialized blob size**: mean and max payload size emitted by `to_bytes()`. Sudden jumps usually mean skewed data or capacity misconfiguration.
- **Compaction counters**: number of compactions per level. Rising compaction frequency implies either a bursty workload or an undersized `capacity`.

### Recommended instrumentation
- Wrap ingestion entry points (e.g., `add`, `extend`) and query methods with metrics (`Counter`/`Histogram`). Expose via Prometheus or your platform’s native telemetry.
- Log sketch metadata at debug level: `capacity`, `size`, level configuration, deterministic seed. Mask PII, logging only aggregates.
- Emit structured events when merges occur, including source shard identifiers and duration.
- Sample serialized blobs and validate round-trips in background tasks to catch corruption early.

### Alerting policies
- **Ingestion stalled**: alert when ingestion throughput drops to zero for more than 1 minute while upstream traffic persists.
- **Merge backlog**: alert when merge queue length exceeds 2× normal baseline for 5 consecutive intervals.
- **Query SLO breach**: alert when P95 query latency exceeds the agreed SLO (e.g., 50 ms) for 3 consecutive intervals.
- **Serialization errors**: alert on any deserialization failures or checksum mismatches.

### Dashboards & diagnostics
- Chart ingestion throughput, query latency, and serialized blob sizes on a single operational dashboard.
- Maintain a table of per-level buffer sizes and compaction counts to aid debugging.
- Surface release version, git SHA, and configuration flags in dashboard annotations.

## Capacity & Configuration Management
- Size `capacity` based on required rank error ε using `ε ≈ 1 / capacity`. Double the capacity if you observe compaction hot spots or serialized blobs exceeding transport limits.
- For workloads with heavy merges, align shard capacities; a single small shard can dominate error.
- Document default seeds and level configuration in configuration management. Keep environment-specific overrides in version control.

## Upgrade Playbook

1. **Review release notes**
   - Read `CHANGELOG.md` for breaking changes, new features, and migration steps.
   - Check dependency bumps, especially the minimum supported Python version and `setuptools` constraints.

2. **Stage the upgrade**
   - Pin the target version in your dependency management tool and deploy to a staging environment.
   - Run the full pytest suite plus representative workload benchmarks (`benchmarks/bench_kll.py`) on staging data.
   - Validate that serialized blobs created by the previous version deserialize correctly with the new release (forwards compatibility) and vice versa (backwards compatibility when rolling back).

3. **Production rollout**
   - Perform a canary deployment (5–10% traffic) and monitor ingestion throughput, query latency, and error rates for at least one compaction window.
   - If metrics remain within SLOs, proceed with progressive rollout to all shards or services.

4. **Rollback procedure**
   - Maintain the previous version pinned and ready for redeploy.
   - Because serialization is version-stable, downgrades are safe provided no breaking schema change is noted in the changelog. Always confirm with staged rollback tests.
   - After rollback, clear metrics annotations and document the incident.

5. **Post-upgrade validation**
   - Confirm dashboards show the new version identifiers.
   - Update operational runbooks with any new configuration flags or behaviours introduced in the release.

## Incident Response Checklist
- Capture failing serialized blobs and store them with timestamps and shard identifiers.
- Dump per-level buffer states via `KLL.debug_state()` (if enabled) or the equivalent introspection helper for forensic analysis.
- Reconstruct workloads that triggered failures using recorded input batches and replay them in a sandbox before patching production.

## Documentation & Runbook Hygiene
- Store this guide alongside other operational runbooks in your organization’s knowledge base.
- Schedule quarterly reviews to update thresholds, metrics, and playbooks in line with observed production behaviour.
- When onboarding new services, link this document from their service runbooks to ensure consistent operational standards.

