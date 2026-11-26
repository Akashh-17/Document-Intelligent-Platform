# Aurora On-Call Runbook

This runbook is for engineers on the Aurora on-call rotation at Helios Labs.
The rotation is one week long and hands over every Monday at 10:00 UTC.

## Severity levels

- **SEV1** — Aurora is down or losing data for multiple tenants. Page the
  incident commander immediately and open the `#inc-aurora` channel.
  Acknowledge within 5 minutes.
- **SEV2** — degraded for one tenant or a single SLO is at risk. Acknowledge
  within 15 minutes.
- **SEV3** — no customer impact yet. Handle during working hours.

## Alert: AuroraIngestLagHigh

Fires when consumer lag on `aurora.events.v1` stays above 500,000 messages for
10 minutes.

1. Check the ingest dashboard for a spike in AUR-1007 backpressure errors.
2. Scale the ingest workers: `aurora-ctl scale ingest --replicas +4`.
3. If lag keeps growing after 15 minutes, escalate to SEV2.

## Alert: AuroraRollupFreshnessBreach

Fires when 1-minute rollups are more than 5 seconds behind the source events,
which breaches the freshness SLO.

1. Check ClickHouse merge pressure on the `aurora_rollups` cluster.
2. Pause non-critical backfills: `aurora-ctl backfill pause --all`.
3. If the breach lasts longer than 30 minutes, declare SEV2.

## Alert: AuroraQueryP99High

Fires when dashboard p99 query latency is above the 800 ms SLO for 15 minutes.

1. Look for a single tenant issuing expensive queries (high AUR-2001 timeout
   rate).
2. Temporarily lower that tenant's concurrency:
   `aurora-ctl tenant limit <tenant-id> --max-concurrency 16`.
3. Contact the tenant's owning team before restoring the default limit of 64.

## Draining a tenant

To move a tenant off a failing shard, run
`aurora-ctl drain --tenant <tenant-id> --target-shard <shard>`. Draining takes
about 20 minutes per terabyte of raw events. Never drain more than two tenants
at the same time.

## Escalation

If the primary on-call does not acknowledge a page within the time limit for
its severity, the page escalates to the secondary on-call, then to the Aurora
engineering manager. Post-incident reviews are mandatory for SEV1 and SEV2 and
must be published within 5 business days.
