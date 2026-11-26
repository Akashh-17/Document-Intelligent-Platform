# Aurora Configuration Reference

Aurora services read configuration from `aurora.yaml`, and every key can be
overridden with an environment variable: upper-case the key and replace dots
with double underscores (for example `query.timeout_ms` becomes
`QUERY__TIMEOUT_MS`). Values below are the v3.2 defaults.

## Ingest

| Key | Default | Description |
|---|---|---|
| `ingest.max_event_bytes` | `262144` | Maximum event size (256 KB). Hard ceiling; cannot be raised. |
| `ingest.default_topic` | `aurora.events.v1` | Kafka topic used when a producer names none. |
| `ingest.backoff_initial_ms` | `200` | First retry delay recommended to producers after AUR-1007. |
| `ingest.backoff_max_ms` | `10000` | Maximum producer retry delay. |
| `ingest.grpc_port` | `7443` | gRPC ingest port (mTLS only). |

## Query gateway

| Key | Default | Description |
|---|---|---|
| `query.timeout_ms` | `30000` | Query timeout. Exceeding it returns HTTP 504 (AUR-2001). |
| `query.max_concurrency_per_tenant` | `64` | Queries in flight per tenant (AUR-2002 when exceeded). |
| `query.max_result_bytes` | `52428800` | 50 MB result-size cap (AUR-2009). |
| `query.cache_ttl_s` | `15` | Result-cache lifetime for identical dashboard queries. |

## Retention and storage

| Key | Default | Description |
|---|---|---|
| `retention.raw_days` | `14` | Raw-event retention. Allowed range is 1–90 days. |
| `retention.rollup_1d_days` | `730` | Daily rollup retention. Fixed; not configurable. |
| `retention.rollup_1m_days` | `30` | 1-minute rollup retention. |
| `storage.tenant_quota_tb` | `2` | Raw-event storage quota per tenant (AUR-4001 when exceeded). |

## Authentication

| Key | Default | Description |
|---|---|---|
| `auth.mode` | `mtls+oidc` | The only supported mode since v3.0. |
| `auth.oidc_issuer` | `https://sso.helios.internal` | Helios SSO issuer URL. |
| `auth.token_ttl_minutes` | `60` | OIDC access-token lifetime. |

## Changing configuration safely

Configuration changes are rolled out with `aurora-ctl config apply`, which
validates the file, applies it to one canary shard, waits 10 minutes, and then
rolls it out to the remaining shards. Use `aurora-ctl config diff` to preview a
change before applying it.
