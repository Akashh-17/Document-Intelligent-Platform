# Aurora API — Error Code Reference (v3.2)

Every error returned by the Aurora query gateway and ingest tier carries a
stable code of the form `AUR-NNNN` in the `error.code` field of the JSON body.
Codes in the 1xxx range come from the ingest tier, 2xxx from the query gateway,
3xxx from authentication, and 4xxx from tenant quotas.

## Ingest tier (1xxx)

### AUR-1001 — EVENT_TOO_LARGE (HTTP 413)
The event exceeds the 256 KB maximum event size. Split the payload into
smaller events or move large blobs to object storage and send a reference.

### AUR-1002 — SCHEMA_MISMATCH (HTTP 422)
The event does not match the registered schema for its topic. Check the
`schema_version` field; producers must register new schema versions in the
schema registry before sending events that use them.

### AUR-1003 — TOPIC_NOT_FOUND (HTTP 404)
The Kafka topic or gRPC stream name is unknown. The default ingest topic is
`aurora.events.v1`; custom topics must be provisioned by the platform team.

### AUR-1007 — INGEST_BACKPRESSURE (HTTP 429)
The ingest tier is shedding load. Producers must retry with exponential
backoff starting at 200 ms, capped at 10 seconds. Do not retry tighter than
that: aggressive retries extend the overload.

## Query gateway (2xxx)

### AUR-2001 — QUERY_TIMEOUT (HTTP 504)
The query exceeded the 30-second query-gateway timeout and was cancelled.
Narrow the time range, pre-aggregate with a coarser rollup (1h or 1d), or
reduce group-by cardinality.

### AUR-2002 — TOO_MANY_CONCURRENT_QUERIES (HTTP 429)
The tenant already has 64 queries in flight, the per-tenant concurrency limit.
Queue queries client-side instead of retrying immediately.

### AUR-2005 — CROSS_TENANT_JOIN (HTTP 400)
The query references datasets owned by more than one tenant. Cross-tenant
joins are not supported by design; export the data and join it offline.

### AUR-2009 — RESULT_TOO_LARGE (HTTP 413)
The result set exceeds 50 MB. Add a `LIMIT`, paginate with the `cursor`
parameter, or aggregate further.

## Authentication (3xxx)

### AUR-3001 — CLIENT_CERT_MISSING (HTTP 401)
No client certificate was presented. Every request must use mTLS.

### AUR-3002 — OIDC_TOKEN_EXPIRED (HTTP 401)
The Helios SSO OIDC token has expired. Tokens are valid for 60 minutes;
refresh them using the refresh-token flow.

### AUR-3004 — PAT_NOT_SUPPORTED (HTTP 403)
A personal access token was sent. Aurora does not accept personal access
tokens; use mTLS plus an OIDC token from Helios SSO.

## Tenant quotas (4xxx)

### AUR-4001 — STORAGE_QUOTA_EXCEEDED (HTTP 507)
The tenant has used its raw-event storage quota (default 2 TB). Reduce the
raw-event retention period or request a quota increase from the platform team.

### AUR-4003 — RETENTION_OUT_OF_RANGE (HTTP 400)
The requested raw-event retention is outside the allowed range of 1 to 90 days.
