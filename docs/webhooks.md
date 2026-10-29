# Webhook delivery and retries

Inbound webhooks are accepted by a thin HTTP handler that verifies the HMAC signature,
writes the raw payload to PostgreSQL and returns 200 immediately. Everything else happens
off the request path.

A BullMQ queue fans the payload out to workers. Each message is keyed by
`tenant_id + provider_message_id`, so a provider that retries the same webhook six times
still produces one job. Redis holds the deduplication key for 72 hours.

Failed deliveries retry with exponential backoff: 30 seconds, 2 minutes, 10 minutes,
1 hour, then 6 hours. After the fifth failure the job moves to a dead-letter queue and the
tenant owner gets an email. Every state change is appended to an audit log table, because
support needs to answer "did this message arrive" without reading worker logs.
