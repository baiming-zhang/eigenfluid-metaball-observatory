# Security

The inference server includes request-size, per-IP rate, and concurrency limits.
An HTTPS reverse proxy is still recommended for a permanent public deployment.

Production deployments should set `METABALL_ALLOWED_ORIGINS`, keep concurrency
and rate limits enabled, and monitor CPU, memory, logs, and service availability.
Report security issues privately to the repository owner.
