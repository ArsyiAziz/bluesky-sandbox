# Verification and diagnostics

Performance data is the most common missing requirement after a fresh install. BlueSky includes OpenAP out of the box, but proprietary data like BADA requires a separate license and is not bundled. Referencing BADA aircraft without it causes runtime failures during environment construction.

Run the diagnostic utility to audit your setup:

```bash
python -m bluesky_sandbox.doctor
# or
bluesky-sandbox-doctor
```

- Environment status: Reports the configured performance model, resolvable aircraft types, BADA installation status, and active lookup paths.
- CI/CD integration: Exits with a non-zero status code when setup issues are detected, making it ideal for build pipelines and container health checks.

See [Performance models](api/performance.md) for the programmatic equivalents, including
{func}`~bluesky_sandbox.sim.performance.bada.bada_available` and
{func}`~bluesky_sandbox.sim.performance.models.available_types`.
