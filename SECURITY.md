# Security policy

## Reporting a vulnerability

Please do not open a public issue for a security problem.

Report it privately through GitHub: open the
[Security tab](https://github.com/araaking/ingatan0-mem/security) of this repository and
choose **Report a vulnerability**. Only the maintainer can read the report.

Include what you can of the following:

- the version or commit you tested,
- the mode (local stdio or HTTP mode),
- the steps to reproduce the problem,
- what an attacker could read, change or delete.

## What counts

Ingatan stores private memory, so these are treated as security problems:

- reading or changing facts of another business or another agent,
- getting past the bearer-token check of HTTP mode, or learning a token from a log, an
  error message or a response,
- text that survives `forget` in the database files,
- a secret that the redaction step should have removed and did not,
- any outbound network call made by the server.

## Supported versions

Only the latest release and the `main` branch receive fixes.

## Deployment note

HTTP mode does not encrypt traffic by itself. Run it behind a reverse proxy with TLS or
inside a private network. See [docs/setup-http.md](docs/setup-http.md).
