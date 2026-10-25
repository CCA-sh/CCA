# Security policy

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's
[private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
on this repository (**Security → Report a vulnerability**). Do not open a public issue.

Only the latest release receives security fixes.

## Dependency auditing

Every push, pull request, and a weekly scheduled run audit both lockfiles:

- Python: `pip-audit` against the production resolution of `uv.lock`, including the
  `observability` extra.
- Website: `npm audit --omit=dev` against `website/package-lock.json`.

Dependabot proposes weekly updates for both ecosystems and monthly updates for GitHub
Actions. When an advisory affects a version that `pyproject.toml` or
`website/package.json` still permits, the direct minimum is raised so fresh installs
cannot resolve to it.

To run the same audits locally:

```bash
uv export --locked --no-dev --all-extras --no-emit-project --no-hashes -o /tmp/requirements-audit.txt
uvx pip-audit -r /tmp/requirements-audit.txt --no-deps --disable-pip
(cd website && npm audit --omit=dev)
```

## Accepted advisories

Advisories that cannot be fixed yet are listed here with the reason they are not
exploitable in CCA, and are passed to the audit with `--ignore-vuln <id>`.

None at present.
