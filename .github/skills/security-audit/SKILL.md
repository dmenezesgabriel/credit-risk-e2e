---
name: security-audit
description: Use when need to perform security audits, scan for vulnerabilities in code or dependencies using semgrep and pip-audit with uv
---

Use `uv/astral` tools functionality to run security audit tools without installing them globally.

## Semgrep — Static Analysis / Code Scanning

Semgrep scans source code for security vulnerabilities, bugs, and anti-patterns.

- **Verify available Semgrep commands**:

```sh
uv run --with semgrep semgrep --help
```

- **Scan current directory with auto-detected rules** (recommended starting point):

```sh
uv run --with semgrep semgrep --config=auto .
```

- **Scan a specific path**:

```sh
uv run --with semgrep semgrep --config=auto path/to/code
```

- **Use a specific ruleset (e.g. OWASP Top 10)**:

```sh
uv run --with semgrep semgrep --config=p/owasp-top-ten .
```

- **Output results as JSON** (useful for CI or further processing):

```sh
uv run --with semgrep semgrep --config=auto --json . > semgrep-results.json
```

- **Scan only Python files**:

```sh
uv run --with semgrep semgrep --config=auto --include="*.py" .
```

## pip-audit — Dependency Vulnerability Scanning

pip-audit checks Python dependencies for known CVEs using the PyPI Advisory Database.

- **Verify available pip-audit commands**:

```sh
uv run --with pip-audit pip-audit --help
```

- **Audit dependencies in the current environment**:

```sh
uv run --with pip-audit pip-audit
```

- **Audit a specific requirements file**:

```sh
uv run --with pip-audit pip-audit -r requirements.txt
```

- **Output results as JSON**:

```sh
uv run --with pip-audit pip-audit --format json -o pip-audit-results.json
```

- **Fix vulnerabilities automatically where possible**:

```sh
uv run --with pip-audit pip-audit --fix
```
