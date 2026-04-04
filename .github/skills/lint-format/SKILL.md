---
name: lint-format
description: Use when need to lint, format, or type-check Python code using tools like Black, Flake8, Mypy, isort, or Ruff with uv
---

Use `uv/astral` tools functionality to run code quality tools without installing them globally.

## Black — Code Formatting

Black enforces a consistent, opinionated code style.

- **Verify available Black commands**:

```sh
uv run --with black black --help
```

- **Format all Python files in the current directory**:

```sh
uv run --with black black .
```

- **Format a specific file or directory**:

```sh
uv run --with black black path/to/file.py
```

- **Check formatting without applying changes (dry-run)**:

```sh
uv run --with black black --check .
```

- **Show a diff of what would change**:

```sh
uv run --with black black --diff .
```

---

## Flake8 — Linting

Flake8 checks for PEP 8 style violations, unused imports, and common errors.

- **Verify available Flake8 commands**:

```sh
uv run --with flake8 flake8 --help
```

- **Lint all Python files in the current directory**:

```sh
uv run --with flake8 flake8 .
```

- **Lint a specific file**:

```sh
uv run --with flake8 flake8 path/to/file.py
```

- **Set max line length**:

```sh
uv run --with flake8 flake8 --max-line-length 120 .
```

- **Ignore specific error codes**:

```sh
uv run --with flake8 flake8 --extend-ignore=E501,W503 .
```

---

## Mypy — Static Type Checking

Mypy checks Python type annotations for correctness.

- **Verify available Mypy commands**:

```sh
uv run --with mypy mypy --help
```

- **Type-check all Python files in the current directory**:

```sh
uv run --with mypy mypy .
```

- **Type-check a specific file or module**:

```sh
uv run --with mypy mypy path/to/file.py
```

- **Strict mode (enables all optional checks)**:

```sh
uv run --with mypy mypy --strict .
```

- **Ignore missing stubs for third-party libraries**:

```sh
uv run --with mypy mypy --ignore-missing-imports .
```

---

## isort — Import Sorting

isort automatically sorts and organizes Python imports.

- **Sort imports in the current directory**:

```sh
uv run --with isort isort .
```

- **Check without applying changes**:

```sh
uv run --with isort isort --check-only .
```

- **Compatible with Black profile**:

```sh
uv run --with isort isort --profile black .
```

---

## Ruff — Fast All-in-One Linter & Formatter

Ruff replaces Flake8, isort, and partially Black — significantly faster.

- **Lint the current directory**:

```sh
uv run --with ruff ruff check .
```

- **Lint and auto-fix fixable issues**:

```sh
uv run --with ruff ruff check --fix .
```

- **Format code (Black-compatible)**:

```sh
uv run --with ruff ruff format .
```

- **Check formatting without applying changes**:

```sh
uv run --with ruff ruff format --check .
```

- **Show all available rules**:

```sh
uv run --with ruff ruff rule --all
```
