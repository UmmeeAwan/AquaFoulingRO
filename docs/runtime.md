# Runtime and dependency notes

## Tested local runtime

| Item | Value |
| --- | --- |
| Python (venv) | **3.12.3** |
| Docker base image | `python:3.12-slim-bookworm` |
| CI (`/.github/workflows/ci.yml`) | `python-version: "3.12"` |

Python 3.13 was requested in the original brief but was **not installed** on the development host. The project constraint remains `python = ">=3.12,<3.15"`. Versions other than 3.12 are **untested** unless CI is extended to matrix them.

## CLI help verification

Verified locally:

```text
poetry run aqua --help
```

Phase 3 commands include: `readiness-quant`, `generate-synthetic`, `benchmark-synthetic`, `benchmark-ocwd`, `benchmark-summary`, plus Phase 2 `audit-data`, `validate`, `normalise`, `readiness-report`, `serve`.

## Typer / Click compatibility pin

| Package | Locked version |
| --- | --- |
| typer | 0.15.3 |
| click | 8.1.8 |

**Reason for pin (observed, not invented):** With Click 8.5.x installed, Typer help/parsing failed with:

`TypeError: Parameter.make_metavar() missing 1 required positional argument: 'ctx'`

`pyproject.toml` therefore constrains `click = ">=8.1.7,<8.2"`. This is an observed compatibility constraint between the installed Typer and Click releases on this machine; do not treat it as a claim about every Click 8.2+ release without re-testing.
