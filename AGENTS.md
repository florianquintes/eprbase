eprbase is a minimal Python package with no external dependencies and only one entry point: `src/eprbase/__init__.py` (`main()` prints "Hello from eprbuild!").

## Commands (verified)

| Task                    | Command                              |
|-------------------------|--------------------------------------|
| build / dev             | `uv venv && uv sync                 \`  <br>                          `& -e run --script pip install .`   |
| run module              | `uv run python -m eprbase           \`\n                        `     &"`\n                        `-c src/eprbase/__init__.py"\``    |

Notes: use `[project.scripts] -> "eprbase = "eprbase:main"` as CLI entrypoint (`uv run pip install .`) rather than calling main() directly. uv_build is the build backend (>=0.11.28,<0.12.0).

## Python environment
File `.python-version` dictates runtime; add `pyright-langserver`, `ruff-lsp`, etc only if you also need corresponding toolchain (`pyright==x.y.z`). Avoid installing arbitrary packages in root until explicitly needed. Add docs/requirements/linting rules incrementally (only when tests/CIs enforce them).
