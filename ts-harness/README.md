# ts-harness

A TypeScript MCP client (the official `@modelcontextprotocol/sdk`) that
spawns the Python server over stdio, lists its tools and resources, and
runs the golden eval suite from the project brief: the 2018 draft board
puts Jalen Brunson in the top 20 for Playmaking, a nonsense stat line
abstains, an ambiguous name ("Young") returns candidates instead of a
guess, and every response matches the envelope schema (checked with
zod). Prints a pass/fail table and exits non-zero on any failure, which
is what `.github/workflows/ts-harness.yml` runs on each push.

```powershell
cd ts-harness
npm install
npm run start
```

Needs `artifacts/` already built (`uv run python scripts/export_artifacts.py`
from the repo root; see the main README) and `uv` on `PATH`, since the
harness launches the server with `uv run nba-scout-mcp`.
