"""nba-scout-mcp: an MCP server exposing the Shopping by Skill NBA draft model as an expert lens."""

import sys as _sys
from pathlib import Path as _Path

# config.py lives at the repo root (sibling to pyproject.toml, not inside
# the package - see README's Architecture section), so it isn't installed
# as part of this package. Every submodule does a plain `import config`;
# make that resolve regardless of how the server is launched (uv run
# nba-scout-mcp, python -m scout_mcp.server, or a test importing it
# directly) by putting the repo root on sys.path exactly once, here, before
# any submodule that needs it gets imported.
_ROOT = _Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))


def main() -> None:
    from scout_mcp.server import main as _main

    _main()


__all__ = ["main"]
