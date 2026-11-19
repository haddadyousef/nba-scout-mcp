"""nba-scout-mcp: an MCP server exposing the Shopping by Skill NBA draft model as an expert lens."""


def main() -> None:
    from scout_mcp.server import main as _main

    _main()


__all__ = ["main"]
