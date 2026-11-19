"""Manual stdio-client smoke test: spawns the server as a subprocess over
stdio (the real transport Claude Desktop uses) and exercises the P0 tools.
Not part of the test suite — ad hoc verification only."""

import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parent.parent


async def main():
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "scout_mcp.server"],
        cwd=str(ROOT),
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()

        tools = await session.list_tools()
        print("TOOLS:", [t.name for t in tools.tools])

        resources = await session.list_resources()
        print("RESOURCES:", [str(r.uri) for r in resources.resources])

        prompts = await session.list_prompts()
        print("PROMPTS:", [p.name for p in prompts.prompts])

        result = await session.call_tool("search_players", {"query": "Tray Yung"})
        print("search_players:", result.content[0].text[:300])

        result = await session.call_tool("get_skill_profile", {"player": "Trae Young"})
        payload = json.loads(result.content[0].text)
        print("get_skill_profile confidence:", payload["confidence"])
        print("get_skill_profile summary:", payload["summary"])

        result = await session.call_tool(
            "draft_class_board", {"year": 2018, "skill": "Playmaking", "top_n": 20}
        )
        payload = json.loads(result.content[0].text)
        print("draft_class_board summary:", payload["summary"])
        names = [r["player"] for r in payload["answer"]["board"]]
        print("Brunson on board:", "Jalen Brunson" in names)

        result = await session.call_tool(
            "project_custom_prospect",
            {"stats": {"PTS": 60, "FGA": 5, "3PA": 40, "Games": 30}, "name": "Nonsense Stat Line"},
        )
        payload = json.loads(result.content[0].text)
        print("project_custom_prospect confidence:", payload["confidence"])

        resource_result = await session.read_resource("scout://model-card")
        print("model-card resource ok:", bool(resource_result.contents))


if __name__ == "__main__":
    asyncio.run(main())
