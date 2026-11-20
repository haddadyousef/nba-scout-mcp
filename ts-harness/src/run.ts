// TypeScript MCP client: spawns the Python server over stdio, lists its
// tools/resources, and runs the golden eval suite from the project brief
// (Section 7): a draft-board check, an out-of-distribution abstain check,
// an ambiguous-name check, and envelope-schema validation (zod) on every
// response. Prints a pass/fail table and exits non-zero on any failure,
// so this is the thing CI runs.

import { fileURLToPath } from "node:url";
import path from "node:path";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { z } from "zod";

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");

const EnvelopeSchema = z.object({
  answer: z.record(z.string(), z.unknown()),
  summary: z.string(),
  confidence: z.enum(["high", "medium", "low", "abstain"]),
  confidence_reasons: z.array(z.string()),
  caveats: z.array(z.string()),
  provenance: z.object({
    prediction_type: z.enum(["out_of_fold", "holdout", "new_input"]),
    model_version: z.string(),
    data_through: z.string(),
  }),
});
type Envelope = z.infer<typeof EnvelopeSchema>;

type CheckResult = { name: string; pass: boolean; detail: string };
const results: CheckResult[] = [];

function record(name: string, pass: boolean, detail: string) {
  results.push({ name, pass, detail });
}

async function callToolEnvelope(client: Client, name: string, args: Record<string, unknown>): Promise<Envelope> {
  const result = await client.callTool({ name, arguments: args });
  if (result.isError) {
    throw new Error(`tool ${name} returned an error: ${JSON.stringify(result.content)}`);
  }
  const structured = (result as { structuredContent?: unknown }).structuredContent;
  const payload = structured ?? JSON.parse((result.content as Array<{ type: string; text?: string }>)
    .find((c) => c.type === "text")!.text!);
  return EnvelopeSchema.parse(payload);
}

async function main() {
  const transport = new StdioClientTransport({
    command: "uv",
    args: ["run", "nba-scout-mcp"],
    cwd: REPO_ROOT,
  });
  const client = new Client({ name: "nba-scout-ts-harness", version: "0.1.0" });

  await client.connect(transport);

  try {
    const tools = await client.listTools();
    record(
      "server exposes all 5 P0 tools",
      ["search_players", "get_skill_profile", "compare_prospects", "explain_tier", "draft_class_board"]
        .every((t) => tools.tools.some((tool) => tool.name === t)),
      `tools: ${tools.tools.map((t) => t.name).join(", ")}`,
    );

    const resources = await client.listResources();
    record(
      "server exposes scout:// resources",
      resources.resources.some((r) => r.uri === "scout://model-card") &&
        resources.resources.some((r) => r.uri === "scout://archetypes"),
      `resources: ${resources.resources.map((r) => r.uri).join(", ")}`,
    );

    // --- golden check 1: draft_class_board(2018, "Playmaking") puts Brunson in the top 20 ---
    try {
      const board = await callToolEnvelope(client, "draft_class_board", {
        year: 2018,
        skill: "Playmaking",
        top_n: 20,
      });
      const names = (board.answer.board as Array<{ player: string }>).map((r) => r.player);
      record(
        "draft_class_board(2018, Playmaking) puts Brunson in the top 20",
        names.includes("Jalen Brunson"),
        `top 20: ${names.join(", ")}`,
      );
    } catch (e) {
      record("draft_class_board(2018, Playmaking) puts Brunson in the top 20", false, String(e));
    }

    // --- golden check 2: an out-of-distribution stat line returns abstain ---
    try {
      const result = await callToolEnvelope(client, "project_custom_prospect", {
        stats: { PTS: 60, FGA: 5, "3PA": 40, Games: 30 },
        name: "Nonsense Stat Line",
      });
      record(
        "an out-of-distribution stat line returns abstain",
        result.confidence === "abstain",
        `confidence: ${result.confidence}`,
      );
    } catch (e) {
      record("an out-of-distribution stat line returns abstain", false, String(e));
    }

    // --- golden check 3: an ambiguous name returns candidates instead of guessing ---
    try {
      const result = await callToolEnvelope(client, "get_skill_profile", { player: "Young" });
      const candidates = result.answer.candidates as string[] | undefined;
      record(
        "an ambiguous name returns candidates instead of guessing",
        result.confidence === "abstain" && Array.isArray(candidates) && candidates.length > 1,
        `confidence: ${result.confidence}, candidates: ${JSON.stringify(candidates)}`,
      );
    } catch (e) {
      record("an ambiguous name returns candidates instead of guessing", false, String(e));
    }

    // --- golden check 4: every response above already matched the envelope schema ---
    // (EnvelopeSchema.parse throws inside callToolEnvelope on any mismatch, which
    // would have already failed the relevant check above.)
    record(
      "every response matched the envelope schema (zod)",
      results.every((r) => r.pass || !r.detail.includes("ZodError")),
      "no ZodError observed",
    );
  } finally {
    await client.close();
  }
}

function printTable() {
  const width = Math.max(...results.map((r) => r.name.length));
  console.log("");
  for (const r of results) {
    const status = r.pass ? "PASS" : "FAIL";
    console.log(`[${status}] ${r.name.padEnd(width)}  ${r.detail}`);
  }
  const passed = results.filter((r) => r.pass).length;
  console.log(`\n${passed}/${results.length} checks passed.`);
}

main()
  .then(() => {
    printTable();
    process.exit(results.every((r) => r.pass) ? 0 : 1);
  })
  .catch((e) => {
    console.error("harness crashed:", e);
    printTable();
    process.exit(1);
  });
