# Tool-selection eval

Ten natural-language questions, asked in Claude Desktop with nba-scout-mcp
connected, checking whether the agent picks the right tool from its
description alone (brief Section 5.4). This has to be run interactively
in Claude Desktop, so it isn't automated; fill in the "Observed" column
after connecting the server (see README Quickstart) and asking each
question in a fresh conversation.

| # | Question | Expected tool(s) | Observed |
|---|---|---|---|
| 1 | "Who's the better rebounding prospect, Trae Young or Jalen Brunson?" | `compare_prospects` | _not yet run_ |
| 2 | "Is Trae Young in your training data or was he held out?" | `search_players` | _not yet run_ |
| 3 | "Give me a full scouting profile on Jalen Brunson." | `get_skill_profile` | _not yet run_ |
| 4 | "Why does the model think Brunson is a good playmaker?" | `explain_tier` | _not yet run_ |
| 5 | "In the 2018 draft, who was the most undervalued playmaker relative to where they were picked?" | `draft_class_board` | _not yet run_ |
| 6 | "I found a current college player averaging 18 PPG, 6 AST on good efficiency. How would the model grade him as a playmaker?" | `project_custom_prospect` | _not yet run_ |
| 7 | "Who else plays like Jalen Brunson?" | `find_comps` | _not yet run_ |
| 8 | "How accurate is this model, and what are its weaknesses?" | `get_model_card` (or `scout://model-card`) | _not yet run_ |
| 9 | "What are the 5 player archetypes the model found?" | `scout://archetypes` resource | _not yet run_ |
| 10 | "Write me a scouting report on Trae Yung." (misspelled) | `search_players` first (resolves the typo), then `get_skill_profile` / `scouting_report` prompt | _not yet run_ |

## Notes for whoever fills this in

- A wrong tool call isn't necessarily a failure if the agent recovers
  (e.g. calls `get_skill_profile` directly on an exact name without
  `search_players` first); note that in the Observed column rather than
  marking it a hard fail.
- If the agent never calls a tool for a question that should need one,
  that's a real miss on the tool description, and the description should
  be tightened (see brief Section 5.4: "say when to use the tool and when
  not to").
