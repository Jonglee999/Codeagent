# Tool retrieval architecture

CodeAgent uses a bounded retrieval funnel instead of treating repository search as an
unrestricted terminal operation. The public REST and WebSocket contracts are unchanged;
the policy is implemented inside the context, capability-selection, and tool layers.

## Layers

1. **Cacheable static prefix.** Execution prompts place stable behavior, selected tool
   descriptions, and the retrieval policy before `SYSTEM_PROMPT_DYNAMIC_BOUNDARY`.
   Repository context and memory follow the boundary.
2. **Parallel pre-injection.** Before the first planning or execution model call, semantic
   repository context and persistent memory are retrieved concurrently. At most five code
   snippets are injected by default (`CONTEXT_PREINJECT_MAX_FILES`). Large repository indexes
   may continue in the background.
3. **Model-driven funnel.** The normal sequence is `list_files` for paths, `search_code` for
   exact or ranked matches, `navigate_code` for definitions/references, and `read_file` for
   bounded source ranges. These are read-only and workspace-confined.
4. **Context firewall.** `search_code(search_type="explore")` performs up to six lexical
   probes plus available semantic and AST retrieval internally. Raw candidates remain inside
   the tool. The model receives at most five ranked files and 24 numbered excerpt lines per
   file. This deterministic path adds no summarizer-model call; a cheaper summarizer can be
   attached later without changing the tool contract.

## Guardrails and token controls

- Regex/literal search has a hard 250-result head limit and a byte limit.
- All model-facing repository paths are normalized relative to the workspace.
- Exact reads and subranges covered by a wider prior read are reused while the file
  fingerprint is unchanged. Writes invalidate only the affected file; terminal and Git
  operations conservatively invalidate all observations.
- Semantic indexing is optional. AST and ripgrep fallbacks remain available while the index
  is absent, degraded, or building.
- `navigate_code` prefers the tree-sitter symbol index for definitions and falls back to a
  bounded whole-word ripgrep query. Reference lookup uses the same bounded fallback until a
  persistent language-server transport is configured.

## Why Explore is deterministic

A model-backed sub-agent is useful for very ambiguous investigations, but making it the only
heavy-search path adds latency, provider coupling, and another failure mode. CodeAgent first
uses deterministic fusion as the always-available context firewall. A future inexpensive
explorer model should consume the same internal candidate set and return the same bounded
`files` shape, so callers and API contracts do not change.

## Observability

The context manifest reports the pre-injection strategy, code snippet count, memory item count,
and configured limit. Tool metrics expose calls and latency; `working_set` diagnostics expose
read requests, hits, misses, invalidations, and hit rate. Explore results explicitly report the
number of raw candidates considered and confirm that raw results were not returned.
