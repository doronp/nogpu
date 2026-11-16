# NoGPU: take your LLM's routine decisions off the LLM

If you call an LLM to pick from a fixed list of labels (intent routing, ticket triage, tool
choice, allow/deny checks, document or email classification), most of those calls are easy.
NoGPU trains a small CPU model on a few thousand of your LLM's past decisions. The model
answers alone only when it is sure and hands the rest back to your LLM.

**Free report in about a minute** on your own data: what share of decisions the CPU model takes
alone, how often it is right when it does (with an exact Clopper-Pearson 95% lower bound), and
latency. Data deleted within 24 hours. $99 one-time to download the model and a small serve
script; it runs offline on any CPU, no account, no API.

Measured example: on 3,000 rows of the public Banking77 set (77 intents), the $99 CPU model answered
66.7% of requests alone, 94.1% accurate when it did (one-sided 95% lower bound 91.4%), about 4 ms
per decision.

## 1. Export your logs (local, stdlib only)

```bash
python3 nogpu_export.py your-llm-logs.jsonl > nogpu.jsonl
```

Reads OpenAI chat completions (stored completions joined with their messages, fine-tune JSONL,
the Responses API), Anthropic Messages replies (tool_use included), Langfuse observations, traces
and API pages, LangSmith runs, and Helicone exports made with `--include-body`, and writes
`{"id", "text", "label"}` lines: the last human message and the decision the LLM returned.
Label spellings that differ only in case, `_` vs space, a `Category:` prefix or trailing
punctuation are merged, and every merge is listed.
System prompts, earlier turns and tool results are never written out. Unrecognised rows are
skipped. Use `--input-field` / `--label-field` for custom shapes; `--demo` runs the self-check.

Already have `{"text": ..., "label": ...}` rows (160 to 10,000; JSONL, a JSON array, or a CSV with
text and label columns)? Skip this step. Fewer than 160 rows can't pass the 90% bar even when every
answer is right.

## 2. Get the report

In the browser: upload `nogpu.jsonl` at https://nogpu.dev

Or from a script or an agent:

```bash
curl -s -F file=@nogpu.jsonl https://nogpu.dev/api/jobs
# -> {"job_id": "...", "claim_token": "...", "job_url": "/job/..."}
curl -s -H "X-Claim-Token: TOKEN" https://nogpu.dev/api/jobs/JOB_ID   # poll until the report is ready
curl -s -X DELETE -H "X-Claim-Token: TOKEN" https://nogpu.dev/api/jobs/JOB_ID   # delete now
```

Open `https://nogpu.dev/job/JOB_ID#TOKEN` to see the report and the checkout link.

- How it works: https://nogpu.dev/how
- Example report: https://nogpu.dev/demo
- Agents: MCP server at https://github.com/doronp/nogpu-mcp

The exporter in this repo is MIT licensed.
