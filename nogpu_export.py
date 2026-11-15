#!/usr/bin/env python3
"""nogpu_export.py: turn LLM decision logs into NoGPU upload rows, locally.

Runs on your machine with the Python standard library only; nothing is sent
anywhere. Emits exactly two things per call, the last HUMAN message and the
decision the model returned, and fails closed: a row whose shape it does not
positively recognise is skipped, never dumped. System prompts, earlier turns,
tool results and request metadata are never written out.

    python3 nogpu_export.py logs.jsonl > nogpu.jsonl          # auto-detect
    python3 nogpu_export.py logs.json --input-field input.query --label-field output.route
    cat logs.jsonl | python3 nogpu_export.py - > nogpu.jsonl

Recognised per row (JSONL or a JSON array; JSON-encoded string fields are
decoded): chat messages under messages / request.messages /
request.body.messages / request_body.messages / input.messages / input
(OpenAI, Anthropic, LangChain "type": "human" turns); the decision from an
OpenAI choices[0] message (content, else the tool call's name or its single
argument), Anthropic / Responses API content blocks, or a plain string or
number under output / response / response_body / completion. Anything else:
point --input-field / --label-field at the exact dotted paths.
Output: one {"id", "text", "label"} per line.
"""
import argparse
import json
import sys

MAX_LABEL_LEN = 200  # the server's limit (pipeline.py)
_WARNED = set()


def _warn_once(msg):
    if msg not in _WARNED:  # GLM P05-7: say so when a system prompt may ride along
        _WARNED.add(msg)
        print(msg, file=sys.stderr)
MSG_PATHS = ("messages", "request.messages", "request.body.messages",
             "request_body.messages", "input.messages", "input")
OUT_PATHS = ("output", "response.body", "response", "response_body", "completion")
SKIP_BLOCKS = ("tool_result", "tool_use", "function_call_output", "image", "image_url",
               "input_image", "thinking", "redacted_thinking", "document")


def decode(v):
    """Logs often store JSON as a string (LiteLLM spend logs, CSV exports)."""
    if isinstance(v, str) and v[:1] in "[{" and v[-1:] in "]}":
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


def dig(obj, path):
    for part in path.split("."):
        obj = decode(obj)
        if isinstance(obj, list) and part.isdigit():
            obj = obj[int(part)] if int(part) < len(obj) else None
        elif isinstance(obj, dict):
            obj = obj.get(part)
        else:
            return None
    return decode(obj)


def block_text(content):
    """Plain text of a message's content; tool results, images, thinking dropped."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [p.get("text") for p in content
                 if isinstance(p, dict) and p.get("type") not in SKIP_BLOCKS
                 and isinstance(p.get("text"), str)]
        return " ".join(t.strip() for t in parts if t.strip())
    return ""


def last_human(messages):
    """Text of the last human turn. A user-role turn that only carries tool
    results is the harness talking, not the human, so the search continues."""
    if not isinstance(messages, list):
        return None
    for m in reversed(messages):
        if isinstance(m, dict) and (m.get("role") or m.get("type")) in ("user", "human"):
            text = block_text(m.get("content"))
            if text:
                return text
    return None


def scalar(v):
    if isinstance(v, bool) or v is None:
        return None if v is None else str(v).lower()
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        return v.strip() or None
    return None


def decision(v):
    """The model's decision, from shapes we recognise; None for anything else."""
    v = decode(v)
    s = scalar(v)
    if s is not None or not isinstance(v, (dict, list)):
        return s
    if isinstance(v, list):  # Anthropic / Responses API content blocks
        return block_text(v) or None
    msg = dig(v, "choices.0.message")
    msg = msg if isinstance(msg, dict) else v  # bare assistant message (Langfuse)
    content = msg.get("content")
    text = block_text(content) if isinstance(content, (str, list)) else ""
    if text:
        return text
    call = dig(msg, "tool_calls.0.function")
    if isinstance(call, dict):
        args = decode(call.get("arguments"))
        if isinstance(args, dict) and len(args) == 1:  # classify(route="billing")
            return scalar(next(iter(args.values())))
        if isinstance(args, dict) and args:  # several arguments: which one is the decision? skip
            return None
        return scalar(call.get("name"))
    out = dig(v, "output.0.content")  # Responses API object
    return block_text(out) or None if isinstance(out, list) else None


def convert(row, i, input_field=None, label_field=None):
    row = decode(row)
    if not isinstance(row, dict):
        return None
    if input_field:
        raw = dig(row, input_field)
        text = last_human(raw) if isinstance(raw, list) else scalar(raw)
    else:
        text = None
        for p in MSG_PATHS:
            text = last_human(dig(row, p))
            if text:
                break
        if not text:  # decoded like every other path; a dict here is unrecognised -> skip
            for key in ("input", "prompt"):
                raw = dig(row, key)
                text = last_human(raw) if isinstance(raw, list) else scalar(raw)
                if text:
                    if key == "prompt" and not isinstance(raw, list):
                        _warn_once("note: a plain 'prompt' string is kept whole; if your logs put the system "
                                   "prompt inside it, re-run with --input-field pointing at the user text only")
                    break
    if label_field:
        label = decision(dig(row, label_field))
    else:
        label = None
        for p in OUT_PATHS:
            got = dig(row, p)
            if got is not None:
                label = decision(got)
                if label:  # an unrecognised field (metadata) does not stop the search
                    break
        if label is None and "choices" in row:
            label = decision(row)
    if not text or not label or len(label) > MAX_LABEL_LEN:
        return None
    rid = row.get("id", row.get("request_id", i))
    if isinstance(rid, bool) or not isinstance(rid, (str, int, float)) or str(rid) == "":
        rid = i  # an id we do not positively recognise is metadata: never written out
    return {"id": str(rid)[:128], "text": text, "label": label}


def rows(path, stats=None):
    """Bytes in, like the server: a BOM is stripped and a torn or non-UTF-8 line is
    counted and skipped instead of crashing halfway through the output."""
    f = sys.stdin.buffer if path == "-" else open(path, "rb")
    with f:
        first = f.read(1)
        if first == b"\xef":  # UTF-8 BOM
            first = first + f.read(2)
            first = b"" if first == b"\xef\xbb\xbf" else first
            first = first or f.read(1)
        while first and first.isspace():
            first = f.read(1)
        if first == b"[":  # a JSON array has to be read whole
            data = first + f.read()
            try:
                yield from json.loads(data)
            except ValueError:  # JSONL whose rows are arrays: read per line, never a traceback (GLM P05-8)
                for raw in data.splitlines():
                    try:
                        yield json.loads(raw)
                    except ValueError:
                        if stats is not None:
                            stats["bad_lines"] = stats.get("bad_lines", 0) + 1
            return
        line = first + f.readline()
        while line:  # JSONL streams: multi-GB logs never sit in memory
            if line.startswith(b"\xef\xbb\xbf"):
                line = line[3:]
            if line.strip():
                try:
                    yield json.loads(line)
                except ValueError:  # torn write or bad UTF-8 (UnicodeDecodeError is a ValueError)
                    if stats is not None:
                        stats["bad_lines"] = stats.get("bad_lines", 0) + 1
            line = f.readline()


SECRET = "SECRET-SYSTEM-PROMPT"
DEMO = [  # every shape a hostile review found (Argon, 2026-11-13), plus the basics
    ({"id": "a", "messages": [{"role": "system", "content": SECRET}, {"role": "user", "content": "refund my order"}],
      "response": {"choices": [{"message": {"content": "refunds"}}]}}, ("refund my order", "refunds")),
    ({"id": "lc", "input": [{"type": "system", "content": SECRET}, {"type": "human", "content": "refund order"}],
      "output": "refunds"}, ("refund order", "refunds")),
    ({"id": "kw", "input": {"system": SECRET, "kwargs": {"api_key": "sk-live-123"}}, "output": "allow"}, None),
    ({"id": "tr", "messages": [{"role": "user", "content": "is this spam?"},
                               {"role": "assistant", "content": [{"type": "tool_use", "name": "lookup"}]},
                               {"role": "user", "content": [{"type": "tool_result", "content": "INTERNAL_DB_ROW"}]}],
      "response": {"content": [{"type": "text", "text": "spam"}]}}, ("is this spam?", "spam")),
    ({"id": "big", "messages": [{"role": "user", "content": "x"}],
      "response": {"id": "r", "usage": {"secret_header": SECRET}}}, None),
    ({"request_id": "ll", "messages": json.dumps([{"role": "system", "content": SECRET}, {"role": "user", "content": "hi"}]),
      "response": json.dumps({"choices": [{"message": {"content": "greet"}}]})}, ("hi", "greet")),
    ({"request_id": "h1", "request_body": {"messages": [{"role": "system", "content": SECRET}, {"role": "user", "content": "cancel"}]},
      "response_body": {"choices": [{"message": {"content": "churn"}}]}}, ("cancel", "churn")),
    ({"id": "p1", "request": {"body": {"messages": [{"role": "user", "content": "cancel"}]}},
      "response": {"body": {"choices": [{"message": {"content": "churn"}}]}}}, ("cancel", "churn")),
    ({"id": "o1", "messages": [{"role": "user", "content": "cancel"}],
      "choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "churn"}}]}}]},
     ("cancel", "churn")),
    ({"id": "one", "messages": [{"role": "user", "content": "where is my card"}],
      "choices": [{"message": {"content": None, "tool_calls": [{"function": {
          "name": "classify", "arguments": json.dumps({"route": "card_arrival"})}}]}}]}, ("where is my card", "card_arrival")),
    ({"id": "lf1", "input": [{"role": "user", "content": "refund"}],
      "output": {"role": "assistant", "content": None, "tool_calls": [{"function": {"name": "refunds"}}]}}, ("refund", "refunds")),
    ({"id": "ws", "messages": [{"role": "user", "content": [{"type": "image_url"}, {"type": "image_url"}]}],
      "output": "x"}, None),
    ({"id": "zero", "input": "legit", "output": 0}, ("legit", "0")),
    ({"id": "long", "input": "q", "output": "w" * 500}, None),
    ({"id": "none", "input": "hello", "output": None}, None),
    ({"id": {"env": {"api_key": "sk-live-123", "system": SECRET}}, "input": "hi", "output": "greet"}, ("hi", "greet")),
    ({"id": "enc", "prompt": json.dumps({"system": SECRET, "user": "refund"}), "completion": "refunds"}, None),
]


def demo():
    for row, want in DEMO:
        got = convert(row, 0)
        assert (got and (got["text"], got["label"])) == (want or None), (row.get("id", row.get("request_id")), got)
        if got:
            assert SECRET not in json.dumps(got) and "INTERNAL_DB_ROW" not in json.dumps(got)
            assert got["text"].strip() and 1 <= len(got["label"]) <= MAX_LABEL_LEN
    assert convert(DEMO[0][0], 1, None, "response.choices.0.message.content")["label"] == "refunds"
    assert convert({"input": {"query": "card?"}, "output": {"route": "card_arrival"}}, 2,
                   "input.query", "output.route") == {"id": "2", "text": "card?", "label": "card_arrival"}
    print(f"demo ok ({len(DEMO)} shapes, no secret leaked)")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("path", nargs="?", help="log file (JSONL or JSON array), or - for stdin")
    p.add_argument("--input-field", help="dotted path to the input, e.g. input.query")
    p.add_argument("--label-field", help="dotted path to the decision, e.g. output.route")
    p.add_argument("--demo", action="store_true", help="run the self-check")
    a = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # the server only takes UTF-8 (GLM r2 P06-S2)
    if a.demo:
        return demo()
    if not a.path:
        if a.input_field or a.label_field:
            p.error("give the log file (or - for stdin) before --input-field / --label-field")
        return demo()
    kept = skipped = 0
    stats = {"bad_lines": 0}
    labels = set()
    for i, r in enumerate(rows(a.path, stats)):
        out = convert(r, i, a.input_field, a.label_field)
        if out is None:
            skipped += 1
            continue
        kept += 1
        labels.add(out["label"])
        print(json.dumps(out, ensure_ascii=False))
    skipped += stats["bad_lines"]
    print(f"kept {kept} rows, {len(labels)} distinct labels, skipped {skipped}", file=sys.stderr)
    if stats["bad_lines"]:
        print(f"warning: {stats['bad_lines']} lines were not valid JSON (torn writes?); "
              "check the end of your log", file=sys.stderr)
    if not kept:
        print("warning: no rows kept; point --input-field / --label-field at your fields", file=sys.stderr)
        sys.exit(1)
    if kept and len(labels) < 2:
        print("warning: fewer than 2 distinct labels; NoGPU needs at least 2.", file=sys.stderr)
    if len(labels) > 200 or (kept >= 20 and len(labels) > kept // 2):
        print("warning: most outputs are distinct, so these look like free text, not a decision. "
              "Point --label-field at the route/class field.", file=sys.stderr)
    if kept and not 40 <= kept <= 10000:
        print("warning: NoGPU accepts 40 to 10,000 rows per upload.", file=sys.stderr)


if __name__ == "__main__":
    main()
