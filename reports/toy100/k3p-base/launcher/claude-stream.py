#!/usr/bin/env python3
"""Render Claude stream-json or Grok streaming-messages-json for try-gan.sh.

Reads the event stream on stdin, keeps the raw events in claude-stream.jsonl,
prints one short tail-able line per event, and writes the final assistant
message to final.md so Claude attempts leave the same artifacts as Codex ones.
Never fails the pipeline: try-gan.sh reports the agent's own exit status.
"""

import argparse
import json
from pathlib import Path
import signal
import sys
import time

LIMIT = 200  # Keep logs inspectable: never echo whole tool outputs or files.


def clip(text, limit=LIMIT):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + f"…(+{len(text) - limit} chars)"


def blocks(message):
    content = (message or {}).get("content")
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [b for b in content or [] if isinstance(b, dict)]


def result_text(block):
    content = block.get("content")
    if isinstance(content, list):
        content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
    return content if isinstance(content, str) else json.dumps(content) if content else ""


class Renderer:
    def __init__(self, run_dir, engine='claude'):
        self.run_dir = run_dir
        self.engine = engine
        self.raw = (run_dir / f"{engine}-stream.jsonl").open("a", buffering=1)
        self.started = time.time()
        self.tools = {}
        self.calls = {}
        self.last_text = ""
        self.result = None
        self.closed = False

    def say(self, kind, text=""):
        elapsed = int(time.time() - self.started)
        print(f"[{elapsed // 3600:02d}:{elapsed // 60 % 60:02d}:{elapsed % 60:02d}] {kind:<6} {text}".rstrip(),
              flush=True)

    def line(self, raw):
        try:
            event = json.loads(raw)
            if not isinstance(event, dict):
                raise ValueError("not an event object")
        except ValueError:
            self.say("stderr", clip(raw, 400))
            return
        self.raw.write(json.dumps(event, separators=(",", ":")) + "\n")
        self.event(event)

    def event(self, event):
        kind = event.get("type")
        if kind == "system":
            subtype = event.get("subtype")
            if subtype == "init":
                self.say("init", f"model={event.get('model')} session={event.get('session_id')} "
                                 f"cwd={event.get('cwd')} permission={event.get('permissionMode')}")
            elif subtype not in ("thinking_tokens",):  # Per-turn config noise.
                self.say("system", clip(subtype or event))
        elif kind == "assistant":
            for block in blocks(event.get("message")):
                if block.get("type") == "text" and block.get("text", "").strip():
                    self.last_text = block["text"]
                    self.say("text", clip(block["text"], 400))
                elif block.get("type") == "tool_use":
                    name = block.get("name", "?")
                    args = block.get("input") or {}
                    detail = args.get("command") or args.get("file_path") or args.get("pattern") or args.get("prompt")
                    self.tools[block.get("id")] = (name, time.time())
                    self.calls[name] = self.calls.get(name, 0) + 1
                    self.say("tool", f"{name}: {clip(detail if detail else args)}")
        elif kind == "user":
            for block in blocks(event.get("message")):
                if block.get("type") != "tool_result":
                    continue
                name, started = self.tools.pop(block.get("tool_use_id"), ("tool", None))
                seconds = f"{time.time() - started:.1f}s" if started else "--"
                text = result_text(block)
                self.say("fail" if block.get("is_error") else "ok",
                         f"{name} ({seconds}) {clip(text)}")
        elif kind == "result":
            self.result = event
            self.say("done", f"{event.get('subtype')} turns={event.get('num_turns')} "
                             f"cost=${event.get('total_cost_usd') or 0:.2f} "
                             f"api_error={event.get('api_error_status')}")
        elif kind == "rate_limit_event":
            info = event.get("rate_limit_info") or {}
            if info.get("status") != "allowed":
                self.say("limit", clip(info))
        elif kind == "error":
            self.say("error", clip(event))

    def close(self):
        if self.closed:
            return
        self.closed = True
        final = (self.result or {}).get("result") or self.last_text
        if final:
            (self.run_dir / "final.md").write_text(final.rstrip() + "\n")
        usage = dict(seconds=round(time.time() - self.started, 1), tool_calls=self.calls,
                     completed=self.result is not None)
        for key in ("subtype", "is_error", "num_turns", "total_cost_usd", "duration_ms",
                    "session_id", "api_error_status", "permission_denials", "usage", "modelUsage"):
            if self.result and key in self.result:
                usage[key] = self.result[key]
        (self.run_dir / f"{self.engine}-usage.json").write_text(json.dumps(usage, indent=2) + "\n")
        if self.result is None:
            self.say("end", "stream ended without a result event (timeout or interrupt); "
                            "final.md holds the last assistant message if any")
        self.raw.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--engine", choices=('claude', 'grok'), default='claude')
    args = parser.parse_args()
    renderer = Renderer(args.run_dir, args.engine)
    for received in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(received, lambda *_: (renderer.close(), sys.exit(0)))
    try:
        for raw in sys.stdin:
            raw = raw.strip()
            if raw:
                renderer.line(raw)
    except (KeyboardInterrupt, BrokenPipeError, OSError):
        pass
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
