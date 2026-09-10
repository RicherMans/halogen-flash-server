#!/usr/bin/env python3
"""deploy/add-llama-swap-bridge.py: re-attach the fork's bridge to entrypoint.sh.

The sync-upstream workflow merges upstream with `-X theirs`, which for
conflicted files keeps the upstream (non-fork) version. Upstream has no
knowledge of the llama-swap bridge, so it is dropped by any real collision in
deploy/entrypoint.sh. This script re-inserts the bridge block and re-wires the
`all` role's wait-kill set onto whatever entrypoint.sh is on disk.

It is idempotent: a file that already carries the block is left untouched.

Fails loudly (non-zero) if the anchors the block is keyed to have moved, so
the sync workflow notices rather than silently shipping a handout without the
fork feature.
"""
import sys

FILE = "deploy/entrypoint.sh"

MARKER = "HALOGEN_LLAMA_SWAP"

_BLOCK = """  # THE LLAMA-SWAP BRIDGE IS ON BY DEFAULT in this fork. It is a pass-through
  # proxy for the api that adds the rate block llama-swap needs to show
  # tokens/s, plus a Prometheus /metrics on the same port. With it on, a
  # client's -p maps to the BRIDGE on HALOGEN_LLAMA_SWAP_PORT (e.g. -p 8732:8732);
  # the api stays on $API_PORT inside the container. Set HALOGEN_LLAMA_SWAP=0
  # for an upstream-identical container, where -p maps to the api directly.
  LLAMA_SWAP_PID=""
  if [ "${HALOGEN_LLAMA_SWAP:-1}" != "0" ]; then
    python3 /usr/local/bin/llama-swap-bridge.py \\
      --listen "0.0.0.0:${HALOGEN_LLAMA_SWAP_PORT:-8732}" \\
      --upstream "127.0.0.1:$API_PORT" &
    LLAMA_SWAP_PID=$!
    echo "halogen: llama-swap bridge on ${HALOGEN_LLAMA_SWAP_PORT:-8732} (tokens/s for a llama-swap front-end)"
  fi
"""

_ANCHOR = "  API_PID=$!\n"
_SENTINEL = "Either process exiting must take the container down"

_HEADER_NOTE = (
    "#   llama-swap bridge   (fork): ON by default; the api is proxied on\n"
    "#            HALOGEN_LLAMA_SWAP_PORT (8732), adding the rate block llama-swap\n"
    "#            parses so its UI can show tokens/s, plus a Prometheus /metrics on\n"
    "#            the same port. Set HALOGEN_LLAMA_SWAP=0 for an upstream-identical\n"
    "#            container (no bridge; -p maps to the api on $API_PORT).\n"
    "#\n"
)
_HEADER_ANCHOR = "# The engine's token protocol has NO AUTH."

_WAIT = 'wait -n "$ENGINE_PID" "$API_PID" $WATCHDOG_PID\n'
_WAIT_NEW = 'wait -n "$ENGINE_PID" "$API_PID" $WATCHDOG_PID $LLAMA_SWAP_PID\n'
_KILL = 'kill -TERM "$ENGINE_PID" "$API_PID" 2>/dev/null || true\n'
_KILL_NEW = 'kill -TERM "$ENGINE_PID" "$API_PID" $LLAMA_SWAP_PID 2>/dev/null || true\n'


def main():
    try:
        text = open(FILE, encoding="utf-8").read()
    except OSError as exc:
        print(f"add-llama-swap-bridge: cannot read {FILE}: {exc}", file=sys.stderr)
        return 1

    if MARKER in text:
        print(f"add-llama-swap-bridge: {MARKER} present already, leaving entrypoint alone")
        return 0

    lines = text.split("\n")
    anchor_idx = None
    starts = [i for i, ln in enumerate(lines) if ln == _ANCHOR.rstrip("\n")]
    for i in starts:
        if _SENTINEL in "\n".join(lines[i:i + 40]):
            anchor_idx = i
            break
    if anchor_idx is None:
        print("add-llama-swap-bridge: all-mode API_PID anchor not found; "
              "upstream must have restructured the `all` role", file=sys.stderr)
        return 1

    block = "\n".join(_BLOCK.splitlines())
    lines[anchor_idx:anchor_idx + 1] = [
        lines[anchor_idx], block,
    ]
    text = "\n".join(lines) + "\n"

    if _HEADER_ANCHOR in text:
        text = text.replace(
            _HEADER_ANCHOR, _HEADER_NOTE + _HEADER_ANCHOR, 1)

    wait_count = text.count(_WAIT)
    kill_count = text.count(_KILL)
    if wait_count != 1:
        print(f"add-llama-swap-bridge: expected exactly one wait -n line, found {wait_count}",
              file=sys.stderr)
        return 1
    if kill_count != 1:
        print(f"add-llama-swap-bridge: expected exactly one kill line, found {kill_count}",
              file=sys.stderr)
        return 1
    text = text.replace(_WAIT, _WAIT_NEW).replace(_KILL, _KILL_NEW)

    with open(FILE, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"add-llama-swap-bridge: re-inserted the bridge in {FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())