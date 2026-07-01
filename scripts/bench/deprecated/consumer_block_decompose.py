#!/usr/bin/env python3
"""STEP-0 discriminator (b): decompose the consumer thread's block into
DECODE-WAIT vs SERIAL-BOOKKEEPING from a Chrome-trace JSON.

Computes SELF-time per span on the consumer tid (default 1) by subtracting the
duration of nested children (no double-count, CLAUDE.md rule 8). Then:
  DECODE-WAIT        = self(wait.block_fetcher_get) + self(wait.future_recv)
  SERIAL-BOOKKEEPING = consumer_wall - DECODE-WAIT
  consumer_wall      = last E ts - first B ts on the consumer tid

Also reports per-span self-time so the serial bucket is itemized.

Usage: consumer_block_decompose.py <trace.json> [consumer_tid]
Conservation check: sum of all self-times on the tid must equal consumer_wall
(within float epsilon) — else the trace has unbalanced B/E and the number is void.
"""
import json
import sys

# Spans that are BLOCKING WAITS on a worker decode (shrink when the engine
# speeds up). Classified by reading the source (chunk_fetcher.rs / block_fetcher.rs):
#   wait.block_fetcher_get  : get_with_prefetch cold get, blocks on rx.recv (1374)
#   wait.future_recv        : blocks on rx.recv of a chunk future (1662, 3501)
#   ttp.rx_recv_block       : try_take_prefetched rx.recv_timeout pump loop (block_fetcher 247)
#   consumer.dispatch_recv  : Deferred-arc inflight rx.recv() (chunk_fetcher 3051)
# Everything else on the consumer tid is SERIAL CPU bookkeeping.
WAIT_SPANS = {
    "wait.block_fetcher_get",
    "wait.future_recv",
    "ttp.rx_recv_block",
    "consumer.dispatch_recv",
}


def main():
    path = sys.argv[1]
    tid = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    # The trace is a possibly-unterminated Chrome-trace JSON array (streaming
    # writer leaves the closing ']' off and a trailing comma). Parse the inner
    # object records line-by-line, tolerant of the missing terminator.
    events = []
    with open(path) as f:
        raw = f.read()
    try:
        events = json.loads(raw)
    except json.JSONDecodeError:
        for line in raw.splitlines():
            s = line.strip().lstrip("[").rstrip(",").strip()
            if not s or s == "]":
                continue
            try:
                events.append(json.loads(s))
            except json.JSONDecodeError:
                pass

    evs = [e for e in events if e.get("tid") == tid and e.get("ph") in ("B", "E")]
    evs.sort(key=lambda e: e["ts"])

    # Stack-based self-time: when a child is open, its duration is attributed to
    # the child; the parent's self-time excludes it.
    stack = []  # entries: [name, begin_ts, child_total]
    self_time = {}  # name -> summed self time
    total_dur = {}  # name -> summed inclusive dur (sanity)
    first_b = None
    last_e = None
    errors = 0

    for e in evs:
        ts = e["ts"]
        name = e["name"]
        if e["ph"] == "B":
            if first_b is None:
                first_b = ts
            stack.append([name, ts, 0.0])
        else:  # E
            last_e = ts
            if not stack:
                errors += 1
                continue
            cname, cbegin, cchild = stack.pop()
            if cname != name:
                # tolerate by matching the most recent same-named open frame
                errors += 1
            dur = ts - cbegin
            selfd = dur - cchild
            self_time[cname] = self_time.get(cname, 0.0) + selfd
            total_dur[cname] = total_dur.get(cname, 0.0) + dur
            if stack:
                stack[-1][2] += dur  # attribute full child dur to parent

    # Chrome-trace ts is in MICROSECONDS. Convert to seconds for the gate.
    US = 1_000_000.0
    consumer_wall = (last_e - first_b) if (first_b is not None and last_e is not None) else 0.0
    sum_self = sum(self_time.values())

    decode_wait = sum(self_time.get(s, 0.0) for s in WAIT_SPANS)
    serial = consumer_wall - decode_wait

    print(f"# trace: {path}  consumer_tid={tid}")
    print(f"# unbalanced B/E events (should be 0): {errors}; leftover open frames: {len(stack)}")
    print(f"consumer_wall_s         = {consumer_wall/US:.4f}")
    print(f"sum_self_time_s         = {sum_self/US:.4f}  (conservation: should ~= consumer_wall)")
    print(f"conservation_gap_s      = {(consumer_wall - sum_self)/US:.4f}  ({100*(consumer_wall-sum_self)/consumer_wall:.2f}% of wall)")
    print()
    print(f"DECODE_WAIT_s           = {decode_wait/US:.4f}  ({100*decode_wait/consumer_wall:.1f}% of wall)")
    print(f"SERIAL_BOOKKEEPING_s    = {serial/US:.4f}  ({100*serial/consumer_wall:.1f}% of wall)")
    print(f"   ^^ the NON-DECODE FLOOR (gate vs 0.54s)")
    print()
    print("# per-span SELF-time on consumer tid (s), sorted:")
    for name, st in sorted(self_time.items(), key=lambda kv: -kv[1]):
        tag = "  [DECODE-WAIT]" if name in WAIT_SPANS else "  [serial]"
        print(f"  {st/US:10.4f}  {name}{tag}")


if __name__ == "__main__":
    main()
