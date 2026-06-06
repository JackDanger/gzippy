#!/bin/bash
# patch_model_params.sh — instrument rapidgzip with the MINIMAL set of spans
# that populate plans/parallel-sm-model.md's discriminating parameters:
#
#   d_c / d_w  — per-chunk worker decode latency, split by decode mode
#                (clean iff predecessor window present at decode-START, else
#                 window-absent). Emitted as a `worker.decode` B/E span tagged
#                {start_bit, mode}. This MIRRORS gzippy's worker.decode span
#                (chunk_fetcher.rs run_decode_task) so Fulcrum's `model` view
#                reads d_c/d_w from BOTH tools the same way.
#
#   L_resolve  — per-chunk tail-window publish on the in-order consumer path.
#                Emitted as a `causal.window_publish` instant carrying
#                {start_bit, end_bit, site, had_markers}, byte-identical in
#                shape to gzippy's causal.window_publish. The inter-event gap
#                between consecutive consumer-ordered publishes IS L_resolve.
#
#   frontier   — first window_publish ts.
#   tail       — last window_publish ts -> EOF (process end).
#
# Vendor sites (vendor/rapidgzip/librapidarchive/src/rapidgzip/GzipChunkFetcher.hpp):
#   - decodeBlock(blockOffset,nextBlockOffset) [~693]  : worker decode entry
#     (sharedWindow presence at ~712 == clean/window-absent predicate).
#   - queueChunkForPostProcessing [~553] m_windowMap->emplace [~572] : the
#     in-order consumer tail-window publish ("critical path that cannot be
#     parallelized", vendor comment ~559-561) == the L_resolve anchor.
#
# Idempotent: re-running is a no-op once the markers are present.
# Pairs with patch_vendor_phase3b.sh (consumer write/crc/future-recv) and
# patch_vendor.sh (TraceV2.hpp install). Run AFTER those.
set -e

VENDOR=${VENDOR:-/root/gzippy/vendor/rapidgzip/librapidarchive/src}
GCF="$VENDOR/rapidgzip/GzipChunkFetcher.hpp"

if [ ! -f "$GCF" ]; then
    echo "ERROR: $GCF not found (set VENDOR=...)" >&2
    exit 1
fi

# Ensure TraceV2.hpp is included (patch_vendor.sh normally does this).
if ! grep -q "TraceV2.hpp" "$GCF"; then
    sed -i '/^#pragma once/a\
#include <core/TraceV2.hpp>' "$GCF"
    echo "PATCHED include: $GCF"
fi

# ---- d_c / d_w : worker.decode span around the instance decodeBlock body ----
# The instance method decodeBlock(blockOffset, nextBlockOffset) [~693] is the
# per-worker decode entry. It computes `sharedWindow` from m_windowMap->get(
# blockOffset) [~712]. Open a worker.decode span right after that lookup so the
# mode tag uses the SAME window-present-at-decode-START predicate gzippy uses,
# and let it cover the `return decodeBlock(...)`.
#
# We wrap by: (a) inserting the span declaration after the sharedWindow block,
# (b) the span is a stack ScopedSpan; the function returns immediately after,
# so the dtor fires at return == decode end. The BGZF synthetic-window branch
# at ~714 also sets sharedWindow, so the predicate is evaluated AFTER it.
if ! grep -q 'worker.decode' "$GCF"; then
    perl -i -0777 -pe '
        s|(sharedWindow = std::make_shared<WindowMap::Window>\(\);\n        \}\n\n)(        return decodeBlock\()|$1        ::tracev2::ScopedSpan _tv2_dec("worker.decode",\n            ([\&]{ static thread_local char _b[96];\n              const bool _hasWin = static_cast<bool>(sharedWindow);\n              std::snprintf(_b, sizeof(_b),\n                "\\"start_bit\\":%zu,\\"mode\\":\\"%s\\"",\n                blockOffset, _hasWin ? "clean" : "window_absent");\n              return _b; }()));\n\n$2|s;
    ' "$GCF"
    if grep -q 'worker.decode' "$GCF"; then
        echo "PATCHED worker.decode (d_c/d_w): $GCF"
    else
        echo "ERROR: worker.decode patch did not match $GCF" >&2
        exit 1
    fi
fi

# ---- L_resolve : causal.window_publish at the consumer tail-window emplace ----
# queueChunkForPostProcessing [~553] emplaces the chunk's tail window at
# windowOffset = encodedOffsetInBits + encodedSizeInBits [~557]. This is the
# in-order consumer publish (vendor: "critical path that cannot be
# parallelized"). Emit a causal.window_publish instant right after BOTH emplace
# branches (empty-footer-window ~570 and getLastWindow ~572) so every publish
# is captured, carrying start_bit/end_bit/site/had_markers shaped exactly like
# gzippy's event.
if ! grep -q 'causal.window_publish' "$GCF"; then
    perl -i -0777 -pe '
        s|(\n)(                m_windowMap->emplaceShared\( windowOffset, std::make_shared<WindowMap::Window>\(\) \);)|$1$2\n                \{ static thread_local char _wb[160]; std::snprintf(_wb, sizeof(_wb), "\\"start_bit\\":%zu,\\"end_bit\\":%zu,\\"site\\":\\"consumer_footer\\",\\"had_markers\\":%s", chunkData->encodedOffsetInBits, windowOffset, chunkData->containsMarkers() ? "true" : "false"); ::tracev2::emit_instant("causal.window_publish", _wb, '"'"'g'"'"'); \}|;
        s|(\n)(                m_windowMap->emplace\( windowOffset, chunkData->getLastWindow\( \*previousWindow \),\n                                      CompressionType::NONE \);)|$1$2\n                \{ static thread_local char _wb[160]; std::snprintf(_wb, sizeof(_wb), "\\"start_bit\\":%zu,\\"end_bit\\":%zu,\\"site\\":\\"consumer\\",\\"had_markers\\":%s", chunkData->encodedOffsetInBits, windowOffset, chunkData->containsMarkers() ? "true" : "false"); ::tracev2::emit_instant("causal.window_publish", _wb, '"'"'g'"'"'); \}|;
    ' "$GCF"
    if grep -q 'causal.window_publish' "$GCF"; then
        echo "PATCHED causal.window_publish (L_resolve): $GCF"
    else
        echo "WARN: causal.window_publish patch did not match" >&2
    fi
fi

# ---- L_resolve (INDEPENDENT) : B/E SPAN around the consumer emplace block ----
# The causal.window_publish INSTANT above gives ordering/frontier/tail but has
# NO duration, so the model could only derive L_resolve as the inter-publish
# GAP == publish_span/N — the telescoping TAUTOLOGY (wall_pred == wall, +0.0%
# residual). To make L_resolve an INDEPENDENT parameter we measure the actual
# SERIAL resolve WORK the in-order consumer spends: open a ScopedSpan covering
# the `if ( !m_windowMap->get( windowOffset ) ) { ... }` block — vendor's own
# comment (~559-561) calls this "the critical path that cannot be parallelized".
# The span DURATION is the per-link serial cost (getLastWindow + emplace), which
# is what Fulcrum's `model` reads as L_resolve. It is INDEPENDENT of where the
# publishes land in time, so wall_pred becomes a real prediction with a NONZERO
# residual. We insert the guard right after the `if (...) {` and rely on the
# block's existing closing brace to end the span.
if ! grep -q 'consumer.window_publish' "$GCF"; then
    perl -i -0777 -pe '
        s|(\n        if \( !m_windowMap->get\( windowOffset \) \) \{\n)|$1            ::tracev2::ScopedSpan _tv2_pub("consumer.window_publish",\n                ([\&]{ static thread_local char _pb[160];\n                  std::snprintf(_pb, sizeof(_pb),\n                    "\\"start_bit\\":%zu,\\"end_bit\\":%zu,\\"site\\":\\"consumer\\",\\"had_markers\\":%s",\n                    chunkData->encodedOffsetInBits, windowOffset,\n                    chunkData->containsMarkers() ? "true" : "false");\n                  return _pb; }()));\n|;
    ' "$GCF"
    if grep -q 'consumer.window_publish' "$GCF"; then
        echo "PATCHED consumer.window_publish SPAN (independent L_resolve): $GCF"
    else
        echo "WARN: consumer.window_publish span patch did not match" >&2
    fi
fi

# ---- worker.decode_mode INSTANT : authoritative clean/window-absent split ----
# gzippy emits a `worker.decode_mode` instant keyed by start_bit so the model
# splits d_c/d_w by the ACTUAL window-present-at-decode-start predicate, not the
# dispatch intent. rapidgzip's worker.decode span already carries the same
# `mode` arg (sharedWindow ? clean : window_absent), so no separate instant is
# needed there — the model reads mode straight off the span. (Documented so the
# two tools' d_c/d_w are known to be apples-to-apples.)

echo "Done. (worker.decode => d_c/d_w ; causal.window_publish INSTANT => ordering/frontier/tail ; consumer.window_publish SPAN => INDEPENDENT L_resolve)"
