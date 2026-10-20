#!/usr/bin/env bash
# Rival-version drift check for tests/fingerprints/rivals.tsv (issue #323).
#
# rivals.tsv pins the rival CLI versions every fixture/board/ledger comparison
# against rivals was taken at. Pins are written by
#   cargo run --release --example fingerprint_tool -- pin-rivals
# which records the FIRST LINE of the tool's own version output — `--version`
# for every rival except libdeflate-gzip, whose version flag is `-V` — as a
# provenance line `#   <name>: <banner>` next to a `# covered: <list>` line.
# A pin is only comparable at the pinned version: #323's phantom-loss root
# cause was a box silently running libdeflate 1.25 against the pinned v1.23
# baseline and scoring tabular L2 +10.3% — drift the ratchet never saw.
#
# This script re-probes each rival listed in `# covered:` with the SAME
# invocation the pin prober used and prints a drift table:
#
#     rival | pinned | installed | MATCH / DRIFT / MISSING
#
# Exit polarity (a check's job is to surface drift loudly, not to skip):
#   0  every covered rival matches, and any absent rival was allowed via
#      --missing-ok
#   1  DRIFT: an installed rival's version differs from its pin, a version
#      probe produced no banner, or a rival is absent and --missing-ok was
#      NOT given
#   2  setup error: bad arguments, a missing/unparseable rivals.tsv, or a
#      covered rival with no pin recorded in the provenance header
#
# Per-rival binary overrides for vendor-built rivals in CI (otherwise the
# CLI is resolved on PATH): GZIP_BIN, PIGZ_BIN, LIBDEFLATE_BIN, IGZIP_BIN.
#
# Usage: bash scripts/rival_version_check.sh [--missing-ok] [--rivals <tsv>]
set -euo pipefail

usage_error() {
    echo "ERROR: $*" >&2
    echo "       usage: bash scripts/rival_version_check.sh [--missing-ok] [--rivals <tsv>]" >&2
    exit 2
}

trim() { sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'; }

missing_ok=0
tsv=""
while [ $# -gt 0 ]; do
    case $1 in
        --missing-ok) missing_ok=1 ;;
        --rivals)
            [ $# -ge 2 ] || usage_error "--rivals requires a path"
            tsv=$2
            shift ;;
        --rivals=*) tsv=${1#--rivals=} ;;
        *) usage_error "unexpected argument '$1'" ;;
    esac
    shift
done

# Resolve the default pin file from the repo root (same pattern as
# scripts/verify_submodule_remotes.sh).
if [ -z "$tsv" ]; then
    root=$(git rev-parse --show-toplevel 2>/dev/null) ||
        usage_error "not inside the gzippy repo and no --rivals given"
    tsv="$root/tests/fingerprints/rivals.tsv"
fi
if [ ! -f "$tsv" ]; then
    echo "ERROR: rivals pin file not found: $tsv" >&2
    echo "       regenerate with: cargo run --release --example fingerprint_tool -- pin-rivals" >&2
    exit 2
fi

# The `# covered:` line names the rivals actually pinned by this file (a pin
# run on a box lacking a rival records no rows and no pin for it).
covered=$(grep -m1 -E '^#[[:space:]]*covered:' "$tsv") ||
    { echo "ERROR: $tsv has no '# covered:' provenance line" >&2; exit 2; }
covered=${covered#*:}
IFS=',' read -r -a covered_arr <<<"$covered"
if [ "${#covered_arr[@]}" -eq 0 ]; then
    echo "ERROR: $tsv's '# covered:' line lists no rivals" >&2
    exit 2
fi

# rival-name -> "CLI:version-flag". The REAL prober invocations, kept in
# lock-step with examples/fingerprint_tool.rs (rival_commands / pin-rivals):
# gzip and pigz answer `--version`; libdeflate-gzip has no --version and
# answers `-V`.
probespec() {
    case $1 in
        gzip)       echo 'gzip:--version' ;;
        pigz)       echo 'pigz:--version' ;;
        libdeflate) echo 'libdeflate-gzip:-V' ;;
        igzip)      echo 'igzip:--version' ;;
        *) return 1 ;;
    esac
}

# The pin line fingerprint_tool wrote for a covered rival:
# `#   <name>: <first version banner line>` (colon-space separated).
pin_for() {
    local name=$1 line
    line=$(grep -m1 -E "^#[[:space:]]+${name}:[[:space:]]" "$tsv") || return 1
    printf '%s' "${line#*: }"
}

# Print the probe's version banner: first line of stdout, falling back to the
# first line of stderr when stdout is empty (exactly the pin prober's rule),
# independent of the tool's exit status.
probe_version() {
    local bin=$1 flag=$2 out=
    out=$("$bin" "$flag" 2>/dev/null || true)
    if [ -z "$out" ]; then
        out=$("$bin" "$flag" 2>&1 >/dev/null || true)
    fi
    printf '%s\n' "$out" | head -n 1 | tr -d '\r' | trim
}

# Resolve the binary to probe: $GZIP_BIN-style override first (the binary the
# measurement itself used), else the CLI on PATH; return 2 for an override
# that names a non-runnable path, 1 for plain absence.
resolve_bin() {
    local envkey=$1 cli=$2 override
    override=${!envkey:-}
    if [ -n "$override" ]; then
        [ -x "$override" ] || return 2
        printf '%s' "$override"
    else
        [ -n "$(command -v -- "$cli" 2>/dev/null)" ] || return 1
        command -v -- "$cli" 2>/dev/null
    fi
}

names=() pins=() installed=() verdicts=()
w_riv=5 w_pin=6 w_ins=9
drift=0
match_n=0
tolerated_n=0

for name in "${covered_arr[@]}"; do
    name=${name//[[:space:]]/}
    pin=$(pin_for "$name") || {
        echo "ERROR: covered rival '$name' has no '#   $name: <banner>' pin line in $tsv" >&2
        echo "       the provenance header is incomplete — regenerate rivals.tsv." >&2
        exit 2
    }
    if [ -z "$pin" ]; then
        echo "ERROR: covered rival '$name' has an empty pin in $tsv" >&2
        exit 2
    fi

    envkey=$(printf '%s_BIN' "$name" | tr '[:lower:]-' '[:upper:]_')
    IFS=':' read -r cli flag <<<"$(probespec "$name")"
    if [ -z "$cli" ]; then
        echo "ERROR: $tsv covers rival '$name' but no probe is defined for it here" >&2
        exit 2
    fi

    verdict=""
    inst=""
    bin_path=""
    if bin_path=$(resolve_bin "$envkey" "$cli" 2>/dev/null); then
        inst=$(probe_version "$bin_path" "$flag")
        if [ "$inst" = "$pin" ]; then
            verdict="MATCH"
            match_n=$((match_n + 1))
        elif [ -z "$inst" ]; then
            inst="(empty $flag output)"
            verdict="DRIFT"
            drift=1
        else
            verdict="DRIFT"
            drift=1
        fi
    else
        rc=$?
        if [ "$rc" -eq 2 ]; then
            inst="ABSENT ($envkey set but not a runnable binary: ${!envkey})"
        else
            inst="ABSENT (no $cli on PATH; no $envkey)"
        fi
        if [ "$missing_ok" -eq 1 ]; then
            verdict="MISSING (allowed by --missing-ok)"
            tolerated_n=$((tolerated_n + 1))
        else
            verdict="DRIFT (absent)"
            drift=1
        fi
    fi
    names+=("$name") pins+=("$pin") installed+=("$inst") verdicts+=("$verdict")
    if [ ${#name} -gt "$w_riv" ]; then w_riv=${#name}; fi
    if [ ${#pin} -gt "$w_pin" ]; then w_pin=${#pin}; fi
    if [ ${#inst} -gt "$w_ins" ]; then w_ins=${#inst}; fi
done

echo "Rival versions vs $(realpath -- "$tsv"):"
echo
printf '%-*s  %-*s  %-*s  %s\n' "$w_riv" rival "$w_pin" "pinned (rivals.tsv)" \
    "$w_ins" installed "verdict"
for i in "${!names[@]}"; do
    printf '%-*s  %-*s  %-*s  %s\n' "$w_riv" "${names[$i]}" "$w_pin" "${pins[$i]}" \
        "$w_ins" "${installed[$i]}" "${verdicts[$i]}"
done
echo

if [ "$drift" -eq 1 ]; then
    echo "FAIL: rival-version drift vs $tsv (issue #323: a pin is only comparable at its"
    echo "pinned version — drifted rivals silently invalidate every rivals.tsv-based"
    echo "comparison; repin with"
    echo "  cargo run --release --example fingerprint_tool -- pin-rivals"
    echo "on a box measured at the new versions, in the PR that changes them)."
    exit 1
fi
echo "OK: $match_n rival(s) match their pins."
if [ "$tolerated_n" -gt 0 ]; then
    echo "    $tolerated_n absent rival(s) tolerated via --missing-ok (the nightly"
    echo "    does not build them; absence stays visible in the table above)."
fi
exit 0
