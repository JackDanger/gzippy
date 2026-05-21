#!/bin/bash
# Filter a parallel-sm JSON-lines log by event kind, partition, or thread.
#
# Usage:
#   scripts/parallel_sm_log_grep.sh LOG ev=speculative_mismatch
#   scripts/parallel_sm_log_grep.sh LOG partition_idx=5
#   scripts/parallel_sm_log_grep.sh LOG thread=consumer
#   scripts/parallel_sm_log_grep.sh LOG ev=consume_done partition_idx=5
#
# Multiple filters AND together. Output is each matching line as-is.

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 LOG_FILE [key=value ...]" >&2
  exit 2
fi

LOG="$1"
shift

if [ ! -f "$LOG" ]; then
  echo "no such log: $LOG" >&2
  exit 2
fi

cmd='cat'
for filter in "$@"; do
  key="${filter%%=*}"
  val="${filter#*=}"
  case "$key" in
    ev|thread)
      cmd="$cmd | grep -F '\"$key\":\"$val\"'"
      ;;
    partition_idx|start_bit|end_bit|seed_bit|found_bit|expected_start|speculative_start|until_bit)
      cmd="$cmd | grep -E '\"$key\":(${val})([,}])'"
      ;;
    *)
      echo "unknown filter key: $key" >&2
      exit 2
      ;;
  esac
done

eval "$cmd '$LOG'"
