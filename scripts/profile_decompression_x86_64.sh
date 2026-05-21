#!/usr/bin/env bash
# Deprecated alias — use profile_single_member_decompression_x86_64.sh
echo "NOTE: profile-decompression-x86_64 renamed to profile-single-member-decompression-x86_64" >&2
exec "$(dirname "$0")/profile_single_member_decompression_x86_64.sh" "$@"
