#!/usr/bin/env bash
# Fail if gzippy pins a submodule SHA that is not on its GitHub remote.
# Catches the common mistake: bump vendor/isa-l or vendor/isal-rs locally
# without pushing JackDanger/isa-l or JackDanger/isal-rs first.
# Also fails when a guarded vendor path is absent from HEAD entirely, when its
# .gitmodules entry (name or url) is missing, or when ls-remote cannot reach
# the remote — an unverifiable pin is not a cleared pin.
set -euo pipefail

root=$(git rev-parse --show-toplevel)
cd "$root"

submodule_name_for_path() {
    local path=$1
    # A missing .gitmodules must not hard-exit under set -e; an empty name is
    # reported as a distinct failure by check_path below.
    git config -f .gitmodules --get-regexp '^submodule\..*\.path$' 2>/dev/null |
        awk -v p="$path" '$2 == p { sub(/^submodule\./, "", $1); sub(/\.path$/, "", $1); print $1; exit }' ||
        true
}

# Returns 0 if the remote lists $sha, 1 if it does not, and 2 if ls-remote
# itself failed (network/auth — the pin could not be verified either way).
remote_has_commit() {
    local url=$1
    local sha=$2
    local remote_refs
    # `git ls-remote URL SHA` treats SHA as a ref name, not a commit id.
    remote_refs=$(git ls-remote "$url") || return 2
    printf '%s\n' "$remote_refs" | awk -v want="$sha" '$1 == want { found=1 } END { exit !found }'
}

check_path() {
    local path=$1
    local sha submodule_name url

    sha=$(git ls-tree HEAD -- "$path" 2>/dev/null | awk '{print $3}')
    if [[ -z "$sha" ]]; then
        echo "ERROR: $path has no entry in HEAD at all — an absent vendor tree is a" >&2
        echo "failure for this guard: vendor/isa-l and vendor/isal-rs must stay pinned." >&2
        echo "Recovery: \`git log --oneline -- $path\` names the commit that removed it." >&2
        echo "Revert that commit, or \`git checkout <last-good> -- $path\`. Note that" >&2
        echo "\`git submodule update --init\` is no remedy here — it no-ops because" >&2
        echo "HEAD carries no gitlink to initialize." >&2
        return 1
    fi

    submodule_name=$(submodule_name_for_path "$path")
    url=""
    if [ -n "$submodule_name" ]; then
        url=$(git config -f .gitmodules --get "submodule.${submodule_name}.url" 2>/dev/null || true)
    fi
    if [ -z "$submodule_name" ] || [ -z "$url" ]; then
        echo "ERROR: $path is pinned in HEAD but missing from .gitmodules (no matching" >&2
        echo "path entry, or no url) — structure, not network: this is not something" >&2
        echo "ls-remote can clear. Restore the .gitmodules entry for $path." >&2
        return 1
    fi

    local rc
    remote_has_commit "$url" "$sha" && rc=0 || rc=$?

    if [ "$rc" -eq 2 ]; then
        echo "ERROR: git ls-remote failed for $url (network or auth), so the $path" >&2
        echo "pin $sha is UNVERIFIED; check it by hand before pushing gzippy." >&2
        return 1
    fi
    if [ "$rc" -ne 0 ]; then
        echo "ERROR: $path pins $sha but that commit is not on $url" >&2
        echo "Push the fork branch before pushing gzippy (e.g. cd $path && git push origin)." >&2
        return 1
    fi
    return 0
}

failed=0
for path in vendor/isa-l vendor/isal-rs; do
    check_path "$path" || failed=1
done

exit "$failed"
