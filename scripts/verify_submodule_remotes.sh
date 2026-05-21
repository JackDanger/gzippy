#!/usr/bin/env bash
# Fail if gzippy pins a submodule SHA that is not on its GitHub remote.
# Catches the common mistake: bump vendor/isa-l or vendor/isal-rs locally
# without pushing JackDanger/isa-l or JackDanger/isal-rs first.
set -euo pipefail

root=$(git rev-parse --show-toplevel)
cd "$root"

submodule_name_for_path() {
    local path=$1
    git config -f .gitmodules --get-regexp '^submodule\..*\.path$' |
        awk -v p="$path" '$2 == p { sub(/^submodule\./, "", $1); sub(/\.path$/, "", $1); print $1; exit }'
}

check_path() {
    local path=$1
    local sha submodule_name url

    sha=$(git ls-tree HEAD "$path" 2>/dev/null | awk '{print $3}')
    [[ -n "$sha" ]] || return 0

    submodule_name=$(submodule_name_for_path "$path")
    url=$(git config -f .gitmodules --get "submodule.${submodule_name}.url")

    if git ls-remote "$url" "$sha" | grep -q .; then
        return 0
    fi

    echo "ERROR: $path pins $sha but that commit is not on $url" >&2
    echo "Push the fork branch before pushing gzippy (e.g. cd $path && git push origin)." >&2
    return 1
}

failed=0
for path in vendor/isa-l vendor/isal-rs; do
    check_path "$path" || failed=1
done

exit "$failed"
