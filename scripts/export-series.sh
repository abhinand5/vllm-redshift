#!/usr/bin/env bash
# Export a model's patch series from a dev branch in a full vLLM clone, then prove it rebuilds that branch.
# usage: scripts/export-series.sh <model> <vllm-clone> <branch>
# Reads models/<model>/series.conf; writes models/<model>/patches/**; sets PATCHES in model.env.
set -euo pipefail
cd "$(dirname "$0")/.."; M=$1; C=$2; BR=$3; D=models/$M; . $D/model.env
EXCL=(':!.claude' ':!.buildkite')
conf=$(grep -vE '^\s*(#|$)' $D/series.conf)
OURS_BASE=$(sed -n 's/^OURS_BASE=//p' <<<"$conf"); [ -n "$OURS_BASE" ] || { echo "series.conf: no OURS_BASE"; exit 1; }
rev() { case $1 in BRANCH) echo "$BR";; OURS_BASE) echo "$OURS_BASE";; *) echo "$1";; esac; }

rm -rf $D/patches; mkdir -p $D/patches; list=(); ours_paths=()
while IFS='|' read -r name from to paths; do
  name=$(xargs <<<"$name"); from=$(rev $(xargs <<<"$from")); to=$(rev $(xargs <<<"$to")); read -ra ps <<<"$paths"
  [ "$from" = "$OURS_BASE" ] && ours_paths+=("${ps[@]}")
  out=$D/patches/$name; mkdir -p "$(dirname "$out")"
  {
    echo "redshift patch: $name"
    echo "cut from: $(git -C "$C" rev-parse --short "$from")..$(git -C "$C" rev-parse --short "$to") -- ${ps[*]}"
    echo "commits:"
    git -C "$C" log --no-merges --reverse --format='  %h %an: %s' "$from..$to" -- "${ps[@]}" "${EXCL[@]}"
    echo
    git -C "$C" diff --binary "$from" "$to" -- "${ps[@]}" "${EXCL[@]}"
  } > "$out"
  grep -q '^diff --git' "$out" || { echo "EMPTY patch $name ($from..$to -- ${ps[*]})"; exit 1; }
  list+=("$name"); echo "wrote $name ($(grep -c '^diff --git' "$out") files)"
done < <(grep '|' <<<"$conf")

# Coverage: every path we changed after OURS_BASE must be in one of our patches.
missing=$(git -C "$C" diff --name-only "$OURS_BASE" "$BR" -- . "${EXCL[@]}" | while read -r f; do
  hit=0; for p in "${ours_paths[@]}"; do [[ $p == . || $f == "$p" || $f == "$p"/* ]] && hit=1; done
  [ $hit = 1 ] || echo "$f"; done)
[ -z "$missing" ] || { echo "NOT COVERED by series.conf:"; echo "$missing"; exit 1; }

# Rebuild check: VLLM_COMMIT + series == branch tree (minus .claude/.buildkite).
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT; mkdir -p "$T/a" "$T/b"
git -C "$C" archive "$VLLM_COMMIT" | tar -x -C "$T/a"; git -C "$T/a" init -q
for p in "${list[@]}"; do git -C "$T/a" apply --whitespace=nowarn "$PWD/$D/patches/$p"; done
rm -rf "$T/a/.git"; git -C "$C" archive "$BR" | tar -x -C "$T/b"
for x in a b; do rm -rf "$T/$x/.claude" "$T/$x/.buildkite"; done
if diff -rq "$T/a" "$T/b" > "$T/diff.txt"; then echo "rebuild check OK: $VLLM_COMMIT + ${#list[@]} patches == $BR ($(git -C "$C" rev-parse --short "$BR"))"
else echo "rebuild check FAILED:"; head -20 "$T/diff.txt"; exit 1; fi

sed -i "s|^PATCHES=.*|PATCHES=\"${list[*]}\"   # applied in order from patches/ (see series.conf)|" $D/model.env
