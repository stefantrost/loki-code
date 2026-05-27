#!/usr/bin/env bash
# Per-file coverage floor check.
#
# Reads cover.out (produced by `go test -coverprofile=cover.out`) and compares
# each file's statement-weighted coverage against the baseline in
# coverage-floors.txt. Fails if any file drops more than $SLACK_PP percentage
# points below its floor.
#
# Files not listed in coverage-floors.txt are ignored (new files require a
# deliberate baseline addition; running this script with --update rewrites the
# floors at the current numbers).
#
# Usage:
#   scripts/coverage-floor.sh              # check
#   scripts/coverage-floor.sh --update     # rewrite floors at current numbers

set -euo pipefail

COVER="${COVER:-cover.out}"
FLOORS="${FLOORS:-coverage-floors.txt}"
SLACK_PP="${SLACK_PP:-2}"

if [[ ! -f "$COVER" ]]; then
  echo "coverage-floor: $COVER not found — run 'go test -coverprofile=$COVER ./...' first" >&2
  exit 2
fi

# Compute current statement-weighted coverage per file.
current=$(awk 'NR==1{next} {
  n=split($1,a,":"); file=a[1];
  total[file]+=$2;
  if ($3+0 > 0) covered[file]+=$2;
}
END {
  for (f in total) {
    pct = (total[f]>0) ? 100*covered[f]/total[f] : 0;
    printf "%s %.1f\n", f, pct;
  }
}' "$COVER" | sort)

if [[ "${1:-}" == "--update" ]]; then
  {
    echo "# Per-file coverage floors. Regenerate with scripts/coverage-floor.sh --update."
    echo "# Each line: <file> <floor-percentage>"
    echo "$current"
  } > "$FLOORS"
  echo "Wrote $FLOORS with $(echo "$current" | wc -l | tr -d ' ') entries."
  exit 0
fi

if [[ ! -f "$FLOORS" ]]; then
  echo "coverage-floor: $FLOORS not found — generate with --update first" >&2
  exit 2
fi

fail=0
declare -a violations
while read -r file pct; do
  [[ -z "$file" || "$file" == \#* ]] && continue
  floor=$(awk -v f="$file" '$1==f { print $2; exit }' "$FLOORS")
  if [[ -z "$floor" ]]; then
    continue
  fi
  if awk -v c="$pct" -v f="$floor" -v s="$SLACK_PP" 'BEGIN { exit !(c < f - s) }'; then
    violations+=("$file: $pct% < floor $floor% (slack ${SLACK_PP}pp)")
    fail=1
  fi
done <<<"$current"

if (( fail )); then
  echo "Coverage floor violations:" >&2
  for v in "${violations[@]}"; do
    echo "  $v" >&2
  done
  echo "" >&2
  echo "If the drop is intentional, run: scripts/coverage-floor.sh --update" >&2
  exit 1
fi

stale=0
declare -a stale_entries
while read -r file _; do
  [[ -z "$file" || "$file" == \#* ]] && continue
  if ! grep -qF "$file" <(echo "$current"); then
    stale_entries+=("$file")
    stale=1
  fi
done < "$FLOORS"

if (( stale )); then
  echo "Stale entries in $FLOORS (files not in cover.out — run 'make cover-update'):" >&2
  for e in "${stale_entries[@]}"; do
    echo "  $e" >&2
  done
  exit 1
fi

echo "Coverage floors OK."
