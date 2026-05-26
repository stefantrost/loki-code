#!/usr/bin/env bash
# Coupling snapshot for loki-code.
#
# Prints afferent (Ca), efferent (Ce), and instability (I = Ce/(Ca+Ce))
# for each internal package. Writes a human table to stdout and, when run
# under GitHub Actions, appends the same table to the workflow summary.
#
# This is a *snapshot*, not a gate — drift is the signal. Run it in CI on
# every push so the trend over time is visible.

set -euo pipefail

MODULE="$(go list -m)"

# All internal packages (the module itself and any subpackages).
# Avoid `mapfile` so this stays compatible with the macOS bash 3.2 default.
INTERNAL=()
while IFS= read -r line; do
  INTERNAL+=("$line")
done < <(go list "${MODULE}/..." 2>/dev/null)

print_row() {
  printf "%-40s %4s %4s %8s\n" "$1" "$2" "$3" "$4"
}

emit() {
  if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    printf "%s\n" "$1" >> "${GITHUB_STEP_SUMMARY}"
  fi
  printf "%s\n" "$1"
}

emit "## Coupling snapshot"
emit ''
emit '```'
emit "$(print_row PACKAGE Ca Ce I)"
emit "$(print_row '----------------------------------------' '----' '----' '--------')"

for pkg in "${INTERNAL[@]}"; do
  # Ce: number of *internal* packages this one imports directly.
  # `|| true` because grep returns 1 when there are no matches; with set -e
  # that would abort the whole script.
  ce=$(go list -f '{{ range .Imports }}{{ . }}{{ "\n" }}{{ end }}' "$pkg" \
        | { grep -E "^${MODULE}(/|$)" || true; } \
        | sort -u | wc -l | tr -d ' ')

  # Ca: number of *internal* packages that import this one directly.
  ca=0
  for other in "${INTERNAL[@]}"; do
    [[ "$other" == "$pkg" ]] && continue
    if go list -f '{{ range .Imports }}{{ . }}{{ "\n" }}{{ end }}' "$other" \
        2>/dev/null | grep -qx "$pkg"; then
      ca=$((ca + 1))
    fi
  done

  total=$((ca + ce))
  if (( total == 0 )); then
    instability="n/a"
  else
    instability=$(awk "BEGIN { printf \"%.2f\", $ce / $total }")
  fi

  emit "$(print_row "$pkg" "$ca" "$ce" "$instability")"
done

emit '```'
emit ''
emit 'Read: low I = stable (safe to depend on). High I = volatile (safe to depend *from*).'
emit 'Healthy direction is `main → clients`, so `clients` should sit near I=0 and `main` near I=1.'
