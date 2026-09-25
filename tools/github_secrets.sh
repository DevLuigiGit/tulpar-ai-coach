#!/usr/bin/env bash
# Copy the LLM / RAG keys from your local .env to GitHub Actions secrets of this repo, so CI also runs the
# router and RAG evals (without them CI runs only the key-free steps). Needs the gh CLI logged in.
# Run it yourself:  bash tools/github_secrets.sh [owner/repo]
# Values go to gh through stdin: never on the command line (ps), never printed.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] || { echo "no .env"; exit 1; }
command -v gh > /dev/null || { echo "gh CLI not found: https://cli.github.com"; exit 1; }
repo=()
[ -n "${1:-}" ] && repo=(--repo "$1")
# .env is read as data, never executed: KEY=value per line, quotes and spaces around the value dropped
get() { grep -E "^$1[[:space:]]*=" .env | head -1 | cut -d= -f2- | sed -E "s/^[[:space:]\"']+//; s/[[:space:]\"']+$//"; }
n=0
for k in GROQ_API_KEY OLLAMA_API_KEY JINA_API_KEY; do
  v="$(get $k || true)"  # a missing line is not an error under set -e -o pipefail
  if [ -z "$v" ]; then echo "skip $k: empty in .env"; continue; fi
  printf '%s' "$v" | gh secret set "$k" ${repo[@]+"${repo[@]}"} > /dev/null
  echo "set $k"
  n=$((n + 1))
done
[ "$n" -eq 0 ] && { echo "no keys filled in .env"; exit 1; }
echo "done: $n secrets set on GitHub (values not printed)"
