#!/usr/bin/env bash
# Copy the API keys from your local .env to the Railway service `coach` of project tulpar-ai-coach.
# Run it yourself after filling .env:  bash tools/railway_keys.sh [--bot]
# The Telegram token is copied only with --bot: a bot token must be polled by ONE process (Railway OR local).
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] || { echo "no .env"; exit 1; }
# .env is read as data, never executed: KEY=value per line, quotes and spaces around the value dropped
get() { grep -E "^$1=" .env | head -1 | cut -d= -f2- | sed -E "s/^[[:space:]\"']+//; s/[[:space:]\"']+$//"; }
args=()
for k in OLLAMA_API_KEY GROQ_API_KEY JINA_API_KEY LANGSMITH_API_KEY; do
  v="$(get $k)"; [ -n "$v" ] && args+=(--set "$k=$v")
done
if [ "${1:-}" = "--bot" ]; then
  v="$(get TELEGRAM_BOT_TOKEN)"; [ -n "$v" ] && args+=(--set "TELEGRAM_BOT_TOKEN=$v")
  v="$(get TRAINER_TELEGRAM_IDS | tr -d '[:space:]')"; [ -n "$v" ] && args+=(--set "TRAINER_TELEGRAM_IDS=$v")
fi
[ ${#args[@]} -eq 0 ] && { echo "no keys filled in .env"; exit 1; }
railway variables --service coach "${args[@]}" > /dev/null
echo "done: $(( ${#args[@]} / 2 )) variables set on Railway (values not printed)"
