#!/usr/bin/env bash
# Copy the API keys from your local .env to the Railway service `coach` of project tulpar-ai-coach.
# Run it yourself from the repo root after filling .env:  bash tools/railway_keys.sh
# The Telegram token is copied only with --bot: a bot token must be polled by ONE process (Railway OR local).
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] || { echo "no .env"; exit 1; }
set -a; . ./.env; set +a
args=()
for k in OLLAMA_API_KEY GROQ_API_KEY JINA_API_KEY LANGSMITH_API_KEY; do
  v="${!k:-}"; [ -n "$v" ] && args+=(--set "$k=$v")
done
if [ "${1:-}" = "--bot" ]; then
  [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && args+=(--set "TELEGRAM_BOT_TOKEN=$TELEGRAM_BOT_TOKEN")
  [ -n "${TRAINER_TELEGRAM_IDS:-}" ] && args+=(--set "TRAINER_TELEGRAM_IDS=$TRAINER_TELEGRAM_IDS")
fi
[ ${#args[@]} -eq 0 ] && { echo "no keys filled in .env"; exit 1; }
railway variables --service coach "${args[@]}"
echo "done: $(( ${#args[@]} / 2 )) variables set on Railway (values not printed)"
