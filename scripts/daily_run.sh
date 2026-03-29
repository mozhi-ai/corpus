#!/usr/bin/env bash
# Daily corpus collection, processing, and publishing.
# Usage: ./scripts/daily_run.sh [--limit N]
#
# Requires HF_TOKEN environment variable to be set.

set -euo pipefail

LIMIT="${1:-}"

echo "=== mozhi-ai daily corpus run ==="
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set"
    exit 1
fi

LIMIT_FLAG=""
if [ -n "$LIMIT" ]; then
    LIMIT_FLAG="--limit $LIMIT"
fi

echo ""
echo "--- Collecting from HuggingFace ---"
uv run mozhi collect --source huggingface $LIMIT_FLAG

echo ""
echo "--- Collecting from Wikipedia ---"
uv run mozhi collect --source wikipedia $LIMIT_FLAG

echo ""
echo "--- Collecting from Project Madurai ---"
uv run mozhi collect --source madurai $LIMIT_FLAG

echo ""
echo "--- Processing pipeline ---"
uv run mozhi process

echo ""
echo "--- Publishing to HuggingFace ---"
uv run mozhi publish --repo mozhi-ai/tamil-corpus

echo ""
echo "=== Done ==="
