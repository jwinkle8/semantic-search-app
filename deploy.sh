#!/usr/bin/env bash
# Deploy the app to its Hugging Face Space (see "Deploying" in README.md).
#
#   .venv/bin/hf auth login   # once, with a token that has write access
#   ./deploy.sh
#
# Uses `hf upload` rather than `git push`: it sends only the files listed below and stores the
# 16 MB chunk file in Hugging Face's Xet storage automatically, so Git LFS isn't needed and the
# GitHub history (which contains files over Hugging Face's 10 MB git limit) never goes to the Hub.
set -euo pipefail

SPACE_NAME="nps-semantic-search"
FILES=(app.py build_index.py requirements.txt README.md data/nps_chunks.json)

cd "$(dirname "$0")"
HF=.venv/bin/hf
PY=.venv/bin/python

# Only deploy committed code, so every Space version maps to exactly one GitHub commit.
if [[ -n "$(git status --porcelain -- "${FILES[@]}")" ]]; then
  echo "Uncommitted changes in deployed files; commit them first:" >&2
  git status --short -- "${FILES[@]}" >&2
  exit 1
fi

HF_USER="$("$PY" -c 'from huggingface_hub import whoami; print(whoami()["name"])')" || {
  echo "Not logged in to Hugging Face. Run: $HF auth login" >&2
  exit 1
}
SPACE_ID="$HF_USER/$SPACE_NAME"
COMMIT="$(git rev-parse --short HEAD)"

"$HF" repos create "$SPACE_ID" --type space --sdk gradio --no-private --exist-ok

INCLUDES=()
for f in "${FILES[@]}"; do INCLUDES+=(--include "$f"); done
"$HF" upload "$SPACE_ID" . . --repo-type space "${INCLUDES[@]}" \
  --commit-message "Deploy $COMMIT: $(git log -1 --format=%s)" \
  --commit-description "GitHub commit $(git rev-parse HEAD)"

echo
echo "Deployed $COMMIT to https://huggingface.co/spaces/$SPACE_ID"
echo "Follow the build with: $HF spaces logs $SPACE_ID"
