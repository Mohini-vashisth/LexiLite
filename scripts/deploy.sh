#!/usr/bin/env bash
# Run from the Mac: deploys a branch (default main) to the home PC over SSH.
# Requires Tailscale signed into the personal account (see README).
#   ./scripts/deploy.sh            # deploy main
#   ./scripts/deploy.sh my-branch  # try a branch on the PC before merging
set -euo pipefail
BRANCH="${1:-main}"
ssh -o ControlMaster=no -o ControlPath=none homepc \
  "powershell -NoProfile -ExecutionPolicy Bypass -File C:\\Users\\lm110\\OneDrive\\Desktop\\Projects\\LexiLite\\scripts\\deploy.ps1 -Branch $BRANCH"
