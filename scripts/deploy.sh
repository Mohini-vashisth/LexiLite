#!/usr/bin/env bash
# Run from the Mac: deploys origin/main to the home PC over SSH.
# Requires Tailscale signed into the personal account (see README).
set -euo pipefail
ssh -o ControlMaster=no -o ControlPath=none homepc \
  'powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\lm110\OneDrive\Desktop\Projects\LexiLite\scripts\deploy.ps1'
