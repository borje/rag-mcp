#!/usr/bin/env bash
# Run prepare-transfer.sh inside a Linux container to produce Linux-compatible wheels.
# Use this instead of running prepare-transfer.sh directly on macOS.
set -euo pipefail

docker run --rm -v "$PWD":/work -w /work python:3.13-slim-bookworm \
  bash -c "apt-get update -qq && apt-get install -y -q curl && bash prepare-transfer.sh"
