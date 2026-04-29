#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${ROOT}/data"
TARGET="${DATA}/ml-100k"
mkdir -p "$DATA"
cd "$DATA"
if [[ ! -f ml-100k.zip ]]; then
  curl -L -o ml-100k.zip "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
fi
unzip -o ml-100k.zip
echo "Ready: ${TARGET}"
