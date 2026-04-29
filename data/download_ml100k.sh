#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET="${DIR}/ml-100k"
mkdir -p "$DIR"
cd "$DIR"
if [[ ! -f ml-100k.zip ]]; then
  curl -L -o ml-100k.zip "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
fi
unzip -o ml-100k.zip
echo "Ready: ${TARGET}"
