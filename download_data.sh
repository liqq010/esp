#!/usr/bin/env bash
# TinyStories validation split; the experiments use only its first ~0.5 MB.
set -euo pipefail
mkdir -p data
curl -L -o data/TinyStoriesV2-GPT4-valid.txt \
  https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
