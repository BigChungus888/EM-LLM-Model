#!/usr/bin/env bash

source .venv/bin/activate

./train.sh \
  --job.config_file flame/models/fla.toml \
  --job.dump_folder experiments/titans-mag-linear-340M \
  --model.config configs/titans_mag_linear_sgd_340M.json \
  --model.tokenizer_path fla-hub/delta_net-1.3B-100B
