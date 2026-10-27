# Full option: LoOPSD route-search workspace

This repository contains the source trees from the local `project/code` folder:

- `loopsd-route-search/`: LoOPSD route search implementation, configs, and tests.
- `nemo-rl/`: NeMo RL checkout used by the route-search adapter.
- `CRISP_Reasoning_Compression/`: CRISP reference checkout.

Each checkout is copied as a source snapshot, without its nested `.git` metadata.
The Python virtual environment, generated outputs/caches, model checkpoints, and
datasets are not included.

Source revisions at snapshot time:

- CRISP: `f99b4098bc9bbed923e96aac389d17aa69bd1f24`
- NeMo RL: `bb4f0d85674344d01656efdbf8eb2a7f3f737ca3`
- LoOPSD route search: `d6546afd4bd9978fda2c5ab36a5c3e2b4396175b`

For the offline B200 route search, keep the Qwen3 checkpoint in the sibling
`project/models/Qwen3-1.7B` directory and GSM8K parquet files in
`project/data/gsm8k`. The runnable config is
`loopsd-route-search/configs/qwen3_1p7b_gsm8k_full_b200.yaml`; see the route
search project's `OFFLINE_B200.md` for environment setup and commands.
