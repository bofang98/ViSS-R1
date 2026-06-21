# ViSS-R1

`ViSS-R1` is a cleaned public code release built on top of the `Video-R1` codebase, with project-specific preprocessing, training, and evaluation scripts organized for this repository.

## Overview

This repository mainly contains:

- `src/r1-v/`: the main training codebase
- `src/scripts/`: training launch scripts
- `src/eval_bench.sh`, `src/eval_bench_parallel.sh`, `src/eval_bench_ssl_parallel.sh`: evaluation entrypoints
- `preprocess/` and `preprocess_SFT/`: preprocessing and annotation utilities
- `src/inference_example.py`: simple inference example

## Setup

We borrow from the [`Video-R1`](https://github.com/tulerfeng/Video-R1) codebase and recommend using the same environment convention as `Video-R1`.

That means:

- reuse the `Video-R1` software stack when possible
- keep dependency versions aligned with `Video-R1`, especially for `transformers`, `vllm`, `trl`, `flash-attn`, and `deepspeed`
- if you already have a working `Video-R1` environment, it should be the preferred starting point for running `ViSS-R1`

A simple setup flow is:

```bash
python -m venv .venv
source .venv/bin/activate
bash setup.sh
```

The `setup.sh` script installs the `r1-v` package and the extra dependencies used by this repository.

## Data

We use [`Video-R1-data`](https://huggingface.co/datasets/Video-R1/Video-R1-data) as the base data source and `preprocess_SFT/anno_sft_72b.py` to distill CoT data, including the SSL-transformation setting used to build transformed reasoning samples.

The script supports the SSL data generation flow through `--transformation`. In the public version, machine-specific paths were removed and replaced with configurable environment variables:

- `QWEN72B_MODEL_PATH`: path to the `Qwen2.5-VL-72B-Instruct` checkpoint
- `VIDEO_R1_DATA_ROOT`: root directory for the image/video assets referenced by the annotation data

Example:

```bash
export QWEN72B_MODEL_PATH=/path/to/Qwen2.5-VL-72B-Instruct
export VIDEO_R1_DATA_ROOT=/path/to/Video-R1-data
CUDA_VISIBLE_DEVICES=0,1 python preprocess_SFT/anno_sft_72b.py --split 0 --transformation
```

## Models

Our released checkpoints are available at:

- `Pretext-GRPO`: `https://huggingface.co/1Xinyaozxy/ViSS-R1-PretextGRPO`
- `ViSS-R1`: `https://huggingface.co/1Xinyaozxy/ViSS-R1`

## Usage

Before running the code, you should adapt local paths for:

- model checkpoints
- training datasets
- evaluation datasets
- output directories

Main entrypoints:

```bash
bash src/scripts/run_sft_video.sh
bash src/scripts/run_grpo_video.sh
bash src/eval_bench.sh
python src/inference_example.py --model_path /path/to/model --video_path /path/to/video.mp4 --question "Your question here"
```

For the SFT stage, launch `src/scripts/run_sft_video.sh` to train the model to generate the `<transform>` tag together with the transformed reasoning output.

For the RL stage, launch `src/scripts/run_grpo_video_SSL.sh` to further enhance the SFT base model through SSL transformations.

All SSL transformations are implemented in [`process_transformation`](./src/r1-v/src/open_r1/trainer/grpo_trainer.py) in [grpo_trainer.py](/mnt/bn/omninas/fangbo/ViSS-R1/src/r1-v/src/open_r1/trainer/grpo_trainer.py:790). The current image branch includes `rotate`, `flip`, and `puzzle`, while the video branch includes `rotate`, `shuffle`, and `arrow`.

## Evaluation

Use `src/eval_bench_parallel.sh` for parallelized evaluation:

```bash
bash src/eval_bench_parallel.sh
```

This evaluation flow is designed for parallel acceleration and requires at least 6 GPUs for6 benchmarks.

## Notes

- This repository does not include datasets, benchmark videos, or model weights.
- The archive was cleaned before publication: backup folders, training logs, and checkpoint metadata were removed.
- Public entry scripts were simplified so they no longer depend on machine-specific `source /...` environment commands.

## Acknowledgements

We thank [`Video-R1`](https://github.com/tulerfeng/Video-R1) for the original codebase and release foundation of this project.

We sincerely appreciate the contributions of the open-source community. The related projects include [`R1-V`](https://github.com/StarsfieldAI/R1-V) and [`DeepSeek-R1`](https://github.com/deepseek-ai/DeepSeek-R1).

## Citation

If you find `ViSS-R1` useful in your research, please consider citing:

```bibtex
@InProceedings{Fang_2026_ViSSR1,
    author    = {Fang, Bo and Song, YuXin and Sun, Haoyuan and Zhang, Xinyao and Wu, Qiangqiang and Wu, Wenhao and Chan, Antoni B.},
    title     = {ViSS-R1: Self-Supervised Reinforcement Video Reasoning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    year      = {2026},
    pages     = {11190-11200}
}
```

## License

See [LICENSE](LICENSE).
