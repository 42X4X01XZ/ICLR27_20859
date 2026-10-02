# A Light Bilevel Refinement Aligns Self-Supervised Representations for Stronger Task-Specific Learning

Official PyTorch implementation of **BiSSLight** — an efficient and scalable variant of the BiSSL framework, which uses a bilevel optimization stage to let the downstream task objective guide self-supervised learning in refining pretrained representations before parameter-efficient fine-tuning (PEFT).

BiSSLight replaces the costly conjugate-gradient (CG) hypergradient approximation of the original BiSSL method with a **block-wise Matrix-Free Approximate Curvature (M-FAC)** implicit gradient approximation. This cuts compute time by more than 10x and peak memory by roughly 3x compared to BiSSL on a ViT-H/14 backbone, while retaining its downstream performance gains.


## Contents

- [Repository overview](#repository-overview)
- [Installation](#installation)
- [Data preparation](#data-preparation)
- [Pretrained MAE checkpoints](#pretrained-mae-checkpoints)
- [Usage](#usage)
  - [BiSSLight training](#bisslight-training)
  - [Downstream PEFT](#downstream-peft)
  - [Baselines](#baselines)
- [Key configuration options](#key-configuration-options)
- [Outputs](#outputs)

## Repository overview

```
ICLR27_20859/
├── data/                  # Train/val split files for CUB200 and Oxford-IIIT Pet
├── source/                # Core library: models, bilevel trainers, solvers, datasets, logging
└── runs/                  # Executable run scripts and their configuration classes
    ├── bisslight/         #   BiSSLight / BiSSL bilevel training
    └── fine_tune/         #   Downstream PEFT (called fine-tuning in the code)
```

The bilevel training entry points live under `runs/bisslight/classification/mae_lora/`: `run.py` trains BiSSLight (M-FAC), and `original_bissl/run.py` the original BiSSL baseline (CG). The downstream PEFT entry points live under `runs/fine_tune/classification/vit_lora/`. The method itself is implemented in `source/bissl/` — see `bisslight_trainer.py` and the implicit-gradient solver `ij_solver_mfac.py` (M-FAC).

All hyperparameters configs (`config.py`) are defined as [typed-argument-parser](https://github.com/swansonk14/typed-argument-parser) classes next to their run scripts (`run.py`). Every run script prints its full configuration at launch, and each class-level argument doubles as a command-line flag (underscores are converted to dashes, e.g. `--lora_rank` becomes `--lora-rank`).

*A note on terminology: the code refers to the downstream adaptation stage as fine-tuning for simplicity (hence the `fine_tune` directories and `ft` flags/run types), but all downstream adaptation in this repository — as in the paper — is LoRA-based PEFT on a frozen backbone. This README follows the paper and says PEFT.*

## Installation

Requires **Python >= 3.10** and a CUDA-capable GPU.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Data preparation

Set `--dataset-root` to a single directory shared by all datasets. Each dataset handler passes this root directly to its (torchvision-based) dataset class, so the datasets are expected directly under it in their standard layouts, e.g.:

```
<dataset_root>/
├── tv_splits/          # Provided in this repository (required for CUB200 and Pets)
├── imagenet/           # ImageNet-1k (only needed for the lower-level pretext task)
│   ├── train/          #   <wnid>/<...>.JPEG
│   └── val/
├── CUB_200_2011/       # CUB200-2011
├── .../                # Pets, Flowers-102, DTD, FGVC-Aircraft, VOC2007
│                       # (standard torchvision layouts)
```

- The lower (pretext) level trains on **ImageNet-1k without labels**; an ImageNet copy is therefore required for all bilevel runs. The upper level and the PEFT runs only need the downstream dataset.
- The smaller downstream datasets can be downloaded automatically by passing `--download 1`. ImageNet must be obtained and arranged manually.
- CUB200 and Pets have no official validation split; the repository ships the train/val partitions used in the paper under `data/tv_splits/`. Either point `--dataset-root` at the repository's `data/` directory (after placing the datasets inside it), or copy `tv_splits/` into your dataset root.

Supported downstream datasets (`--d_dataset`): `dtd`, `pets`, `flowers`, `aircrafts`, `cub200`, `voc07` (VOC07 is evaluated with 11-point mAP, all others with top-1 accuracy).

## Pretrained MAE checkpoints

The bilevel runs expect the official MAE checkpoints (pretrained on ImageNet-1k at 224x224), which must be named exactly as below:

| Backbone  | Checkpoint file                        | Download |
|-----------|----------------------------------------|----------|
| ViT-B/16  | `mae_pretrain_vit_base_full.pth`       | `https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base_full.pth` |
| ViT-L/16  | `mae_pretrain_vit_large_full.pth`      | `https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large_full.pth` |
| ViT-H/14  | `mae_pretrain_vit_huge_full.pth`       | `https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge_full.pth` |

The backbone architecture is inferred from the checkpoint filename. The corresponding pretrained MAE decoder weights are loaded from the same checkpoint file.

## BiSSLight-trained LoRA models

LoRA weights (rank 64, applied to attention query/value projections, frozen backbone) pretrained with BiSSLight on MAE backbones:

| Backbone  | Pets | DTD | VOC07 | Flowers | CUB200 | Aircrafts |
|-----------|--------|------|---------|-----------|-----|-------|
| ViT-B/16  | link soon available | link soon available | link soon available | link soon available | link soon available | link soon available |
| ViT-L/16  | link soon available | link soon available | link soon available | link soon available | link soon available | link soon available |
| ViT-H/14  | link soon available | link soon available | link soon available | link soon available | link soon available | link soon available |

## Usage

All runs are launched with `torchrun` (single node). The bilevel runs use one process per GPU (DDP). The PEFT runs use a single process to drive their repeats/learning-rate search with [Ray Tune](https://ray.readthedocs.io/), which runs one PEFT trial per visible GPU in parallel. 

If `--wandb-api-key` is omitted, metrics are logged to the console only.

### BiSSLight training

Applies the BiSSLight bilevel stage to an MAE-pretrained backbone for a given downstream dataset, using LoRA (rank 64 by default in the paper) on the attention query and value projections of every layer while keeping the backbone frozen:

```bash
torchrun --nproc_per_node=1 runs/bisslight/classification/mae_lora/run.py \
    --dataset-root /path/to/dataset_root \
    --model-root /path/to/output_dir \
    --pretrained-backbone-path /path/to/mae_pretrain_vit_base_full.pth \
    --d-dataset cub200 \
    --d-lr 0.00067 \
    --lora-rank 64 \
```

This runs T=500 alternations of `lower_num_iter=20` pretext (MAE) steps and `upper_num_iter=8` downstream (Classification) steps, with a 20-epoch linear-probing warm-up of the classification head (`--d-head-warmup-epochs`), the regularization weight `lam=0.001`, and the M-FAC quantities computed on CPU (`--ij-grad-calc-device`).

Bilevel training fits two separate LoRA sets on the shared frozen backbone: one optimized on the pretext task (the lower level) and one on the downstream task (the upper level). Only the **fitted lower-level LoRA parameters** are saved — these are the method's output, since the downstream objective has steered the pretext-side optimization toward representations aligned with the downstream task, and they initialize the subsequent PEFT stage.

Notable arguments: `--epochs` (number of alternations), `--lam`, `--lower-num-iter`, `--upper-num-iter`, `--p-batch-size` (pretext batch size, default 256), `--d-batch-size` (downstream batch size, default 64), `--p-lr` (default 0.01, LoRA modules) and `--p-head-lr` (default 1e-4, MAE decoder), `--use-amp`, `--use-grad-checkpointing`, `--ij-solver-lora-param-grouping` (M-FAC block partitioning; the paper uses the default `layer`), and `--unfreeze-norm-layers`.

### Downstream PEFT

Downstream adaptation with standard LoRA PEFT, initialized from the fitted lower-level LoRA parameters saved by the BiSSLight training (see above). Pass the backbone checkpoint and config YAML produced by the BiSSLight training (both written to `--model-root`); downstream dataset, LoRA rank, image size and related settings are inherited from the BiSSLight config:

```bash
torchrun --nproc_per_node=1 runs/fine_tune/classification/vit_lora/post_bisslight_ft/run.py \
    --dataset-root /path/to/dataset_root \
    --model-root /path/to/output_dir \
    --pretrained-backbone-path /path/to/mae_pretrain_vit_base_full.pth \
    --bissl-pretrained-model-path /path/to/output_dir/bisslight_..._backbone.pth.tar \
    --bissl-config-path /path/to/output_dir/bisslight_..._config.yaml \
    --use-hpo 1 \
```

By default (`--use-hpo 1`), the script runs a grid search over learning rates `[1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]`, which is the range used for BiSSLight-trained backbones in the paper. Pass `--use-hpo 0 --lr <lr>` to run PEFT with a specified learning rate instead (still with a randomly generated seed, unless that is also specified).

### Baselines

**Direct PEFT (applied directly to the pretrained backbone):** the paper's PEFT baseline. This is the exact same PEFT procedure as the Downstream PEFT run above — same trainer, LoRA configuration, and Ray-driven learning-rate search — differing only in the initialization (freshly initialized LoRA modules on the pretrained backbone, rather than the fitted lower-level LoRA parameters saved by the BiSSLight training) and in the learning-rate grid, which spans `[1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]`. The scripts are separate only to keep the differing configuration handling apart:

```bash
torchrun --nproc_per_node=1 runs/fine_tune/classification/vit_lora/post_pretext_ft/run.py \
    --dataset-root /path/to/dataset_root \
    --model-root /path/to/output_dir \
    --pretrained-backbone-path /path/to/mae_pretrain_vit_base_full.pth \
    --dataset cub200 \
    --lora-rank 64 \
    --use-hpo 1 \
```

**Original BiSSL (CG-based):** the original BiSSL method with identical architectures, schedules and LoRA configuration, differing only in the conjugate-gradient hypergradient approximation (`--hinv-solver cg_lw`, with `--cg-iter-num` CG iterations per iHVP estimate). The CG solver is far more compute- and memory-hungry than M-FAC, and the BiSSL results in the paper were produced with 2-4 GPUs per run:

```bash
torchrun --nproc_per_node=4 runs/bisslight/classification/mae_lora/original_bissl/run.py \
    ... # same arguments as the BiSSLight training above
```

**PEFT from a BiSSL-trained backbone:** analogous to the Downstream PEFT run, but launched from `runs/fine_tune/classification/vit_lora/post_bisslight_ft/original_bissl/run.py` using the outputs of the BiSSL (CG) run.

## Key configuration options

The table below lists the most relevant flags across all scripts, partitioned by where they apply. The bilevel and PEFT scripts express a few shared concepts under different names (`--d-dataset` vs. `--dataset`); such pairs are combined into a single row, with the bilevel variant first and the PEFT variant second (in both the argument and default columns). This is a quick reference, not an exhaustive list: standard training utilities (`--seed`, `--use-amp`, `--use-grad-checkpointing`, drop-path rates, weight decay, and so on) are omitted, and every flag is defined (with a comment) in the typed-argument-parser class next to its run script; each script prints its full configuration at launch. The original BiSSL baseline additionally exposes the CG-specific `--cg-iter-num`, `--cg-lam-dampening` and `--cg-verbose`.

| Argument | Applies to | Default | Description |
|---|---|---|---|
| `--dataset-root` | All runs | — (required) | Dataset root directory (see [Data preparation](#data-preparation)) |
| `--model-root` | All runs | `None` | Output directory for checkpoints and configs; required to save outputs (nothing is saved if omitted) |
| `--pretrained-backbone-path` | All runs | — (required) | Path to an official MAE checkpoint (see table above) |
| `--lora-rank` | All runs | `None` | LoRA rank r (paper default: 64), applied to attention query/value projections; inherited from the BiSSLight config when running PEFT from a BiSSLight-trained backbone |
| `--epochs` | All runs | `500` / `200` | BiSSLight: number of alternations T. PEFT: training epochs |
| `--d-dataset` / `--dataset` | BiSSLight / PEFT | — (required) | Downstream dataset: `dtd`, `pets`, `flowers`, `aircrafts`, `cub200`, `voc07` |
| `--d-lr` / `--lr` | BiSSLight / PEFT | — (required) / `None` | Downstream learning rate; searched over by the HPO grid in the PEFT runs unless `--use-hpo 0` |
| `--d-batch-size` / `--batch-size` | BiSSLight / PEFT | `64` / `256` | Downstream batch size |
| `--lower-num-iter` | BiSSLight | `20` | Lower-level (pretext) steps per alternation |
| `--upper-num-iter` | BiSSLight | `8` | Upper-level (downstream) steps per alternation |
| `--lam` | BiSSLight | `0.001` | Lower-level regularization weight (also fixes the M-FAC damping) |
| `--p-batch-size` | BiSSLight | `256` | Pretext (lower-level) batch size |
| `--p-lr` / `--p-head-lr` | BiSSLight | `0.01` / `1e-4` | Lower-level LoRA / MAE decoder learning rates |
| `--d-head-warmup-epochs` | BiSSLight | `20` | Linear-probing warm-up epochs for the classification head |
| `--use-hpo` | PEFT | `1` | Grid-search the learning rate; `0` for a fixed `--lr` |
| `--bissl-pretrained-model-path` | PEFT on BiSSLight-trained backbone | — (required) | Only for the PEFT runs that initialize from a BiSSLight-trained backbone (the Downstream PEFT run, and its original-BiSSL analogue): path to the backbone checkpoint saved by the BiSSLight training, used as the LoRA initialization |
| `--bissl-config-path` | PEFT on BiSSLight-trained backbone | `None` | Only for the same runs: path to the config YAML saved by the BiSSLight training; optional but recommended |

## Outputs

If `--model-root` is set, the bilevel runs periodically write two files that share a common name prefix `<prefix> = <run_type>_<backbone>_l-<pretext_task>_u-<downstream_task>-<d_dataset>_lora-r<rank>_seed<seed>` (the LoRA rank and seed parts appear only when set), e.g. `bisslight_vit_base_patch16_l-mae_u-classification-cub200_lora-r64_seed0`:

- `<prefix>_backbone.pth.tar` — the fitted lower-level parameters (LoRA weights, plus unfrozen norm layers if enabled), used to initialize downstream PEFT;
- `<prefix>_config.yaml` — the full configuration used for the run, stored for reference; the PEFT runs on a BiSSLight-trained backbone expect it via `--bissl-config-path`.

The PEFT runs also save checkpoints in `--model-root` (when provided; overwritten whenever validation accuracy improves), using a simpler prefix without the task and dataset tags: `<prefix> = <run_type>_<backbone>_<dataset>_lora-r<rank>_seed<seed>`. Each save writes the fitted backbone (LoRA) parameters to `<prefix>_backbone.pth.tar`, the classification head to `<prefix>_head.pth.tar`, and the run configuration to `<prefix>_config.yaml`.
