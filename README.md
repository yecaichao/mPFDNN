# mPFDNN / FAMS_DNN

mPFDNN is a Material-Property-Field-based Deep Neural Network framework for atomistic modeling. The current training branch is centered on the `FAMS_DNN` model family and exposes its main training interface through `ptagnn.cli.my_run_train`.

## Installation

```bash
git clone <your-fork-or-mirror-url> mPFDNN
cd mPFDNN

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python setup.py install
```

## Current Highlights

Compared with the original public branch, the current codebase includes:

- `pairwise_tanh_chebyshev` radial features
- pairwise scaling modes: `none`, `element`, `element_channel`, `element_l`, `element_channel_l`
- explicit `l`-resolved radial features via `--radial_with_l`
- `radial_version=v2` as the current default
- trainable element-dependent interaction scaling
- configurable `--use_self_connection`
- activation checkpointing via `--activation_checkpoint`
- single-node multi-GPU DDP
- streamed XYZ training via `--stream_train`
- pair-scaling warmup and dedicated learning-rate control
- `universal` as the default loss

## Current FAMS_DNN Behavior

- The recommended training entrypoint is `python -m ptagnn.cli.my_run_train`.
- This branch is intentionally focused on `FAMS_DNN`; legacy non-FAMS model families and their dedicated training paths have been removed.
- `radial_version` defaults to `v2`.
- `loss` defaults to `universal`.
- `freeze_pair_scaling_epochs` defaults to `20`.
- `virials_impl` now defaults to `atomic`.
- When virials or stress are requested, the default path computes edge forces, builds atomic virials, then aggregates graph virials and stress.
- The compatibility path `--virials_impl=displacement` keeps the previous displacement/autograd implementation.
- Output tensors follow the convention `stress = -virials / volume`.
- The final `FAMS_DNN` readout is linear.
- `MLP_irreps` is still accepted for interface compatibility.

## Key Training Parameters

### Model and radial branch

- `--radial_type`
  `bessel | gaussian | chebyshev | pairwise_tanh_chebyshev`
- `--pair_scaling`
  `none | element | element_channel | element_l | element_channel_l`
  Default: `element_channel`
- `--radial_with_l`
  Enable explicit `l`-resolved radial channels
- `--radial_version`
  `v1 | v2`
  Default: `v2`
- `--activation_checkpoint`
  Enable activation checkpointing
- `--use_self_connection`
  `true | false`
  Default: `true`

### Stress and virials

- `--virials_impl`
  `atomic | displacement`
  Default: `atomic`
- `--compute_stress`
  Enable stress output when required by the workflow
- `--virials_key`
  Reference virials field name. Default: `virials`
- `--stress_key`
  Reference stress field name. Default: `stress`

### Streaming and data loading

- `--stream_train`
  Stream the training XYZ instead of loading it fully into memory
- `--stream_shuffle_buffer_size`
  Approximate shuffle buffer size for streamed training
  Default: `1024`
- `--num_workers`
  DataLoader workers. Default: `0`
- `--persistent_workers`
  Default: `true`
- `--prefetch_factor`
  Default: `2`

### Optimization and control

- `--loss`
  Default: `universal`
- `--pair_scaling_lr`
  Overrides the learning rate for pair-scaling parameters
  Default: `5 * --lr` when pair scaling is active
- `--freeze_pair_scaling_epochs`
  Default: `20`
- `--max_num_steps`
  Optional hard stop on optimizer steps
- `--skip_validation`
  Skip validation passes during training
- `--skip_final_evaluation`
  Skip final train/valid/test evaluation

## Data Requirements

Training files must be ASE-readable extended XYZ files.

Required fields:

- `positions`
- `numbers`
- `energy`
- `forces`

Optional fields:

- `stress`
- `virial` or `virials`
- `charges`
- `config_type`
- `config_weight`
- `config_energy_weight`
- `config_forces_weight`
- `config_stress_weight`
- `config_virials_weight`

Automatic behavior:

- If `virials` is present but `stress` is missing, stress is reconstructed from virials and cell volume.
- If `stress` is present but `virials` is missing, virials are reconstructed from stress and cell volume.
- If both are missing, their loss weights become `0`.
- For `universal`, `stress`, `virials`, and `huber`, stress and virial computations are disabled automatically when the training set has no active stress or virial labels.

## Recommended Baseline

```text
num_channels=16
num_recursions=1
product='[3,3]'
pair_scaling=element_channel_l
radial_with_l
freeze_pair_scaling_epochs=20
loss=universal
virials_impl=atomic
```

## Example Commands

### 1. Single-GPU in-memory training

```bash
python -m ptagnn.cli.my_run_train \
  --device=cuda \
  --train_file=/path/to/data/train.xyz \
  --valid_file=/path/to/data/valid.xyz \
  --test_file=/path/to/data/test.xyz \
  --name=single_gpu_demo \
  --model=FAMS_DNN \
  --E0s=average \
  --loss=universal \
  --interaction_first=Residual_InteractionBlock \
  --interaction=Residual_InteractionBlock \
  --radial_type=pairwise_tanh_chebyshev \
  --pair_scaling=element_channel_l \
  --radial_with_l \
  --radial_version=v2 \
  --num_recursions=1 \
  --product='[3,3]' \
  --max_ell=3 \
  --r_max=6.0 \
  --max_L=0 \
  --num_channels=16 \
  --num_radial_basis=10 \
  --batch_size=16 \
  --valid_batch_size=16 \
  --max_num_epochs=120
```

### 2. Single-node multi-GPU DDP training

```bash
python -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=2 \
  -m ptagnn.cli.my_run_train \
  --device=cuda \
  --train_file=/path/to/data/train.xyz \
  --valid_file=/path/to/data/valid.xyz \
  --test_file=/path/to/data/test.xyz \
  --name=ddp_demo \
  --model=FAMS_DNN \
  --E0s=average \
  --loss=universal \
  --interaction_first=Residual_InteractionBlock \
  --interaction=Residual_InteractionBlock \
  --radial_type=pairwise_tanh_chebyshev \
  --pair_scaling=element_channel_l \
  --radial_with_l \
  --num_recursions=1 \
  --product='[3,3]' \
  --max_ell=3 \
  --r_max=6.0 \
  --max_L=0 \
  --num_channels=16 \
  --num_radial_basis=10 \
  --batch_size=8 \
  --valid_batch_size=16 \
  --max_num_epochs=120
```

Notes:

- In DDP, `--batch_size` is per-rank.
- If single-GPU uses `--batch_size=16`, then 2-GPU DDP should use `--batch_size=8` per rank to keep the global batch size fixed.

### 3. Streaming training

```bash
python -m ptagnn.cli.my_run_train \
  --device=cuda \
  --train_file=/path/to/data/train_large.xyz \
  --valid_file=/path/to/data/valid.xyz \
  --test_file=/path/to/data/test.xyz \
  --name=stream_demo \
  --model=FAMS_DNN \
  --E0s=average \
  --loss=universal \
  --interaction_first=Residual_InteractionBlock \
  --interaction=Residual_InteractionBlock \
  --radial_type=pairwise_tanh_chebyshev \
  --pair_scaling=element_channel_l \
  --radial_with_l \
  --num_recursions=1 \
  --product='[3,3]' \
  --max_ell=3 \
  --r_max=6.0 \
  --max_L=0 \
  --num_channels=16 \
  --num_radial_basis=10 \
  --stream_train \
  --stream_shuffle_buffer_size=4096 \
  --batch_size=16 \
  --valid_batch_size=16 \
  --max_num_epochs=120
```

Notes:

- Streaming training requires `--valid_file`.
- `--stream_shuffle_buffer_size=0` disables buffer shuffling.
- Validation and final export still run on rank 0 under DDP.

## Compatibility Note

The current default virial implementation is atomic:

```bash
--virials_impl=atomic
```

To force the previous graph-level displacement/autograd implementation:

```bash
--virials_impl=displacement
```

## Additional Documentation

- Detailed user manual: `docs/mpfdnn-user-manual-20260403.md`
