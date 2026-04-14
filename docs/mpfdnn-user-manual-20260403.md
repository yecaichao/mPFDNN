# mPFDNN / FAMS_DNN User Manual

## Scope

This manual describes the current training interface and runtime behavior of the `FAMS_DNN` branch in this repository.

The authoritative training entrypoint is:

- `python -m ptagnn.cli.my_run_train`

## Current Model Behavior

Current `FAMS_DNN` behavior differs from the original early branch in several important ways:

- `radial_version` default is `v2`
- default loss is `universal`
- `freeze_pair_scaling_epochs` default is `20`
- interaction scale is element-dependent and trainable, initialized to `1.0`
- interaction shift is fixed to `0.0`
- isolated atoms are forced to have zero interaction contribution, so isolated-atom energy is exactly `E0`
- final readout in `FAMS_DNN` is linear
- `MLP_irreps` is currently kept for interface compatibility
- `use_self_connection` is configurable
- `virials_impl` now defaults to `atomic`

## Stress and Virials Convention

The current default virial path is:

- edge force
- atomic virial
- graph virial
- graph stress

Current defaults:

- `--virials_impl=atomic`
- `stress = -virials / volume`

Compatibility path:

- `--virials_impl=displacement`
  Use the previous displacement/autograd graph-level implementation

`atomic_virials` is still available as an output field and may be reused internally when virials or stress are requested.

## Entrypoints

### Single-GPU training

Recommended invocation:

```bash
python -m ptagnn.cli.my_run_train \
  --device=cuda \
  ...
```

### Multi-GPU training

Current single-node DDP training is supported through:

```bash
python -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=<N> \
  -m ptagnn.cli.my_run_train \
  --device=cuda \
  ...
```

Notes:

- DDP is implemented in `ptagnn.cli.my_run_train`
- validation, checkpoint saving, final evaluation, and model export are only performed on rank `0`
- for fair single-card vs multi-card comparison, keep the global batch size constant

Example:

- single GPU: `--batch_size=16`
- 2 GPU DDP: `--batch_size=8` per rank

## Data Requirements

Supported training files are ASE-readable extended XYZ files.

### Required fields

- `positions`
- `numbers`
- `energy`
- `forces`

### Optional fields

- `stress`
- `virial` or `virials`
- `charges`
- `config_type`
- `config_weight`
- `config_energy_weight`
- `config_forces_weight`
- `config_stress_weight`
- `config_virials_weight`

### Current automatic behavior

- If `virials` is provided but `stress` is absent, stress is reconstructed from virials and cell volume
- If `stress` is provided but `virials` is absent, virials are reconstructed from stress and cell volume
- If both `stress` and `virials` are absent, their weights become `0`
- For `universal`, `stress`, `virials`, and `huber`, stress and virial computations are disabled automatically when the training set has no active stress or virial labels

## Current Energy Formula

For atom `i`, current `FAMS_DNN` uses:

```math
E_i = E_0(Z_i) + s(Z_i)\sum_t e_i^{(t)}
```

where:

- `E0(Z_i)` is the element baseline term
- `s(Z_i)` is an element-dependent trainable scale
- there is no trainable global interaction shift term
- for isolated atoms, the interaction sum is forced to `0`

Current radial pair-scaling initialization for `pairwise_tanh_chebyshev`:

- `a ≈ 1.5`
- `b = -2.25`

## Recommended Baseline Configurations

### Small stable baseline

- `num_channels=16`
- `num_recursions=1`
- `product='[3,3]'`
- `pair_scaling=element_channel_l`
- `radial_with_l`
- `freeze_pair_scaling_epochs=20`
- `loss=universal`
- `virials_impl=atomic`

### Fair single-GPU vs multi-GPU comparison

- single GPU: `batch_size=16`
- 2 GPU DDP: `batch_size=8` per rank

### High-throughput comparison

- multi-GPU with the same per-rank batch as single-GPU
- this increases global batch size and may change optimization behavior

## Parameter Reference

### Experiment and output paths

- `--name`
  Required experiment tag
- `--seed`
  Default: `123`
- `--log_dir`
  Default: `logs`
- `--model_dir`
  Default: `.`
- `--checkpoints_dir`
  Default: `checkpoints`
- `--results_dir`
  Default: `results`
- `--downloads_dir`
  Default: `downloads`

### Runtime and precision

- `--device`
  `cpu | cuda | mps`
- `--default_dtype`
  `float32 | float64`
  Current default: `float64`
- `--log_level`
  Default: `INFO`
- `--num_workers`
  Default: `0`
- `--pin_memory`
  Default: `true`
- `--persistent_workers`
  Default: `true`
- `--prefetch_factor`
  Default: `2`

### Model family and geometry

- `--model`
  Current working branch uses `FAMS_DNN`
- `--r_max`
  Cutoff radius
- `--max_ell`
  Maximum spherical-harmonic order
- `--num_recursions`
  Effective interaction layers: `num_recursions + 1`
- `--product`
  Supports either a single integer or a list such as `'[3,2]'`
- `--interaction`
  Main interaction block class
- `--interaction_first`
  First interaction block class
- `--num_channels`
  Embedding channels
- `--max_L`
  Used to derive scalar hidden irreps when set with `num_channels`
- `--hidden_irreps`
  Hidden node irreps
- `--MLP_irreps`
  Kept for interface compatibility
- `--gate`
  `silu | tanh | abs | None`
- `--use_self_connection`
  `true | false`
  Current default: `true`

### Radial branch

- `--radial_type`
  `bessel | gaussian | chebyshev | pairwise_tanh_chebyshev`
- `--num_radial_basis`
  Number of radial basis functions
- `--num_cutoff_basis`
  Polynomial cutoff basis order
- `--radial_MLP`
  Width list for radial weight MLP or grouped MLP
  Default: `[16]`
- `--pair_scaling`
  `none | element | element_channel | element_l | element_channel_l`
  Parser default: `element_channel`
- `--radial_with_l`
  Enable explicit `l`-resolved radial channels
- `--radial_version`
  `v1 | v2`
  Current default: `v2`
- `--activation_checkpoint`
  Enable activation checkpointing

### Scaling and neighbor normalization

- `--scaling`
  `std_scaling | rms_forces_scaling | no_scaling`
  Current default: `rms_forces_scaling`
- `--avg_num_neighbors`
  Message normalization factor
- `--compute_avg_num_neighbors`
  If true, compute from the training set

### Stress and virials control

- `--virials_impl`
  `atomic | displacement`
  Current default: `atomic`
- `--compute_stress`
  Enable stress output when required by the workflow
- `--compute_forces`
  Enable force output
- `--virials_key`
  Default: `virials`
- `--stress_key`
  Default: `stress`

### Data files and field mapping

- `--train_file`
  Required
- `--valid_file`
  Optional
- `--valid_fraction`
  Used when `valid_file` is not provided
- `--test_file`
  Optional
- `--stream_train`
  Stream the training XYZ instead of loading it fully into memory
- `--stream_shuffle_buffer_size`
  Approximate shuffle buffer size for streamed training
  Default: `1024`
- `--E0s`
  Atomic baseline initialization mode
  Supported values:
  - `average`
  - `reference`
  - explicit dictionary string
- `--E0s_reference_file`
  Required when `--E0s=reference`
- `--energy_key`
  Default: `energy`
- `--forces_key`
  Default: `forces`
- `--charges_key`
  Default: `charges`

### Loss selection

- `--loss`
  `ef | weighted | forces_only | virials | stress | huber | universal | l1l2energyforces`
  Current default: `universal`

Current practical meaning:

- `weighted`
  MSE energy + MSE forces
- `forces_only`
  MSE forces only
- `virials`
  MSE energy + MSE forces + MSE virials
- `stress`
  MSE energy + MSE forces + MSE stress
- `huber`
  Huber energy + Huber forces + Huber stress
- `universal`
  Huber energy + conditional Huber forces + optional Huber stress
- `l1l2energyforces`
  MAE energy + normed-force loss

### Loss weights

- `--forces_weight`
- `--energy_weight`
- `--virials_weight`
- `--stress_weight`
- `--config_type_weights`
- `--huber_delta`

SWA counterparts:

- `--swa_forces_weight`
- `--swa_energy_weight`
- `--swa_virials_weight`
- `--swa_stress_weight`

### Optimizer and schedule

- `--optimizer`
  `adam | adamw`
- `--batch_size`
- `--valid_batch_size`
- `--lr`
- `--pair_scaling_lr`
  Defaults to `5 * --lr` when pair scaling is active
- `--swa_lr`
- `--weight_decay`
- `--amsgrad`
- `--scheduler`
  Current default: `ReduceLROnPlateau`
- `--lr_factor`
- `--scheduler_patience`
- `--lr_scheduler_gamma`

### EMA and SWA

- `--ema`
- `--ema_decay`
- `--swa`
- `--start_swa`

### Training control

- `--max_num_epochs`
- `--max_num_steps`
- `--patience`
- `--eval_interval`
- `--skip_validation`
- `--freeze_pair_scaling_epochs`
  Current default: `20`
- `--log_pair_ab_interval`
- `--clip_grad`
- `--restart_latest`
- `--keep_checkpoints`
- `--save_cpu`
- `--skip_final_evaluation`

### Logging and W&B

- `--wandb`
- `--wandb_project`
- `--wandb_entity`
- `--wandb_name`
- `--wandb_log_hypers`

## Example Commands

### Single-GPU example

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
  --virials_impl=atomic \
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

### Single-node multi-GPU example

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
  --radial_version=v2 \
  --virials_impl=atomic \
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

### Streaming example

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
  --radial_version=v2 \
  --virials_impl=atomic \
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

Streaming note:

- `--stream_train` requires `--valid_file`

## Current DDP Status

Current single-node multi-GPU DDP behavior:

- model initialization uses identical seed on all ranks
- atomic energies are broadcast from rank `0`
- `avg_num_neighbors` is broadcast from rank `0`
- training uses `DistributedSampler` for in-memory datasets
- validation and final export run only on rank `0`
- `find_unused_parameters=True` is enabled for robustness

## Recommended Practice

- Use `ptagnn.cli.my_run_train` for DDP and streamed training
- Keep global batch size fixed when comparing single-GPU and multi-GPU optimization quality
- Use `--virials_impl=displacement` only when you explicitly need the previous graph-level compatibility path
- Prefer generic, explicit dataset paths in scripts intended for sharing
