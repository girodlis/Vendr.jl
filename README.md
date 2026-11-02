# Vendr.jl

A Julia package to orchestrate glacier inversion campaigns using the ODINN.jl ecosystem.

## Quick Start

All campaigns use the **single `Project.toml` at the root** (`../../../`). You can run from either location:

### Option 1: From the campaign directory
```bash
cd inversions/02_synthetic/21_autoAD
julia --project=../../../ run.jl
```

### Option 2: From Vendr.jl root
```bash
julia --project=. inversions/02_synthetic/21_autoAD/run.jl
```

## Campaign Structure

Each campaign directory contains:
- `config/` - Configuration: `campaign.toml` and `scenarios.jl` (generates `scenarios.toml`)
- `data/` - Campaign-specific data
- `outputs/` - Results and logs
- `run.jl` - Main execution script

### Scenarios

`scenarios.toml` is not versioned: generate it before the first run with

```bash
julia inversions/02_synthetic/21_classic_no_reg/config/scenarios.jl
```

Each `[[scenarios]]` entry overrides the defaults below (all keys are optional):

```toml
[[scenarios]]
id = "S1"
loss_type = "HV"              # "H", "V" or "HV"
use_tim = false               # use the TI mass balance model
sparsity_H = true             # restrict H observations to the GlaThiDa mask
sparsity_V_sigma = 0.7        # V sparsity mask smoothing (0.0 = dense V)
sparsity_V_threshold = 0.02   # V sparsity mask threshold
regularization_weight = 3e37  # rheology regularization weight on A (0.0 = none)
# rgi_ids = ["RGI60-11.02773"] # override the campaign glaciers
```

### Compact Run Workflow

```mermaid
flowchart LR
    A[run.jl] --> B[Load config]
    B --> C[Build campaign context]
    C --> D[Loop scenarios]
    D --> E[Prepare setup]
    E --> F[Run inversion]
    F --> G[Save scenario outputs]
    G --> D
    D -->|done| H[Build campaign summary]
    H --> I[(outputs/results)]
```

## Current Scope (v1)

- Inversion target: `A` (scalar)
- Data mode: synthetic ground truth
- Time mode: short non-transient setup (`2010:2011`)

## Main API

- `build_campaign_run_context(campaign_root)`
- `run_campaign!(context)`

## Next Steps

The code contains explicit `TODO:` comments for planned extensions:

- Trainable `C` (creep parameter) target
- Transient configuration (longer time periods)
- Observed-data mode (without synthetic ground truth generation)
- Target-aware metrics and plots (`A` vs `C`)
- Distributed `A` and field-based error analysis
