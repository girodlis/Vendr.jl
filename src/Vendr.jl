__precompile__() 

"""
    Vendr

Orchestrate glacier inversion campaigns using the ODINN.jl ecosystem.

Vendr provides a framework for running systematic parameter inversion experiments
on multiple glaciers with different configurations. It handles:

- **Campaign management**: Define experiments via TOML configuration
- **Scenario matrix**: Combine different loss functions, masking strategies, and parameters
- **Ground truth generation**: Create synthetic reference simulations
- **Model inversion**: Train neural network inversions via ODINN.jl
- **Results analysis**: Compute metrics and generate comparison plots

## Typical Workflow

```julia
using Vendr

# Load campaign configuration from directory
context = build_campaign_run_context("path/to/campaign")

# Run all scenarios
run_campaign!(context)

# Analyze results
print_summary(context.csv_file)
```

## Key Modules

- [`Config`](@ref): Configuration loading and validation
- [`Models`](@ref): Simulation and inversion model builders  
- [`Simulations`](@ref): Campaign execution orchestration
- [`Analysis`](@ref): Metrics computation and reporting
- [`Plotting`](@ref): Visualization utilities
"""
module Vendr

# ##############################################
# ###########       PACKAGES     ##############
# ##############################################

# ODINN subpackages
using Reexport
@reexport using ODINN # imports Muninn and Sleipnir

# Utilities
using JLD2
#using Revise
using Statistics
using CSV
using DataFrames
using TOML

# ##############################################
# ############    SETUP           ###############
# ##############################################

#using Logging

# ##############################################
# ############   VENDR LIBRARIES  ##############
# ##############################################

# Configuration
include("config/campaign.jl")

# Models
include("models/laws.jl")
include("models/parameters.jl")
include("models/inversion.jl")
include("models/forward.jl")

# Utilities
include("utils/formatting.jl")
include("utils/io.jl")

# Analysis and reporting
include("analysis/metrics.jl")
include("analysis/reporting.jl")

# Simulations
include("simulations/preparation.jl")
include("simulations/runners.jl")

# Plotting utilities
include("plotting/plotting.jl")

# ##############################################
# #############      EXPORTS     ##############
# ##############################################

# Configuration and structures
export CampaignConfig,
       ScenarioConfig,
       CampaignRunContext

# Setup and configuration functions
export resolve_config_path,
       load_campaign_config,
       load_scenarios,
       validate_campaign,
       generate_tstops_gt,
       build_campaign_run_context

# Models
export build_parameters,
       resolve_law_A,
       resolve_law_C,
       resolve_loss,
       resolve_autoAD,
       build_ground_truth,
       spinup_glacier_state!,
       build_inversion_model

# Simulation utilities
export prepare_scenario_setup,
       prepare_scenario,
       scenario_banner,
       scenario_label

# Simulation execution
export run_scenario!,
       run_campaign!,
       spinup_historical_forward_simulation!

# Results and reporting
export compute_relative_error,
       relative_error_temperate_ice_percent,
       save_results_csv,
       append_inversion_summary_csv,
       restore_unmasked_references!,
       save_comparison_grids!,
       print_summary,
       print_summary_table

# Plotting
export plot_loss_evolution,
       plot_losses,
       plot_relative_error_boxplot,
       plot_thickness_differences_grid,
       plot_target_difference_grid,
       plot_target_comparison

end # module