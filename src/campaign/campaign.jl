"""
    campaign.jl

Campaign orchestration API.

This file exposes a simple v1 layout:
- `config.jl`: types + TOML/config/context handling
- `model_builders.jl`: laws, params, and model builders
- `runners.jl`: scenario/campaign execution
- `reporting.jl`: metrics, CSV, and postprocess helpers
"""

# Include all submodules
include("config.jl")
include("model_builders.jl")
include("runners.jl")
include("reporting.jl")

# Export public API types
export CampaignConfig,
       ScenarioConfig,
       CampaignRunContext

# Export public configuration functions
export load_campaign_config,
       load_scenarios,
       validate_campaign,
       default_simple_scenarios,
       build_campaign_run_context

# Export public runner functions
export prepare_scenario_setup,
       run_scenario!,
       run_campaign!

# Export public helper functions
export scenario_epochs_adam,
       scenario_epochs_linesearch,
       scenario_banner,
       scenario_label
