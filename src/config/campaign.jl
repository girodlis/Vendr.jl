using TOML

"""
    ScenarioConfig

Configuration for a single inversion scenario within a campaign.

# Fields
- `id::String`: Scenario identifier (e.g., "S01")
- `loss_type::String`: Type of empirical loss ("H", "V", or "HV")
- `use_tim::Bool`: Enable time-integrated mass balance in loss
- `sparsity_H::Bool`: Mask ice thickness observations (true = sparse)
- `sparsity_V_sigma::Float64`: Gaussian blur radius for velocity mask (0 = no mask)
- `sparsity_V_threshold::Float64`: Threshold for binary velocity masking
- `use_optim_autoAD::Bool`: Enable automatic differentiation in UDE optimization
- `rgi_ids::Vector{String}`: RGI glacier identifiers to process
"""
Base.@kwdef struct ScenarioConfig
    id::String
    loss_type::String
    use_tim::Bool
    sparsity_H::Bool
    sparsity_V_sigma::Float64
    sparsity_V_threshold::Float64
    use_optim_autoAD::Bool = true
    rgi_ids::Vector{String}
end

"""
    CampaignConfig

Global configuration for a glacier inversion campaign.

# Fields
- `name::String`: Campaign name for identification
- `results_dir::String`: Output directory for results
- `rgi_ids::Vector{String}`: Default RGI glacier IDs (can be overridden per scenario)
- `epochs_adam::Int`: Number of Adam optimization epochs (must be > 0)
- `epochs_linesearch::Int`: Number of L-BFGS line search epochs (must be > 0)
- `invert_target::Symbol`: Parameter to invert (:A only in v1)
- `law_A::Symbol`: Flow rate law (:TemperateA, :CuffeyPatersonScalar, :CuffeyPatersonGridded)
- `law_C::Symbol`: Basal sliding law (:SyntheticC, :None)
- `tspan::Tuple{Float64, Float64}`: Simulation time span (start_year, end_year)
- `step::Float64`: Simulation time step for inversion (1/12 for monthly, 1.0 for yearly)
- `step_gt::Union{Float64, Nothing}`: Time step for ground truth observations (if nothing, uses `step`)
- `n_observations_gt::Union{Int, Nothing}`: Number of evenly-spaced GT observations (if set, overrides `step_gt`)
- `spinup::Bool`: Enable initial spinup simulation
- `spinup_tspan::Tuple{Float64, Float64}`: Spinup time span
"""
Base.@kwdef struct CampaignConfig
    name::String
    results_dir::String
    rgi_ids::Vector{String}
    gridScalingFactor::Int
    epochs_adam::Int
    epochs_linesearch::Int
    invert_target::Symbol
    law_A::Symbol
    law_C::Symbol
    tspan::Tuple{Float64, Float64}
    step::Float64
    step_gt::Union{Float64, Nothing} = nothing
    spinup::Bool
    spinup_tspan::Tuple{Float64, Float64}
end

Base.@kwdef struct CampaignRunContext
    campaign::CampaignConfig
    scenarios::Vector{ScenarioConfig}
    campaign_root::String
    results_dir::String
    csv_file::String
    scenario_inversions::Vector  # Vector{Inversion}
    scenario_predictions::Vector  # Vector{Prediction}
    n_epochs_adam::Int
    n_epochs_linesearch::Int
end

function resolve_config_path(campaign_root::String, config_name::String)::String
    standard = joinpath(campaign_root, "config", config_name)
    fallback = joinpath(campaign_root, config_name)

    isfile(standard) && return standard
    isfile(fallback) && return fallback
    return standard
end

function _deep_merge_dicts(base::Dict{String, Any}, overrides::Dict{String, Any})
    merged = deepcopy(base)
    for (key, override_value) in overrides
        if haskey(merged, key) && merged[key] isa Dict && override_value isa Dict
            merged[key] = _deep_merge_dicts(
                Dict{String, Any}(merged[key]),
                Dict{String, Any}(override_value),
            )
        else
            merged[key] = deepcopy(override_value)
        end
    end
    return merged
end

function _toml_section(raw::Dict{String, Any}, key::String)
    section = get(raw, key, Dict{String, Any}())
    section isa Dict || error("Expected [$key] to be a TOML table")
    return Dict{String, Any}(section)
end

# ##############################################
# ###########       CAMPAIGN     ##############
# ##############################################

"""
    load_campaign_config(campaign_root::String; overrides=Dict()) -> CampaignConfig

Load and validate campaign configuration from TOML file.

Looks for `campaign.toml` in `campaign_root/config/` or `campaign_root/`.
Supports merging configuration overrides for programmatic customization.

# Arguments
- `campaign_root::String`: Root directory of campaign
- `overrides::Dict{String, Any}`: Override values merged into loaded config

# Returns
Validated `CampaignConfig` struct

# Throws
- `ErrorException`: If required config sections are missing or invalid
"""
function load_campaign_config(
        campaign_root::String;
        overrides::Dict{String, Any} = Dict{String, Any}(),
)::CampaignConfig
    path = resolve_config_path(campaign_root, "campaign.toml")
    raw = isfile(path) ? TOML.parsefile(path) : Dict{String, Any}()
    raw = _deep_merge_dicts(raw, overrides)

    campaign_section = _toml_section(raw, "campaign")
    run_section = _toml_section(raw, "run")
    model_section = _toml_section(raw, "model")
    simulation_section = _toml_section(raw, "simulation")

    name = String(get(campaign_section, "name", "campaign"))
    results_dir = String(get(campaign_section, "results_dir", get(run_section, "results_dir", "outputs/results")))
    if !isabspath(results_dir)
        results_dir = joinpath(campaign_root, results_dir)
    end

    rgi_ids = _as_string_vector(get(run_section, "rgi_ids", nothing), [
        "RGI60-11.02072",
        "RGI60-11.02773",
        "RGI60-11.02249",
        "RGI60-11.00746",
    ])
    isempty(rgi_ids) && error("No RGI IDs specified in campaign configuration")

    epochs_adam = Int(get(run_section, "epochs_adam", 20))
    epochs_adam > 0 || error("epochs_adam must be positive, got $epochs_adam")
    
    epochs_linesearch = Int(get(run_section, "epochs_linesearch", 30))
    epochs_linesearch > 0 || error("epochs_linesearch must be positive, got $epochs_linesearch")

    invert_target = Symbol(String(get(model_section, "invert_target", "A")))

    law_A = Symbol(String(get(model_section, "law_A", "TemperateA")))
    # TODO: default to SyntheticC once C inversion path is implemented.
    law_C = Symbol(String(get(model_section, "law_C", "None")))

    gridScalingFactor = Int(get(simulation_section, "gridScalingFactor", 1))

    tspan = _as_tspan(get(simulation_section, "tspan", nothing), (2010.0, 2011.0); key = "tspan")

    step = Float64(get(simulation_section, "step", 1/12))
    #step = (s = String(get(simulation_section, "step", "monthly"))) == "monthly" ? 1/12 : 1.0

    step_gt_raw = get(simulation_section, "step_gt", nothing)
    step_gt = step_gt_raw === nothing ? nothing : Float64(step_gt_raw)

    spinup = Bool(get(simulation_section, "spinup", true))
    spinup_tspan = _as_tspan(get(simulation_section, "spinup_tspan", nothing), (2009.0, 2010.0); key = "spinup_tspan")


    return CampaignConfig(
        name = name,
        results_dir = results_dir,
        rgi_ids = rgi_ids,
        gridScalingFactor = gridScalingFactor,
        epochs_adam = epochs_adam,
        epochs_linesearch = epochs_linesearch,
        invert_target = invert_target,
        law_A = law_A,
        law_C = law_C,
        tspan = tspan,
        step = step,
        step_gt = step_gt,
        spinup = spinup,
        spinup_tspan = spinup_tspan,
    )
end

"""
    validate_campaign(campaign::CampaignConfig) -> Nothing

Validate campaign configuration for consistency and support in current package version.

Checks:
- `invert_target` is :A (v1 limitation)
- `law_A` is one of supported options
- `law_C` is one of supported options

# Throws
- `ErrorException`: If configuration is invalid
"""
function validate_campaign(campaign::CampaignConfig)::Nothing
    campaign.invert_target == :A || error(
        "v1 currently supports invert_target = :A only (got $(campaign.invert_target)). "
    )
    campaign.law_A in (:TemperateA, :CuffeyPatersonScalar, :CuffeyPatersonGridded) || error(
        "Unsupported law_A=$(campaign.law_A). Choose from: :TemperateA, :CuffeyPatersonScalar, :CuffeyPatersonGridded"
    )
    campaign.law_C in (:SyntheticC, :None) || error(
        "Unsupported law_C=$(campaign.law_C). Choose from: :SyntheticC, :None"
    )
    return nothing
end

"""
    generate_tstops_gt(campaign::CampaignConfig) -> Vector{Float64}

Generate time stops for ground truth observations based on campaign config.

If `step_gt` is set, uses this step; otherwise uses `step` (default behavior).
"""
function generate_tstops_gt(campaign::CampaignConfig)::Vector{Float64}
    t0, t1 = campaign.tspan
    step = something(campaign.step_gt, campaign.step)
    tstops = collect(range(t0, t1, step = step))
    t1 ∉ tstops && push!(tstops, t1)
    return tstops
end

# ##############################################
# ###########       SCENARIOS     ##############
# ##############################################

function _scenario_from_dict(entry::Dict{String, Any}, fallback_id::String, fallback_rgi_ids::Vector{String})
    return ScenarioConfig(
        id = String(get(entry, "id", fallback_id)),
        loss_type = String(get(entry, "loss_type", "H")),
        use_tim = Bool(get(entry, "use_tim", false)),
        sparsity_H = Bool(get(entry, "sparsity_H", false)),
        sparsity_V_sigma = Float64(get(entry, "sparsity_V_sigma", 0.0)),
        sparsity_V_threshold = Float64(get(entry, "sparsity_V_threshold", 0.0)),
        use_optim_autoAD = Bool(get(entry, "use_optim_autoAD", false)),
        rgi_ids = haskey(entry, "rgi_ids") ? String.(entry["rgi_ids"]) : fallback_rgi_ids,
    )
end

"""
    load_scenarios(campaign_root::String, campaign::CampaignConfig; overrides=Dict()) -> Vector{ScenarioConfig}

Load scenario matrix from TOML configuration file.

Looks for `scenarios.toml` in `campaign_root/config/` or `campaign_root/`.
Each scenario defines variations on loss function, masking, and other hyperparameters.

# Arguments
- `campaign_root::String`: Campaign root directory
- `campaign::CampaignConfig`: Campaign config (provides RGI ID defaults)
- `overrides::Dict{String, Any}`: Override scenario values

# Returns
Vector of `ScenarioConfig` structs, empty if scenarios.toml not found

# Throws  
- `ErrorException`: If scenario entries have invalid types
"""
function load_scenarios(
        campaign_root::String,
        campaign::CampaignConfig;
        overrides::Dict{String, Any} = Dict{String, Any}(),
)::Vector{ScenarioConfig}
    path = resolve_config_path(campaign_root, "scenarios.toml")
    raw = isfile(path) ? TOML.parsefile(path) : Dict{String, Any}()
    raw = _deep_merge_dicts(raw, overrides)

    scenarios = ScenarioConfig[]
    scenarios_list = get(raw, "scenarios", Any[])
    if isempty(scenarios_list)
        return scenarios
    end
    for (idx, entry) in enumerate(scenarios_list)
        scenario = _scenario_from_dict(Dict{String, Any}(entry), "S$(idx)", campaign.rgi_ids)
        push!(scenarios, scenario)
    end
    @info "Loaded $(length(scenarios)) scenarios successfully"
    return scenarios
end

# ##############################################
# #############      CONTEXT     ###############
# ##############################################

"""
    build_campaign_run_context(campaign_root::String; overrides=Dict()) -> CampaignRunContext

Load campaign and scenarios, set up results directory, and create execution context.

# Arguments
- `campaign_root::String`: Root directory containing campaign.toml and scenarios.toml
- `overrides::Dict{String, Any}`: Configuration overrides for programmatic use

# Returns
`CampaignRunContext` with all necessary information for `run_campaign!`

# Side Effects
- Creates `results_dir` if it does not exist
- Initializes CSV results file path
"""
function build_campaign_run_context(
        campaign_root::String;
        overrides::Dict{String, Any} = Dict{String, Any}(),
)::CampaignRunContext
    campaign = load_campaign_config(campaign_root; overrides = overrides)
    scenarios = load_scenarios(campaign_root, campaign; overrides = overrides)

    results_dir = campaign.results_dir
    isdir(results_dir) || mkpath(results_dir)
    csv_file = joinpath(results_dir, "campaign_summary.csv")

    scenario_inversions = []
    scenario_predictions = []

    return CampaignRunContext(
        campaign = campaign,
        scenarios = scenarios,
        campaign_root = campaign_root,
        results_dir = results_dir,
        csv_file = csv_file,
        scenario_inversions = scenario_inversions,
        scenario_predictions = scenario_predictions,
        n_epochs_adam = campaign.epochs_adam,
        n_epochs_linesearch = campaign.epochs_linesearch,
    )
end

# Helpers

function _as_tspan(value, fallback::Tuple{Float64, Float64}; key::String = "tspan")
    if isnothing(value)
        return fallback
    end
    length(value) == 2 || error("$(key) must contain exactly two values")
    return (Float64(value[1]), Float64(value[2]))
end

_as_string_vector(value, fallback::Vector{String}) = isnothing(value) ? fallback : String.(value)


