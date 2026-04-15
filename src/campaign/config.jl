"""
    campaign/config.jl

Campaign configuration and context setup.

This file intentionally groups types, parsing utilities, and config/context
construction to keep the v1 workflow easy to read in one place.
"""

using TOML
using Sleipnir

# -----------------------------
# Types
# -----------------------------

Base.@kwdef struct CampaignConfig
    name::String
    results_dir::String
    rgi_ids::Vector{String}
    epochs_adam::Int
    epochs_linesearch::Int
    invert_target::Symbol
    law_A::Symbol
    law_C::Symbol
    spinup::Bool
    spinup_tspan::Tuple{Float64, Float64}
    use_transient_inversion::Bool
end

Base.@kwdef struct ScenarioConfig
    id::String
    loss_type::String
    use_tim::Bool
    sparsity_H::Bool
    sparsity_V_sigma::Float64
    sparsity_V_threshold::Float64
    rgi_ids::Vector{String}
    epochs_adam::Union{Nothing, Int} = nothing
    epochs_linesearch::Union{Nothing, Int} = nothing
end

Base.@kwdef struct CampaignRunContext
    campaign::CampaignConfig
    scenarios::Vector{ScenarioConfig}
    campaign_root::String
    results_dir::String
    csv_file::String
    scenario_results_by_glacier::Dict{String, Vector{Sleipnir.Results}}
    scenario_labels::Vector{String}
    n_epochs_adam::Int
    n_epochs_linesearch::Int
end

# -----------------------------
# Small parsing helpers
# -----------------------------

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

_as_string_vector(value, fallback::Vector{String}) = isnothing(value) ? fallback : String.(value)

function _as_tspan(value, fallback::Tuple{Float64, Float64})
    if isnothing(value)
        return fallback
    end
    length(value) == 2 || error("spinup_tspan must contain exactly two values")
    return (Float64(value[1]), Float64(value[2]))
end

# -----------------------------
# Config loading
# -----------------------------

function resolve_config_path(campaign_root::String, config_name::String)::String
    standard = joinpath(campaign_root, "config", config_name)
    fallback = joinpath(campaign_root, config_name)

    isfile(standard) && return standard
    isfile(fallback) && return fallback
    return standard
end

function load_campaign_config(
        campaign_root::String;
        overrides::Dict{String, Any} = Dict{String, Any}(),
)
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

    epochs_adam = Int(get(run_section, "epochs_adam", 20))
    epochs_linesearch = Int(get(run_section, "epochs_linesearch", 30))

    invert_target = Symbol(String(get(model_section, "invert_target", "A")))
    law_A = Symbol(String(get(model_section, "law_A", "TemperateA")))
    # TODO: default to SyntheticC once C inversion path is implemented.
    law_C = Symbol(String(get(model_section, "law_C", "None")))

    spinup = Bool(get(simulation_section, "spinup", true))
    spinup_tspan = _as_tspan(get(simulation_section, "spinup_tspan", nothing), (2009.0, 2010.0))
    # TODO: add configurable tspan/tstops for transient runs.
    use_transient_inversion = Bool(get(simulation_section, "use_transient_inversion", false))

    return CampaignConfig(
        name = name,
        results_dir = results_dir,
        rgi_ids = rgi_ids,
        epochs_adam = epochs_adam,
        epochs_linesearch = epochs_linesearch,
        invert_target = invert_target,
        law_A = law_A,
        law_C = law_C,
        spinup = spinup,
        spinup_tspan = spinup_tspan,
        use_transient_inversion = use_transient_inversion,
    )
end

function _scenario_from_dict(entry::Dict{String, Any}, fallback_id::String, fallback_rgi_ids::Vector{String})
    return ScenarioConfig(
        id = String(get(entry, "id", fallback_id)),
        loss_type = String(get(entry, "loss_type", "H")),
        use_tim = Bool(get(entry, "use_tim", false)),
        sparsity_H = Bool(get(entry, "sparsity_H", false)),
        sparsity_V_sigma = Float64(get(entry, "sparsity_V_sigma", 0.0)),
        sparsity_V_threshold = Float64(get(entry, "sparsity_V_threshold", 0.0)),
        rgi_ids = haskey(entry, "rgi_ids") ? String.(entry["rgi_ids"]) : fallback_rgi_ids,
        epochs_adam = haskey(entry, "epochs_adam") ? Int(entry["epochs_adam"]) : nothing,
        epochs_linesearch = haskey(entry, "epochs_linesearch") ? Int(entry["epochs_linesearch"]) : nothing,
    )
end

function load_scenarios(
        campaign_root::String,
        campaign::CampaignConfig;
        overrides::Dict{String, Any} = Dict{String, Any}(),
)
    path = resolve_config_path(campaign_root, "scenarios.toml")
    raw = isfile(path) ? TOML.parsefile(path) : Dict{String, Any}()
    raw = _deep_merge_dicts(raw, overrides)

    scenarios = ScenarioConfig[]
    for (idx, entry) in enumerate(raw["scenarios"])
        push!(scenarios, _scenario_from_dict(Dict{String, Any}(entry), "S$(idx)", campaign.rgi_ids))
    end
    return scenarios
end

function validate_campaign(campaign::CampaignConfig)
    campaign.invert_target == :A || error("v1 currently supports invert_target = :A only.")
    campaign.law_A in (:TemperateA, :CuffeyPaterson) || error("Unsupported law_A=$(campaign.law_A).")
    campaign.law_C in (:SyntheticC, :None) || error("Unsupported law_C=$(campaign.law_C).")
    return nothing
end

function build_campaign_run_context(
        campaign_root::String;
        overrides::Dict{String, Any} = Dict{String, Any}(),
)
    campaign = load_campaign_config(campaign_root; overrides = overrides)
    scenarios = load_scenarios(campaign_root, campaign; overrides = overrides)

    results_dir = campaign.results_dir
    isdir(results_dir) || mkpath(results_dir)
    csv_file = joinpath(results_dir, "inversion_summary.csv")

    scenario_results_by_glacier = Dict(rgi_id => Sleipnir.Results[] for rgi_id in campaign.rgi_ids)
    scenario_labels = String[]

    return CampaignRunContext(
        campaign = campaign,
        scenarios = scenarios,
        campaign_root = campaign_root,
        results_dir = results_dir,
        csv_file = csv_file,
        scenario_results_by_glacier = scenario_results_by_glacier,
        scenario_labels = scenario_labels,
        n_epochs_adam = campaign.epochs_adam,
        n_epochs_linesearch = campaign.epochs_linesearch,
    )
end
