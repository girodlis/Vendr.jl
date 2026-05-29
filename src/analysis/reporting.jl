using CSV
using DataFrames

"""
    save_epoch_counts_csv(context::CampaignRunContext;
        csv_path=joinpath(context.results_dir, "epoch_counts.csv"),
        n_epochs_adam=nothing,
        overwrite=true,
    ) -> String

Save a CSV with the realized Adam/BFGS epoch counts for each scenario in a
completed campaign run.

The counts are inferred from the finished loss history of each inversion.
`total_epochs` is the number of optimization updates, computed as
`max(length(losses) - 1, 0)`.
"""
function save_epoch_counts_csv(
        context::CampaignRunContext;
        csv_path::AbstractString = joinpath(context.results_dir, "epoch_counts.csv"),
        n_epochs_adam::Union{Int, Nothing} = nothing,
        overwrite::Bool = true,
    )::String
    adam_target = something(n_epochs_adam, context.campaign.epochs_adam)
    completed_scenarios = min(length(context.scenario_inversions), length(context.scenarios))
    file_exists = isfile(csv_path)
    mode = (overwrite || !file_exists) ? "w" : "a"
    write_header = overwrite || !file_exists

    open(csv_path, mode) do io
        if write_header
            println(io, "scenario,scenario_label,total_epochs,epoch_adam,epoch_bfgs,n_losses,status")
        end

        for i in 1:completed_scenarios
            scenario = context.scenarios[i]
            inversion = context.scenario_inversions[i]
            losses = inversion.results.stats.losses
            n_losses = length(losses)
            total_epochs = max(n_losses - 1, 0)
            epoch_adam = min(adam_target, total_epochs)
            epoch_bfgs = max(total_epochs - epoch_adam, 0)
            label = scenario_label(i, scenario)

            println(io, join([scenario.id, label, string(total_epochs), string(epoch_adam), string(epoch_bfgs), string(n_losses), "completed"], ","))
        end

        for i in (completed_scenarios + 1):length(context.scenarios)
            scenario = context.scenarios[i]
            label = scenario_label(i, scenario)
            println(io, join([scenario.id, label, "", "", "", "", "pending"], ","))
        end
    end

    return csv_path
end

"""
    save_results_csv(
        csv_path::AbstractString,
        inversion,
        prediction;
        scenario_id::AbstractString,
        rgi_ids::Vector{String},
        loss_type::String,
        use_tim::Bool,
        epoch_adam::Int,
        epoch_linesearch::Int,
        sparsity_H::Bool,
        sparsity_V_sigma::Float64,
        sparsity_V_threshold::Float64,
        target::Symbol=:A,
        overwrite::Bool=false,
    ) -> Nothing

Save per-glacier relative errors to CSV results file.

Appends or creates CSV with computed relative errors comparing inversion
results against ground truth prediction.

# Arguments
- `csv_path::AbstractString`: Path to results CSV file
- `inversion`: Completed inversion with trained θ
- `prediction`: Ground truth simulation
- `scenario_id::AbstractString`: Scenario identifier
- `rgi_ids::Vector{String}`: Glacier RGI identifiers
- `loss_type::String`: Loss function type ("H", "V", "HV")
- `use_tim::Bool`: Whether time-integrated mass balance was used
- `epoch_adam::Int`: Number of Adam epochs
- `epoch_linesearch::Int`: Number of L-BFGS epochs
- `sparsity_H::Bool`: Ice thickness masking flag
- `sparsity_V_sigma::Float64`: Velocity mask Gaussian blur radius
- `sparsity_V_threshold::Float64`: Velocity mask threshold
- `target::Symbol`: Parameter evaluated (:A or :C)
- `overwrite::Bool`: Overwrite file instead of append

# Side Effects
- Creates/appends to CSV file
"""
function save_results_csv(
        csv_path::AbstractString,
        inversion,
        prediction;
        scenario_id::AbstractString,
        rgi_ids::Vector{String},
        loss_type::String,
        use_tim::Bool,
        epoch_adam::Int,
        epoch_linesearch::Int,
        sparsity_H::Bool,
        sparsity_V_sigma::Float64,
        sparsity_V_threshold::Float64,
        target::Symbol=:A,
        overwrite::Bool = false,
)::Nothing
    
    file_exists = isfile(csv_path)
    θ = inversion.results.stats.θ
    mode = (overwrite || !file_exists) ? "w" : "a"
    write_header = overwrite || !file_exists
    
    n_results = 0

    open(csv_path, mode) do io
        if write_header
            println(io,
                "scenario,rgi_id,target,loss_type,use_tim,epoch_adam,epoch_linesearch,sparsity_H,sparsity_V_sigma,sparsity_V_threshold,relative_error_percent")
        end

        for (glacier_idx, rgi_id) in enumerate(rgi_ids)
            rel_err_pct = compute_relative_error(inversion, prediction, glacier_idx; θ=θ, target=target)
            println(io,
                "$(scenario_id),$(rgi_id),$(target),$(loss_type),$(use_tim),$(epoch_adam),$(epoch_linesearch),$(sparsity_H),$(sparsity_V_sigma),$(sparsity_V_threshold),$(rel_err_pct)")
            n_results += 1
        end
    end
    return nothing
end

function save_comparison_grids!(;
    rgi_ids::Vector{String},
    results_dir::String,
    scenarios::Vector{ScenarioConfig},
    scenario_inversions::Union{Vector, Nothing} = nothing,
    scenario_predictions::Union{Vector, Nothing} = nothing,
)
    # Derive scenario_results_by_glacier on-demand from scenario_inversions
    scenario_results_by_glacier = Dict(rgi_id => Sleipnir.Results[] for rgi_id in rgi_ids)
    if !isnothing(scenario_inversions)
        for inversion in scenario_inversions
            for glacier_result in inversion.results.simulation
                push!(scenario_results_by_glacier[glacier_result.rgi_id], glacier_result)
            end
        end
    end
    
    # Generate scenario_labels on-demand from scenarios
    scenario_labels = [scenario_label(i, scenario) for (i, scenario) in enumerate(scenarios)]
    
    # Save thickness grids
    for (i, rgi_id) in enumerate(rgi_ids)
        scenario_results = scenario_results_by_glacier[rgi_id]
        isempty(scenario_results) && continue
        thickness_grid = plot_thickness_differences_grid(
            scenario_results,
            scenario_labels;
            figsize = (1800, 1800),
            plotContour = true,
        )
        save(joinpath(results_dir, "thickness_differences_grid_$(rgi_id).png"), thickness_grid)
    end

    # Save target grids if provided
    if !isnothing(scenario_inversions) && !isnothing(scenario_predictions)
        for (i, rgi_id) in enumerate(rgi_ids)
            scenario_results = scenario_results_by_glacier[rgi_id]
            isempty(scenario_results) && continue
            #TODO: plot target differences grid only when griddedA
            target_grid = plot_target_difference_grid(
                scenario_inversions,
                scenario_predictions,
                i,
                :A,
                scenario_labels;
                relative_error = true,
                results_dir = results_dir,
            )
            save(joinpath(results_dir, "target_differences_grid_$(rgi_id).png"), target_grid)
        end
    end

    return nothing
end

function print_summary(csv_file::String)
    df = CSV.read(csv_file, DataFrame)
    if !hasproperty(df, :target)
        df.target = fill("A", nrow(df))
    end

    summary = combine(
        groupby(df, [:target, :loss_type, :use_tim, :sparsity_H, :sparsity_V_sigma, :sparsity_V_threshold]),
        :relative_error_percent => mean => :mean_rel_err_pct,
        :relative_error_percent => median => :median_rel_err_pct,
        :relative_error_percent => minimum => :min_rel_err_pct,
        :relative_error_percent => maximum => :max_rel_err_pct,
        :relative_error_percent => std => :std_rel_err_pct,
        nrow => :n,
    )

    CSV.sort!(summary, [:target, :loss_type, :use_tim, :sparsity_H, :sparsity_V_sigma, :sparsity_V_threshold])
    println(summary)
    CSV.write(joinpath(dirname(csv_file), "summary_table.csv"), summary)
    return summary
end


