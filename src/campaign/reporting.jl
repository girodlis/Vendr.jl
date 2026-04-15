export to_scalar,
       relative_error_temperate_ice_percent,
       append_inversion_summary_csv,
       restore_unmasked_references!,
       save_comparison_grids!,
       print_summary_table

to_scalar(x::Number) = Float64(x)

function to_scalar(x::AbstractArray)
    isempty(x) && error("Cannot extract scalar from empty array")
    return Float64(first(x))
end

function relative_error_temperate_ice_percent(inversion, glacier_idx; θ=nothing)
    # TODO: generalize to target-aware metrics (A/C) and field metrics for distributed inversions.
    θ_value = isnothing(θ) ? inversion.results.stats.θ : θ
    isnothing(θ_value) && error("No trained parameters (θ) found in inversion.results.stats.θ")

    A_gt_value = to_scalar(Huginn.polyA_PatersonCuffey()(0.0))
    A_pred_value = to_scalar(eval_law(inversion.model.iceflow.A, inversion, glacier_idx, (;), θ_value))
    return 100 * ((A_pred_value - A_gt_value) / A_gt_value)
end

function append_inversion_summary_csv(
        csv_path,
        inversion;
        scenario_id,
        rgi_ids,
        loss_type,
        use_tim,
        epoch_adam,
        epoch_linesearch,
        sparsity_H,
        sparsity_V_sigma,
        sparsity_V_threshold,
        overwrite::Bool = false,
    )
    # TODO: extend CSV schema with target + metric_name + metric_value for A/C and distributed runs.
    file_exists = isfile(csv_path)
    θ = inversion.results.stats.θ
    mode = (overwrite || !file_exists) ? "w" : "a"
    write_header = overwrite || !file_exists

    open(csv_path, mode) do io
        if write_header
            println(io,
                "scenario,rgi_id,loss_type,use_tim,epoch_adam,epoch_linesearch,sparsity_H,sparsity_V_sigma,sparsity_V_threshold,relative_error_percent")
        end

        for (glacier_idx, rgi_id) in enumerate(rgi_ids)
            rel_err_pct = relative_error_temperate_ice_percent(inversion, glacier_idx; θ=θ)
            println(io,
                "$(scenario_id),$(rgi_id),$(loss_type),$(use_tim),$(epoch_adam),$(epoch_linesearch),$(sparsity_H),$(sparsity_V_sigma),$(sparsity_V_threshold),$(rel_err_pct)")
        end
    end
end

function restore_unmasked_references!(glacier_result, unmasked_references)
    rgi_id = glacier_result.rgi_id

    if haskey(unmasked_references.thickness, rgi_id)
        glacier_result.H_ref = deepcopy(unmasked_references.thickness[rgi_id])
    end

    if haskey(unmasked_references.velocity, rgi_id)
        velocity_ref = unmasked_references.velocity[rgi_id]
        glacier_result.V_ref = deepcopy(velocity_ref.vabs)
    end

    return glacier_result
end

function save_comparison_grids!(;
    rgi_ids::Vector{String},
    scenario_results_by_glacier::Dict,
    scenario_labels::Vector{String},
    results_dir::String,
)
    for rgi_id in rgi_ids
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

    return nothing
end

function print_summary_table(csv_file::String)
    df = CSV.read(csv_file, DataFrame)

    summary = combine(
        groupby(df, [:loss_type, :use_tim, :sparsity_H, :sparsity_V_sigma, :sparsity_V_threshold]),
        :relative_error_percent => mean => :mean_rel_err_pct,
        :relative_error_percent => median => :median_rel_err_pct,
        :relative_error_percent => minimum => :min_rel_err_pct,
        :relative_error_percent => maximum => :max_rel_err_pct,
        :relative_error_percent => std => :std_rel_err_pct,
        nrow => :n,
    )

    sort!(summary, [:loss_type, :use_tim, :sparsity_H, :sparsity_V_sigma, :sparsity_V_threshold])
    println(summary)
    return summary
end
