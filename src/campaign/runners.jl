"""
    campaign/runners.jl

Main orchestration and execution functions for scenarios and campaigns.
"""

"""
    prepare_scenario_setup(campaign::CampaignConfig, scenario::ScenarioConfig) -> Tuple{Model, Vector{Glacier}, Parameters, NamedTuple}

Prepare all components needed for a scenario's inversion run.

Sets up parameters, ground truth predictions, inversion model, and applies sparsity masks.
"""
function prepare_scenario_setup(campaign::CampaignConfig, scenario::ScenarioConfig)
    params = _build_parameters(campaign, scenario)
    # TODO: branch by data_mode (:synthetic vs :observed) and bypass synthetic GT generation for observed data.
    prediction = _build_ground_truth_prediction(campaign, scenario, params)
    glaciers = prediction.glaciers

    unmasked_thickness_ref = Dict(
        glacier.rgi_id => deepcopy(glacier.thicknessData.H)
        for glacier in glaciers
        if !isnothing(glacier.thicknessData)
    )

    unmasked_velocity_ref = Dict(
        glacier.rgi_id => (
            vabs = deepcopy(glacier.velocityData.vabs),
            vx = deepcopy(glacier.velocityData.vx),
            vy = deepcopy(glacier.velocityData.vy),
        )
        for glacier in glaciers
        if !isnothing(glacier.velocityData)
    )

    unmasked_references = (
        thickness = unmasked_thickness_ref,
        velocity = unmasked_velocity_ref,
    )

    # Apply velocity sparsity mask (spatially coherent random masking)
    if scenario.sparsity_V_sigma > 0.0 && scenario.sparsity_V_threshold > 0.0
        for glacier in glaciers
            vdata = glacier.velocityData
            ntstops = length(vdata.vabs)
            base_mask = .!glacier.mask
            vel_mask = Sleipnir.random_spatially_coherent_mask(base_mask;
                sigma = scenario.sparsity_V_sigma,
                threshold = scenario.sparsity_V_threshold,
            )
            vel_mask_f = Float64.(vel_mask)
            for it in 1:ntstops
                glacier.velocityData.vx[it] .*= vel_mask_f
                glacier.velocityData.vy[it] .*= vel_mask_f
                glacier.velocityData.vabs[it] .*= vel_mask_f
            end
        end
    end

    # Apply thickness sparsity mask (Glathida-only)
    if scenario.sparsity_H
        for glacier in glaciers
            hdata = glacier.thicknessData
            ntstops = length(hdata.H)
            glathida_mask = glacier.H_glathida .> 0.0
            for it in 1:ntstops
                glacier.thicknessData.H[it] .*= glathida_mask
            end
        end
    end

    inv_model = _build_inversion_model(campaign, scenario, params, glaciers)
    return inv_model, glaciers, params, unmasked_references
end

"""
    scenario_banner(i::Int, n::Int, scenario::ScenarioConfig) -> String

Format a header string for scenario execution.
"""
function scenario_banner(i::Int, n::Int, scenario::ScenarioConfig)
    return "=== Scenario $(i)/$(n) : $(scenario.id) | loss=$(scenario.loss_type) tim=$(scenario.use_tim) H=$(scenario.sparsity_H) V=$(scenario.sparsity_V_sigma), $(scenario.sparsity_V_threshold)"
end

"""
    scenario_label(i::Int, scenario::ScenarioConfig) -> String

Format a human-readable label for a scenario (used in plots and tables).
"""
function scenario_label(i::Int, scenario::ScenarioConfig)
    return "$(scenario.id) | loss=$(scenario.loss_type) | tim=$(scenario.use_tim) | H=$(scenario.sparsity_H) | Vσ=$(scenario.sparsity_V_sigma) | Vthr=$(scenario.sparsity_V_threshold)"
end

"""
    run_scenario!(context::CampaignRunContext, scenario::ScenarioConfig, scenario_idx::Int) -> Nothing

Execute a single scenario: invert, save results, and accumulate context.

Steps:
1. Prepare scenario (models, parameters, sparsity masks)
2. Run inversion (forward + gradient-based optimization)
3. Append metrics to CSV
4. Plot loss evolution
5. Restore original (unmasked) observational data
6. Update campaign context with results
"""
function run_scenario!(context::CampaignRunContext, scenario::ScenarioConfig, scenario_idx::Int)
    inv_model, glaciers, params, unmasked_references = prepare_scenario_setup(context.campaign, scenario)
    inversion = Inversion(inv_model, glaciers, params)
    run!(inversion)

    # TODO: make summary metrics target-aware (A vs C) and field-aware for distributed parameters.
    append_inversion_summary_csv(
        context.csv_file,
        inversion;
        scenario_id = scenario.id,
        rgi_ids = scenario.rgi_ids,
        loss_type = scenario.loss_type,
        use_tim = scenario.use_tim,
        epoch_adam = scenario_epochs_adam(scenario, context.campaign),
        epoch_linesearch = scenario_epochs_linesearch(scenario, context.campaign),
        sparsity_H = scenario.sparsity_H,
        sparsity_V_sigma = scenario.sparsity_V_sigma,
        sparsity_V_threshold = scenario.sparsity_V_threshold,
        overwrite = (scenario_idx == 1),
    )

    plot_losses(
        inversion = inversion,
        n_epochs_adam = scenario_epochs_adam(scenario, context.campaign),
        scenario_id = scenario_idx,
        results_dir = context.results_dir,
        loss_type = scenario.loss_type,
        use_tim = scenario.use_tim,
        sparsity_H = scenario.sparsity_H,
        sparsity_V_sigma = scenario.sparsity_V_sigma,
        sparsity_V_threshold = scenario.sparsity_V_threshold,
        log_scale = false,
    )

    for glacier_result in inversion.results.simulation
        restore_unmasked_references!(glacier_result, unmasked_references)
        push!(context.scenario_results_by_glacier[glacier_result.rgi_id], glacier_result)
    end

    push!(context.scenario_labels, scenario_label(scenario_idx, scenario))
    return nothing
end

"""
    run_campaign!(context::CampaignRunContext) -> Nothing

Execute all scenarios in a campaign and generate comparison outputs.

Steps:
1. Loop through all scenarios, calling `run_scenario!` for each
2. Save thickness and velocity comparison grids (all scenarios, per glacier)
3. Print summary table (metrics aggregated across glaciers)
4. Generate boxplot of relative errors
"""
function run_campaign!(context::CampaignRunContext)
    for (i, scenario) in enumerate(context.scenarios)
        println(scenario_banner(i, length(context.scenarios), scenario))
        run_scenario!(context, scenario, i)
    end

    save_comparison_grids!(;
        rgi_ids = context.campaign.rgi_ids,
        scenario_results_by_glacier = context.scenario_results_by_glacier,
        scenario_labels = context.scenario_labels,
        results_dir = context.results_dir,
    )

    print_summary_table(context.csv_file)

    # TODO: generate target-aware plots (A/C) and alternative field metrics for distributed inversion.
    plot_relative_error_boxplot(
        csv_path = context.csv_file,
        output_path = joinpath(context.results_dir, "temperate_ice_absolute_relative_error_log_boxplot.png"),
        log_scale = true,
        use_absolute_error = true,
    )

    return nothing
end
