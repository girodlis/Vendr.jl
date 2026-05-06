"""
    run_scenario!(context::CampaignRunContext, scenario::ScenarioConfig, scenario_idx::Int) -> Nothing

Run one inversion scenario and append results to campaign context.

Prepares models, runs the inversion, saves results, and stores inversion/prediction
in the campaign context for later analysis.

# Arguments
- `context::CampaignRunContext`: Campaign execution context
- `scenario::ScenarioConfig`: Scenario configuration
- `scenario_idx::Int`: Scenario index (1-based, for logging)

# Side Effects
- Appends rows to context.csv_file (first scenario overwrites)
- Stores inversion and prediction in context
"""
function run_scenario!(context::CampaignRunContext, scenario::ScenarioConfig, scenario_idx::Int)::Nothing
    inv_model, glaciers, params, unmasked_references, prediction = prepare_scenario(context.campaign, scenario)
    inversion = Inversion(inv_model, glaciers, params)
    run!(inversion)

    save_results_csv(
        context.csv_file,
        inversion,
        prediction;
        scenario_id = scenario.id,
        rgi_ids = scenario.rgi_ids,
        target = context.campaign.invert_target,
        loss_type = scenario.loss_type,
        use_tim = scenario.use_tim,
        epoch_adam = context.campaign.epochs_adam,
        epoch_linesearch = context.campaign.epochs_linesearch,
        sparsity_H = scenario.sparsity_H,
        sparsity_V_sigma = scenario.sparsity_V_sigma,
        sparsity_V_threshold = scenario.sparsity_V_threshold,
        overwrite = (scenario_idx == 1),
    )

    should_restore_references = !isnothing(unmasked_references)
    for glacier_result in inversion.results.simulation
        if should_restore_references
            restore_unmasked_references!(glacier_result, unmasked_references)
        end
    end

    push!(context.scenario_inversions, inversion)
    push!(context.scenario_predictions, prediction)
    return nothing
end

"""
    run_campaign!(context::CampaignRunContext) -> CampaignRunContext

Execute all scenarios in a campaign and generate outputs.

Iterates through all scenarios, runs inversions, and aggregates results
into comparison plots and CSV summary.

# Arguments
- `context::CampaignRunContext`: Fully initialized campaign context

# Returns
Context with populated results dictionaries

# Side Effects
- Creates CSV file with results
- Generates comparison plot files in results_dir
"""
function run_campaign!(context::CampaignRunContext)::CampaignRunContext
    for (i, scenario) in enumerate(context.scenarios)
        println(scenario_banner(i, length(context.scenarios), scenario))
        run_scenario!(context, scenario, i)
    end

    save_comparison_grids!(;
        rgi_ids = context.campaign.rgi_ids,
        results_dir = context.results_dir,
        scenarios = context.scenarios,
        scenario_inversions = context.scenario_inversions,
        scenario_predictions = context.scenario_predictions,
    )
    
    return context
end