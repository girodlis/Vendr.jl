using Revise
#cd(@__DIR__)

using Vendr
using ODINN
using Sleipnir

# ##############################################
# ###########       CONFIG     ##############
# ##############################################

campaign_root = dirname(abspath(@__FILE__))

# 1) load campaign/scenarios from config
campaign = load_campaign_config(campaign_root)
validate_campaign(campaign)

scenarios = load_scenarios(campaign_root, campaign)

results_dir = campaign.results_dir
isdir(results_dir) || mkpath(results_dir)
csv_file = joinpath(results_dir, "campaign_summary.csv")

rgi_paths = get_rgi_paths()

scenario_results_by_glacier = Dict(rgi_id => Sleipnir.Results[] for rgi_id in campaign.rgi_ids)
scenario_labels = String[]

# ##############################################
# ###########       SCENARIOS     ##############
# ##############################################

for (i, scenario) in enumerate(scenarios)
    println("\n" * scenario_banner(i, length(scenarios), scenario))

    # 2) params definition for inversion
    #params = build_parameters(campaign, scenario)
    params = Parameters(
        simulation = SimulationParameters(
            use_MB = scenario.use_tim,
            tspan = campaign.tspan,
            test_mode = false,
            multiprocessing = true,
            workers = 4,
            use_glathida_data = true,
            #TODO: issue to get grid scaled velocity data
            #gridScalingFactor = 4,
            rgi_paths = rgi_paths,
        ),
        hyper = Hyperparameters(
            batch_size = length(scenario.rgi_ids),
            epochs = [campaign.epochs_adam, campaign.epochs_linesearch],
            optimizer = [
                ODINN.Adam(0.01),
                ODINN.LBFGS(linesearch = ODINN.LineSearches.BackTracking(iterations = 5)),
            ],
        ),
        physical = PhysicalParameters(
            minA = 8e-19,
            maxA = 8e-15,
        ),
        UDE = resolve_autoAD(scenario),
        solver = Huginn.SolverParameters(
            step = campaign.step, # inversion steps
        ),
    )

    # 3) si spinup -> spinup
    if campaign.spinup
        println("Spinup enabled (applied in synthetic GT generation).")
    else
        println("Spinup disabled.")
    end

    # 4) synthetic -> generate GT ; observed -> recover data
    prediction = build_ground_truth(campaign, scenario, params)
    glaciers = prediction.glaciers

    # 5) apply masks if requested
    glaciers, unmasked_references = Vendr.apply_masks(scenario, glaciers)

    # 6) prepare inversion model
    inv_model = build_inversion_model(campaign, scenario, params, glaciers)

    # 7) run inversion
    inversion = Inversion(inv_model, glaciers, params)
    run!(inversion)

    # 8) save scenario outputs
    save_results_csv(
        csv_file,
        inversion,
        prediction;
        scenario_id = scenario.id,
        rgi_ids = scenario.rgi_ids,
        target = campaign.invert_target,
        loss_type = scenario.loss_type,
        use_tim = scenario.use_tim,
        epoch_adam = campaign.epochs_adam,
        epoch_linesearch = campaign.epochs_linesearch,
        sparsity_H = scenario.sparsity_H,
        sparsity_V_sigma = scenario.sparsity_V_sigma,
        sparsity_V_threshold = scenario.sparsity_V_threshold,
        overwrite = (i == 1),
    )

    should_restore_references = !isnothing(unmasked_references)
    for glacier_result in inversion.results.simulation
        if should_restore_references
            restore_unmasked_references!(glacier_result, unmasked_references)
        end
        push!(context.scenario_results_by_glacier[glacier_result.rgi_id], glacier_result)
    end

    push!(context.scenario_labels, scenario_label(scenario_idx, scenario))
end

# ##############################################
# ###########       ANALYSIS     ##############
# ##############################################

print_summary(csv_file)

save_comparison_grids!(;
        rgi_ids = context.campaign.rgi_ids,
        scenario_results_by_glacier = context.scenario_results_by_glacier,
        scenario_labels = context.scenario_labels,
        results_dir = context.results_dir,
    )