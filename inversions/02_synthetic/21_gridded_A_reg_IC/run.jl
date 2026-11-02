using Revise
cd(@__DIR__)

using Vendr
using ODINN
using CairoMakie

# Campaign root is the current directory (path of this script)
campaign_root = dirname(abspath(@__FILE__))

# Load campaign config and scenarios from campaign_root/config/
# campaign.toml already configures: gridded A (law_A = CuffeyPatersonGridded),
# regularization on A (scenarios.toml: regularization_weight), and no spinup
# (simulation.spinup = false).
context = build_campaign_run_context(campaign_root)

# ==================================================================
# Inversion jointe de A (gridded) + conditions initiales (H₀)
# ==================================================================
# `build_inversion_model`/`run_campaign!` ne savent inverser que A : on
# construit donc ici, scénario par scénario, un Model étendu avec un
# régresseur supplémentaire `IC` (ODINN.InitialCondition), en réutilisant
# les briques existantes de Vendr pour tout le reste (params, ground
# truth, régresseur A gridded, sauvegarde des résultats, plots).

# A priori pour H₀ : Farinotti19 + bruit gaussien corrélé spatialement.
# Le H₀ de vérité terrain vient déjà de Farinotti19 (source par défaut),
# donc partir de :Farinotti19 tel quel donnerait un a priori identique à
# la vérité (test non pertinent) ; :Farinotti19Random perturbe ce champ
# pour vérifier que l'inversion peut effectivement le retrouver.
ic_initialization = :Farinotti19Random

# Poids de régularisation (Tikhonov) sur H₀, en plus de la régularisation
# sur A déjà pilotée par scenario.regularization_weight.
λ_IC = 1e-4

function build_ic_loss(campaign::CampaignConfig, scenario::ScenarioConfig, λ_IC::Float64)
    h_loss = LossH(loss = L2Sum(distance = 0))
    empirical = if scenario.loss_type == "H"
        h_loss
    elseif scenario.loss_type == "V"
        LossV()
    elseif scenario.loss_type == "HV"
        LossHV(hLoss = h_loss, vLoss = LossV())
    else
        error("Unsupported loss_type=$(scenario.loss_type).")
    end

    losses = Any[empirical]
    λs = Float64[1.0]

    if scenario.regularization_weight > 0.0
        push!(losses, RheologyRegularization())
        push!(λs, scenario.regularization_weight)
    end

    push!(losses, InitialThicknessRegularization(campaign.tspan[1]))
    push!(λs, λ_IC)

    return MultiLoss(losses = Tuple(losses), λs = Tuple(λs))
end

function build_ic_UDE_params(campaign::CampaignConfig, scenario::ScenarioConfig, λ_IC::Float64)
    loss = build_ic_loss(campaign, scenario, λ_IC)
    return scenario.use_optim_autoAD ?
        UDEparameters(
            optim_autoAD = ODINN.Optimization.AutoZygote(),
            grad = SciMLSensitivityAdjoint(),
            empirical_loss_function = loss,
            initial_condition_filter = :Zang1980,
        ) :
        UDEparameters(
            optim_autoAD = ODINN.NoAD(),
            grad = DiscreteAdjoint(),
            empirical_loss_function = loss,
            initial_condition_filter = :Zang1980,
        )
end

function run_scenario_with_ic!(
        context::CampaignRunContext,
        scenario::ScenarioConfig,
        scenario_idx::Int,
        ic_initialization::Symbol,
        λ_IC::Float64,
)::Nothing
    campaign = context.campaign
    base_params = build_parameters(campaign, scenario)
    params = Parameters(
        physical = base_params.physical,
        simulation = base_params.simulation,
        hyper = base_params.hyper,
        solver = base_params.solver,
        UDE = build_ic_UDE_params(campaign, scenario, λ_IC),
    )

    prediction = build_ground_truth(campaign, scenario, params)
    glaciers = prediction.glaciers
    glaciers, unmasked_references = Vendr.apply_masks(campaign, scenario, glaciers)

    base_model = build_inversion_model(campaign, scenario, params, glaciers)
    trainable_A = base_model.trainable_components.A
    ic = InitialCondition(params, glaciers, ic_initialization)
    model = Model(
        iceflow = base_model.iceflow,
        mass_balance = base_model.mass_balance,
        regressors = (A = trainable_A, IC = ic),
    )

    inversion = Inversion(model, glaciers, params)
    run!(inversion)

    save_results_csv(
        context.csv_file,
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

for (i, scenario) in enumerate(context.scenarios)
    println(scenario_banner(i, length(context.scenarios), scenario))
    run_scenario_with_ic!(context, scenario, i, ic_initialization, λ_IC)
end

save_epoch_counts_csv(context)

save_comparison_grids!(
    rgi_ids = context.campaign.rgi_ids,
    results_dir = context.results_dir,
    scenarios = context.scenarios,
    scenario_inversions = context.scenario_inversions,
    scenario_predictions = context.scenario_predictions,
)

print_summary(context.csv_file)

plot_relative_error_boxplot(
    csv_path = context.csv_file,
    output_path = joinpath(context.results_dir, "temperate_ice_absolute_relative_error_log_boxplot.png"),
    log_scale = true,
    use_absolute_error = true,
)

# ============ Plots Ground Truth + All Scenarios (A) ============
# Labels adaptés pour mettre en avant la régularisation (A et H₀)
scenario_labels = [
    if scenario.regularization_weight > 0.0
        "$(scenario_label(1, scenario)) | λ_A=$(scenario.regularization_weight) | λ_IC=$(λ_IC)"
    else
        "$(scenario_label(1, scenario)) | λ_IC=$(λ_IC)"
    end
    for (i, scenario) in enumerate(context.scenarios)
]

for (glacier_idx, rgi_id) in enumerate(context.campaign.rgi_ids)
    fig = plot_all_scenarios_with_ground_truth(
        context.scenario_inversions,
        context.scenario_predictions,
        glacier_idx,
        :A,
        scenario_labels;
        plotContour = true,
        colormap = :YlGnBu
    )
    save(joinpath(context.results_dir, "all_scenarios_A_$(rgi_id).png"), fig)
    println("Saved: all_scenarios_A_$(rgi_id).png")
end

# ============ Plots: conditions initiales H₀ (a priori vs vérité vs inversé) ============
for (scenario_idx, scenario) in enumerate(context.scenarios)
    inversion = context.scenario_inversions[scenario_idx]
    prediction = context.scenario_predictions[scenario_idx]

    for (glacier_idx, rgi_id) in enumerate(scenario.rgi_ids)
        glacier = prediction.glaciers[glacier_idx]
        H0_gt = glacier.H₀
        H0_inv = inversion.results.stats.initial_conditions[rgi_id]
        mask = H0_gt .> 0.0

        H0_gt_masked = ifelse.(mask, H0_gt, NaN)
        H0_inv_masked = ifelse.(mask, H0_inv, NaN)
        H0_diff = H0_inv_masked .- H0_gt_masked

        hmax = max(maximum(filter(isfinite, H0_gt_masked)), maximum(filter(isfinite, H0_inv_masked)))
        dmax = maximum(abs.(filter(isfinite, H0_diff)))

        x = glacier.Coords["lon"]
        y = glacier.Coords["lat"]

        fig = Figure(size = (1300, 400))

        ax1 = Axis(fig[1, 1], aspect = DataAspect(), title = "H₀ vérité terrain")
        hm1 = heatmap!(ax1, Sleipnir.reverseForHeatmap(H0_gt_masked, x, y),
            colormap = :Blues, colorrange = (0, hmax))
        Colorbar(fig[1, 2], hm1, label = "m")

        ax2 = Axis(fig[1, 3], aspect = DataAspect(), title = "H₀ inversé")
        hm2 = heatmap!(ax2, Sleipnir.reverseForHeatmap(H0_inv_masked, x, y),
            colormap = :Blues, colorrange = (0, hmax))
        Colorbar(fig[1, 4], hm2, label = "m")

        ax3 = Axis(fig[1, 5], aspect = DataAspect(), title = "diff H₀ (inversé − vérité)")
        hm3 = heatmap!(ax3, Sleipnir.reverseForHeatmap(H0_diff, x, y),
            colormap = :RdBu, colorrange = (-dmax, dmax))
        Colorbar(fig[1, 6], hm3, label = "m")

        fig[0, :] = Label(fig, "$(rgi_id) — scénario $(scenario.id)", fontsize = 14, font = :bold)

        save(joinpath(context.results_dir, "ic_comparison_$(scenario.id)_$(rgi_id).png"), fig)
        println("Saved: ic_comparison_$(scenario.id)_$(rgi_id).png")
    end
end
