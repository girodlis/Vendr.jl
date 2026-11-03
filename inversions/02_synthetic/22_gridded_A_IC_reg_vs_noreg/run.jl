using Revise
cd(@__DIR__)

using Vendr
using ODINN
using CairoMakie
using Statistics

campaign_root = dirname(abspath(@__FILE__))

context = build_campaign_run_context(campaign_root)

# ==================================================================
# Joint inversion of A (gridded) + initial conditions (H₀)
# ==================================================================
ic_initialization = :Farinotti19Random
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
    model = ODINN.Model(
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

# ============ Plot: A ground truth | A without reg. | A with reg. ============
function distributed_glacier_mask(glacier, target_size::Tuple{Int, Int}; distance::Int = 1)
    h0 = glacier.H₀
    size(h0) == target_size && return Sleipnir.is_in_glacier(h0, distance)
    h0_interior = h0[1:(end - 1), 1:(end - 1)]
    size(h0_interior) == target_size && return Sleipnir.is_in_glacier(h0_interior, distance)
    return trues(target_size)
end

function median_outlier_filter(field::Matrix{Float64}; factor::Float64 = 6.0)
    valid = filter(isfinite, vec(field))
    isempty(valid) && return field, 0, NaN
    cutoff = factor * median(valid)
    is_outlier = isfinite.(field) .& (field .> cutoff)
    return ifelse.(is_outlier, NaN, field), count(is_outlier), cutoff
end

noreg_idx = findfirst(s -> s.regularization_weight == 0.0, context.scenarios)
reg_idx = findfirst(s -> s.regularization_weight > 0.0, context.scenarios)
isnothing(noreg_idx) && error("No scenario without regularization (regularization_weight == 0.0) found.")
isnothing(reg_idx) && error("No scenario with regularization (regularization_weight > 0.0) found.")

for (glacier_idx, rgi_id) in enumerate(context.campaign.rgi_ids)
    glacier = context.scenario_predictions[noreg_idx].glaciers[glacier_idx]
    x = glacier.Coords["lon"]
    y = glacier.Coords["lat"]

    gt_A = Vendr._target_value(context.scenario_predictions[noreg_idx], glacier_idx, :A; θ = nothing)
    gt_A = gt_A isa Matrix ? gt_A : Array(gt_A)

    A_noreg = Vendr._target_value(
        context.scenario_inversions[noreg_idx], glacier_idx, :A;
        θ = context.scenario_inversions[noreg_idx].results.stats.θ,
    )
    A_reg = Vendr._target_value(
        context.scenario_inversions[reg_idx], glacier_idx, :A;
        θ = context.scenario_inversions[reg_idx].results.stats.θ,
    )
    A_noreg = A_noreg isa Matrix ? A_noreg : Array(A_noreg)
    A_reg = A_reg isa Matrix ? A_reg : Array(A_reg)

    mask = distributed_glacier_mask(glacier, size(gt_A))

    A_UNIT = 1e-17
    gt_A_masked = ifelse.(mask, gt_A ./ A_UNIT, NaN)
    A_noreg_masked = ifelse.(mask, A_noreg ./ A_UNIT, NaN)
    A_reg_masked = ifelse.(mask, A_reg ./ A_UNIT, NaN)

    gt_A_masked, n_gt, cutoff_gt = median_outlier_filter(gt_A_masked)
    A_noreg_masked, n_noreg, cutoff_noreg = median_outlier_filter(A_noreg_masked)
    A_reg_masked, n_reg, cutoff_reg = median_outlier_filter(A_reg_masked)
    if n_gt + n_noreg + n_reg > 0
        println(
            "Excluding outlier pixel(s) from A plot ($(rgi_id)) [×10⁻¹⁷ Pa⁻³ yr⁻¹]: " *
            "gt=$(n_gt) (cutoff $(round(cutoff_gt, sigdigits=3))), " *
            "no-reg=$(n_noreg) (cutoff $(round(cutoff_noreg, sigdigits=3))), " *
            "reg=$(n_reg) (cutoff $(round(cutoff_reg, sigdigits=3))).",
        )
    end

    positive_values = Float64[]
    for field in (gt_A_masked, A_noreg_masked, A_reg_masked)
        append!(positive_values, vec(field[isfinite.(field) .& (field .> 0)]))
    end
    A_vmin, A_vmax = minimum(positive_values), maximum(positive_values)

    nx, ny = size(mask)
    ctr = Sleipnir.Contour.contour(collect(1:nx), 1 + ny .- collect(1:ny), mask, 0.5)

    panels = [
        ("A ground truth", gt_A_masked),
        ("A inverted, without\nregularization", A_noreg_masked),
        ("A inverted, with\nregularization", A_reg_masked),
    ]

    fig = Figure(size = (1500, 550))
    heatmaps = []
    for (i, (title, field)) in enumerate(panels)
        # Title as a manually line-broken Label spanning the column (row 1)
        # rather than Axis(title=...): with narrow, aspect-locked columns a
        # single long line would overflow into the neighboring panel.
        # NB: word_wrap=true must NOT be used here — its bounding-box
        # resolution depends on the resolved column width, which is itself
        # driven by the Aspect-locked colsize! below and the Axis's
        # DataAspect(); combining the two creates a resize feedback loop
        # that overflows the stack in Makie 0.24 (ComputePipeline backend).
        Label(fig[1, i], title; fontsize = 22, font = :bold, tellwidth = false)
        ax = Axis(fig[2, i], aspect = DataAspect(),
            xlabelsize = 20, ylabelsize = 20, xticklabelsize = 17, yticklabelsize = 17)
        hm = heatmap!(ax, Sleipnir.reverseForHeatmap(field, x, y),
            colormap = :YlGnBu, colorrange = (A_vmin, A_vmax))
        for curve in ctr.lines
            lines!(ax, first.(curve.vertices), last.(curve.vertices), color = :black, linewidth = 1.5, alpha = 0.5)
        end
        ax.xlabel = "Lon"
        ax.ylabel = i == 1 ? "Lat" : ""
        i > 1 && hideydecorations!(ax, grid = false)
        push!(heatmaps, hm)
    end
    Colorbar(fig[2, length(panels) + 1], heatmaps[1], label = "A (×10⁻¹⁷ Pa⁻³ yr⁻¹)",
        labelsize = 20, ticklabelsize = 17)

    for i in 1:length(panels)
        colsize!(fig.layout, i, Aspect(2, nx / ny))
    end
    resize_to_layout!(fig)

    save(joinpath(context.results_dir, "A_reg_vs_noreg_$(rgi_id).png"), fig)
    println("Saved: A_reg_vs_noreg_$(rgi_id).png")
end

# ============ Plots: initial conditions H₀ (ground truth vs inverted) ============
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

        ax1 = Axis(fig[1, 1], aspect = DataAspect(), title = "H₀ ground truth")
        hm1 = heatmap!(ax1, Sleipnir.reverseForHeatmap(H0_gt_masked, x, y),
            colormap = :Blues, colorrange = (0, hmax))
        Colorbar(fig[1, 2], hm1, label = "m")

        ax2 = Axis(fig[1, 3], aspect = DataAspect(), title = "H₀ inverted")
        hm2 = heatmap!(ax2, Sleipnir.reverseForHeatmap(H0_inv_masked, x, y),
            colormap = :Blues, colorrange = (0, hmax))
        Colorbar(fig[1, 4], hm2, label = "m")

        ax3 = Axis(fig[1, 5], aspect = DataAspect(), title = "diff H₀ (inverted − ground truth)")
        hm3 = heatmap!(ax3, Sleipnir.reverseForHeatmap(H0_diff, x, y),
            colormap = :RdBu, colorrange = (-dmax, dmax))
        Colorbar(fig[1, 6], hm3, label = "m")

        fig[0, :] = Label(fig, "$(rgi_id) — scenario $(scenario.id)", fontsize = 14, font = :bold)

        save(joinpath(context.results_dir, "ic_comparison_$(scenario.id)_$(rgi_id).png"), fig)
        println("Saved: ic_comparison_$(scenario.id)_$(rgi_id).png")
    end
end
