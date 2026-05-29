"""
    plotting/loss.jl

Loss plotting utilities. Kept separate from campaign summary/grid plots so this part
can be migrated more easily in the future.
"""

"""
    plot_loss_evolution(inversion, n_epochs_adam;
        scenario_id,
        results_dir=".",
        loss_type=nothing,
        use_tim=nothing,
        sparsity_H=nothing,
        sparsity_V=nothing,
    )

Plot the loss evolution and save one PNG per scenario.
The output file is written in `results_dir` with a scenario-specific name.
"""
function plot_loss_evolution(
    inversion::Inversion;
        n_epochs_adam::Int = 2,
    scenario_id::Union{Int, AbstractString} = 1,
    scenario_label::Union{Nothing, AbstractString} = nothing,
        results_dir::String = "outputs/results",
        loss_type::String = "H",
        use_tim::Bool = false,
        sparsity_H::Bool = false,
        sparsity_V_sigma::Float64 = 0.7,
        sparsity_V_threshold::Float64 = 0.001,
        log_scale::Bool = false,
    )
    losses = inversion.results.stats.losses
    epochs_total = 1:length(losses)
    n_epochs_linesearch = max(length(epochs_total) - n_epochs_adam, 0)

    if log_scale && any(<=(0), losses)
        error("plot_losses(log_scale=true) requires strictly positive losses. Found values <= 0.")
    end

    scenario_text = scenario_label === nothing ? string(scenario_id) : String(scenario_label)
    plot_title = log_scale ? "Loss Evolution (log10) - $(scenario_text)" : "Loss Evolution - $(scenario_text)"

    p = Plots.plot(
        size = (1200, 750),
        dpi = 300,
        margin = 10Plots.mm,
        legend = :topright,
        legendfontsize = 11,
        palette = :Set1_5,
        bottom_margin = 20Plots.mm,
        left_margin = 14Plots.mm,
        top_margin = 15Plots.mm,
        right_margin = 10Plots.mm,
        yscale = log_scale ? :log10 : :identity,
        grid = true,
        gridalpha = 0.3,
        gridstyle = :dash,
        gridlinewidth = 0.5,
    )

    if log_scale
        min_positive_loss = minimum(losses)
        max_loss = maximum(losses)
        Plots.ylims!(p, min_positive_loss * 0.8, max_loss * 1.2)
    end

    Plots.plot!(
        p,
        epochs_total,
        losses,
        xlabel = "Epoch",
        ylabel = log_scale ? "Loss (log10)" : "Loss",
        title = plot_title,
        label = "Overall Loss",
        color = :black,
        lw = 1.5,
        alpha = 0.6,
        linestyle = :solid,
        guidefontsize = 12,
        titlefontsize = 14,
        xtickfontsize = 11,
        ytickfontsize = 11,
        framestyle = :box,
    )

    if n_epochs_adam > 0
        adam_epochs = epochs_total[1:min(n_epochs_adam + 1, length(epochs_total))]
        adam_losses = losses[1:min(n_epochs_adam + 1, length(losses))]
        Plots.plot!(
            p,
            adam_epochs,
            adam_losses,
            label = "Adam optimizer",
            color = :steelblue,
            lw = 2.5,
            alpha = 0.8,
            fillalpha = log_scale ? 0.0 : 0.2,
            fill = log_scale ? nothing : (0, :steelblue),
        )
    end

    if n_epochs_adam < length(epochs_total)
        ls_start = max(n_epochs_adam + 1, 1)
        ls_epochs = epochs_total[ls_start:end]
        ls_losses = losses[ls_start:end]
        Plots.plot!(
            p,
            ls_epochs,
            ls_losses,
            label = "Line search",
            color = :darkorange,
            lw = 2.5,
            alpha = 0.8,
            fillalpha = log_scale ? 0.0 : 0.2,
            fill = log_scale ? nothing : (0, :darkorange),
        )
    end

    Plots.vline!(
        p,
        [n_epochs_adam + 1],
        label = "Algorithm switch",
        linestyle = :dash,
        color = :crimson,
        lw = 3,
        alpha = 0.95,
    )

    y_top = maximum(losses)
    y_bottom = minimum(losses)
    x_start = first(epochs_total)
    x_end = last(epochs_total)
    x_range = x_end - x_start

    if log_scale
        log_y_bottom = log10(y_bottom)
        log_y_top = log10(y_top)
        y_annotation = 10.0 ^ (log_y_bottom + 0.88 * (log_y_top - log_y_bottom))
    else
        y_range = y_top - y_bottom
        y_annotation = y_bottom + 0.88 * y_range
    end

    x_annotation = x_start + 0.02 * x_range

    params_line1 = "loss=$(loss_type) | tim=$(use_tim)"
    params_line2 = "H=$(sparsity_H) | V=$(sparsity_V_sigma), $(sparsity_V_threshold)"
    params_line3 = "Adam=$(n_epochs_adam) | LS=$(n_epochs_linesearch)"
    params_text = "$(params_line1)\n$(params_line2)\n$(params_line3)"

    Plots.annotate!(
        p,
        x_annotation,
        y_annotation,
        Plots.text(params_text, 8, :left, :black, "Arial"),
    )

    suffix = _build_loss_suffix(loss_type, use_tim, sparsity_H, sparsity_V_sigma, sparsity_V_threshold)
    if log_scale
        suffix *= "_log"
    end

    isdir(results_dir) || mkpath(results_dir)
    output_path = joinpath(results_dir, "loss_evolution_scenario$(scenario_id)$(suffix).png")
    Plots.savefig(p, output_path)
    return output_path
end


"""
    plot_loss_evolution(scenario_inversions, scenarios;
        results_dir="outputs/results",
        n_epochs_adam=nothing,
        scenario_labels=nothing,
        log_scale=false,
    )

Plot the loss evolution for every scenario in a campaign/run.
This is the preferred entry point when you already have `context.scenario_inversions`
and `context.scenarios` available after a run.
"""
function plot_loss_evolution(
        scenario_inversions::AbstractVector,
        scenarios::AbstractVector;
        results_dir::AbstractString = "outputs/results",
        n_epochs_adam::Union{Int, Nothing} = nothing,
        scenario_labels::Union{Nothing, AbstractVector{<:AbstractString}} = nothing,
        log_scale::Bool = false,
    )
    length(scenario_inversions) == length(scenarios) || error("scenario_inversions and scenarios must have the same length")

    output_paths = String[]
    for (i, (inversion, scenario)) in enumerate(zip(scenario_inversions, scenarios))
        label = scenario_labels === nothing ? scenario_label(i, scenario) : String(scenario_labels[i])
        push!(output_paths, plot_loss_evolution(
            inversion;
            scenario_id = scenario.id,
            scenario_label = label,
            results_dir = results_dir,
            n_epochs_adam = n_epochs_adam,
            log_scale = log_scale,
        ))
    end

    return output_paths
end


"""
    plot_loss_evolution(context::CampaignRunContext; ...)

Convenience wrapper for a completed run context.
"""
function plot_loss_evolution(
        context::CampaignRunContext;
        results_dir::AbstractString = context.results_dir,
        n_epochs_adam::Union{Int, Nothing} = context.campaign.epochs_adam,
        scenario_labels::Union{Nothing, AbstractVector{<:AbstractString}} = nothing,
        log_scale::Bool = false,
    )
    return plot_loss_evolution(
        context.scenario_inversions,
        context.scenarios;
        results_dir = results_dir,
        n_epochs_adam = n_epochs_adam,
        scenario_labels = scenario_labels,
        log_scale = log_scale,
    )
end



function _build_loss_suffix(loss_type, use_tim, sparsity_H, sparsity_V_sigma, sparsity_V_threshold)
    suffix = ""

    if loss_type isa String && !isempty(loss_type)
        suffix *= "_loss$(loss_type)"
    end

    if use_tim isa Bool
        suffix *= "_tim$(use_tim)"
    end

    if sparsity_H isa Bool
        suffix *= "_H$(sparsity_H)"
    end

    if sparsity_V_sigma isa Number && sparsity_V_sigma != 0.0
        suffix *= "_V_sigma$(sparsity_V_sigma)"
    end

    if sparsity_V_threshold isa Number && sparsity_V_threshold != 0.0
        suffix *= "_V_threshold$(sparsity_V_threshold)"
    end

    return suffix
end
