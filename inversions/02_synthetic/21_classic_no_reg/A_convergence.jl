using CairoMakie
using Printf

# Run in the same REPL as run.jl (needs `context`, already populated by run_campaign!).
# Convergence of the inverted creep parameter A over iterations, for three
# representative scenarios (same glacier as the showcase figure, for consistency):
#   S1  = loss on H only, dense observations, no tim
#   S2  = loss on V only, dense observations, no tim
#   S12 = loss on H+V,   H masked, V sparse, no tim

glacier_id = "RGI60-11.01450"

# (scenario_id, legend label, color) — Okabe-Ito colorblind-safe palette
curves = [
    ("S1",  L"L_H(\theta)",   colorant"#E69F00"),
    ("S2",  L"L_V(\theta)",   colorant"#56B4E9"),
    ("S12", L"L_{HV}(\theta)", colorant"#009E73"),
]

function _scenario_glacier(scenario_id::String, glacier_id::String)
    si = findfirst(s -> s.id == scenario_id, context.scenarios)
    isnothing(si) && error("Scenario $(scenario_id) not found in context.scenarios")
    scenario = context.scenarios[si]
    inversion = context.scenario_inversions[si]
    prediction = context.scenario_predictions[si]
    gi = findfirst(==(glacier_id), scenario.rgi_ids)
    isnothing(gi) && error("Glacier $(glacier_id) not found in scenario $(scenario_id)")
    return scenario, inversion, prediction, gi
end

_A_scalar(sim, gi; θ = nothing) = only(Vendr._target_value(sim, gi, :A; θ = θ))

const _SUPERSCRIPT = Dict('0'=>'⁰','1'=>'¹','2'=>'²','3'=>'³','4'=>'⁴','5'=>'⁵','6'=>'⁶','7'=>'⁷','8'=>'⁸','9'=>'⁹','-'=>'⁻')
_superscript(n::Integer) = join(_SUPERSCRIPT[c] for c in string(n))

function build_A_convergence(glacier_id::String, curves)
    # Gather all curves + ground truth first, so we can compute a common power-of-ten
    # scale factor and explicit y-tick positions from the actual data range: on a
    # linear axis with tiny absolute values (~1e-18 to ~3e-16), Makie's automatic tick
    # locator can fail to find any "nice" round numbers and silently produce zero
    # labelled ticks.
    histories = NamedTuple[]
    gt_value = nothing
    for (sid, label, color) in curves
        _, inversion, prediction, gi = _scenario_glacier(sid, glacier_id)
        A_hist = [_A_scalar(inversion, gi; θ = θ) for θ in inversion.results.stats.θ_hist]
        push!(histories, (label = label, color = color, A_hist = A_hist))
        gt_value === nothing && (gt_value = _A_scalar(prediction, gi; θ = nothing))
    end

    all_values = vcat((h.A_hist for h in histories)..., [gt_value])
    exponent = floor(Int, log10(maximum(abs.(all_values))))
    scale = exp10(exponent)
    factor(v) = v / scale

    y_ticks_pos = range(minimum(all_values), maximum(all_values); length = 6)
    y_ticks_labels = [Printf.@sprintf("%.2f", factor(v)) for v in y_ticks_pos]

    gt_legend_label = "Ground truth (A = $(Printf.@sprintf("%.2e", gt_value)) Pa⁻³ yr⁻¹)"

    fontsize_title = 40
    fontsize_label = 34
    fontsize_ticks = 30
    fontsize_legend = 28

    fig = Figure(size = (1600, 1100), fontsize = fontsize_ticks)
    ax = Axis(
        fig[1, 1],
        xlabel = "Iteration",
        ylabel = "A (×10$(_superscript(exponent)) Pa⁻³ yr⁻¹)",
        yticks = (y_ticks_pos, y_ticks_labels),
        title = "Convergence of the inverted creep parameter A",
        titlesize = fontsize_title,
        xlabelsize = fontsize_label,
        ylabelsize = fontsize_label,
        xticklabelsize = fontsize_ticks,
        yticklabelsize = fontsize_ticks,
    )

    for h in histories
        lines!(ax, 1:length(h.A_hist), h.A_hist; label = h.label, color = h.color, linewidth = 5)
    end

    hlines!(ax, [gt_value]; color = :black, linestyle = :dash, linewidth = 3.5, label = gt_legend_label)
    axislegend(ax, position = :rb, framevisible = false, labelsize = fontsize_legend)

    resize_to_layout!(fig)
    outpath = joinpath(context.results_dir, "A_convergence_$(glacier_id).png")
    CairoMakie.save(outpath, fig; px_per_unit = 3)
    println("Saved: ", outpath)
    return outpath
end

build_A_convergence(glacier_id, curves)
