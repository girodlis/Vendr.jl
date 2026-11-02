using CairoMakie

# Run in the same REPL as run.jl (needs `context`, already populated by run_campaign!).
# Produces one "showcase" figure for a paper: predicted / ground truth / relative
# error, for a hand-picked glacier and a couple of hand-picked scenarios that tell a
# clean story (clean H inversion vs. a realistic, harder HV/sparse case that still
# holds up). Panels are tagged (a), (b), (c)... only; the scenario/field/error
# description for each letter is printed at the end for use in the LaTeX caption.
#
# Color scales are shared across rows for each quantity (thickness, velocity,
# relative error) so that the *magnitude* of the error is visually comparable
# between scenarios — a per-row auto-scaled colorbar would make every scenario
# look "equally red" even though the real error differs by an order of magnitude.

glacier_id = "RGI60-11.01450"

# Each entry: (scenario_id, field) — field = :H or :V.
# NOTE: rows sharing the same `field` are assumed contiguous (used below to decide
# how to span the shared colorbars); reorder `rows` accordingly if you change this.
rows = [
    ("S1",  :H),   # clean H inversion: near-perfect baseline
    ("S12", :H),   # realistic sparse HV: thickness reconstruction
    ("S12", :V),   # realistic sparse HV: velocity reconstruction
]

output_format = :png  # :png or :pdf (PDF keeps contours/text vector, heatmap rasterized)

function _panel_result(scenario_id::String, glacier_id::String)
    si = findfirst(s -> s.id == scenario_id, context.scenarios)
    isnothing(si) && error("Scenario $(scenario_id) not found in context.scenarios")
    scenario = context.scenarios[si]
    inversion = context.scenario_inversions[si]
    prediction = context.scenario_predictions[si]
    gi = findfirst(==(glacier_id), scenario.rgi_ids)
    isnothing(gi) && error("Glacier $(glacier_id) not found in scenario $(scenario_id)")
    results = inversion.results.simulation[gi]
    rel_err = compute_relative_error(inversion, prediction, gi)
    return scenario, results, rel_err
end

function _row_arrays(results, field::Symbol)
    field_ref = Symbol(field, "_ref")
    pred = Float64.(getproperty(results, field)[end])
    gt   = Float64.(getproperty(results, field_ref)[end])
    mask = Sleipnir.is_in_glacier(Float64.(results.H_ref[end]), 1)
    pred = ifelse.(mask, pred, NaN)
    gt   = ifelse.(mask, gt, NaN)
    relerr = ifelse.(mask .& (gt .!= 0), 100.0 .* (pred .- gt) ./ gt, NaN)
    return pred, gt, relerr, mask
end

function _add_panel_label!(fig, row, col, letter)
    Label(
        fig[row, col, TopLeft()], "($letter)";
        fontsize = 16, font = :bold, padding = (0, 6, 6, 0),
        halign = :left, valign = :top,
    )
    return nothing
end

_finite_max(vs...) = maximum(filter(isfinite, vcat(vs...)); init = 1.0)

# Wrapped in a function: a top-level `for` loop mutating variables declared outside
# it (h_ref_hm, etc.) is ambiguous soft-scope in a script/include'd file. Inside a
# function, scoping is unambiguous, so this sidesteps the issue entirely.
function build_showcase(glacier_id::String, rows; output_format::Symbol = :png)
    # --- Pass 1: gather all row data + compute shared color scales ---
    row_data = NamedTuple[]
    for (sid, field) in rows
        scenario, results, rel_err = _panel_result(sid, glacier_id)
        pred, gt, relerr, mask = _row_arrays(results, field)
        push!(row_data, (
            sid = sid, field = field, scenario = scenario, results = results,
            rel_err = rel_err, pred = pred, gt = gt, relerr = relerr, mask = mask,
        ))
    end

    h_max = _finite_max((vcat(d.pred, d.gt) for d in row_data if d.field == :H)...)
    v_max = _finite_max((vcat(d.pred, d.gt) for d in row_data if d.field == :V)...)
    relerr_lim = _finite_max((abs.(d.relerr) for d in row_data)...)
    relerr_lim > 0 || (relerr_lim = 1.0)

    # --- Pass 2: draw ---
    fig = Figure(size = (1500, 1500))
    n_panels = 3 * length(row_data)
    all_letters = [Char('a' + i - 1) for i in 1:n_panels]
    caption_lines = String[]

    h_ref_hm = nothing
    v_ref_hm = nothing
    err_ref_hm = nothing

    for (row, d) in enumerate(row_data)
        seq_max = d.field == :H ? h_max : v_max
        field_label = d.field == :H ? "Thickness (m)" : "Velocity norm (m/yr)"
        row_letters = all_letters[(3 * (row - 1) + 1):(3 * row)]

        x, y = d.results.x, d.results.y
        nx, ny = size(d.mask)
        ctr = Sleipnir.Contour.contour(collect(1:nx), 1 + ny .- collect(1:ny), d.mask, 0.5)
        # Drop tiny spurious loops (single/few-pixel islands or nunatak noise), keep the real outline
        outline = filter(curve -> length(curve.vertices) > 15, ctr.lines)
        # Thin-ice zone (H < 1 m): relative error blows up here since it divides by a
        # near-zero reference — flag it rather than let it read as a genuine failure.
        thin_mask = d.results.H_ref[end] .< 1.0
        thin_ctr = Sleipnir.Contour.contour(collect(1:nx), 1 + ny .- collect(1:ny), thin_mask, 0.5)

        ax1 = Axis(fig[row, 1], aspect = DataAspect(), title = row == 1 ? "Predicted" : "")
        hm1 = Makie.heatmap!(ax1, Sleipnir.reverseForHeatmap(d.pred, x, y); colormap = :viridis, colorrange = (0, seq_max))

        ax2 = Axis(fig[row, 2], aspect = DataAspect(), title = row == 1 ? "Ground truth" : "")
        Makie.heatmap!(ax2, Sleipnir.reverseForHeatmap(d.gt, x, y); colormap = :viridis, colorrange = (0, seq_max))

        ax3 = Axis(fig[row, 3], aspect = DataAspect(), title = row == 1 ? "Relative error (%)" : "")
        hm3 = Makie.heatmap!(ax3, Sleipnir.reverseForHeatmap(d.relerr, x, y); colormap = :RdBu_11, colorrange = (-relerr_lim, relerr_lim))

        d.field == :H && h_ref_hm === nothing && (h_ref_hm = hm1)
        d.field == :V && v_ref_hm === nothing && (v_ref_hm = hm1)
        err_ref_hm === nothing && (err_ref_hm = hm3)

        for (ax, col, letter) in zip((ax1, ax2, ax3), (1, 2, 3), row_letters)
            hidedecorations!(ax)
            for curve in outline
                lines!(ax, first.(curve.vertices), last.(curve.vertices), color = :black, linewidth = 1, alpha = 0.4)
            end
            _add_panel_label!(fig, row, col, letter)
        end
        for curve in thin_ctr.lines
            length(curve.vertices) > 5 || continue
            lines!(ax3, first.(curve.vertices), last.(curve.vertices), color = :red, linewidth = 1, alpha = 0.6)
        end

        field_name = d.field == :H ? "ice thickness" : "surface velocity magnitude"
        push!(caption_lines,
            "($(row_letters[1])-$(row_letters[3])) $(field_name), scenario $(d.sid) " *
            "(loss=$(d.scenario.loss_type), tim=$(d.scenario.use_tim), sparsity_H=$(d.scenario.sparsity_H), " *
            "V_sigma=$(d.scenario.sparsity_V_sigma)); predicted / ground truth / relative error. " *
            "A relative error = $(round(d.rel_err, digits = 3))%. Red outline: thin ice (H < 1 m), " *
            "where relative error is inflated by near-zero reference values.",
        )
    end

    # Shared colorbars, spanning the contiguous rows that use each quantity.
    h_rows = [row for (row, d) in enumerate(row_data) if d.field == :H]
    v_rows = [row for (row, d) in enumerate(row_data) if d.field == :V]
    !isnothing(h_ref_hm) && Colorbar(fig[first(h_rows):last(h_rows), 4], h_ref_hm; label = "Thickness (m)", width = 14)
    !isnothing(v_ref_hm) && Colorbar(fig[first(v_rows):last(v_rows), 4], v_ref_hm; label = "Velocity norm (m/yr)", width = 14)
    Colorbar(fig[:, 5], err_ref_hm; label = "Relative error (%)", width = 14)

    rowgap!(fig.layout, 8)
    colgap!(fig.layout, 8)

    resize_to_layout!(fig)
    outpath = joinpath(context.results_dir, "showcase_$(glacier_id).$(output_format)")
    CairoMakie.save(outpath, fig; px_per_unit = output_format == :png ? 3 : 1)
    println("Saved: ", outpath)
    println()
    println("Caption reference (paste/adapt into the LaTeX caption):")
    for line in caption_lines
        println("  ", line)
    end
    return outpath
end

build_showcase(glacier_id, rows; output_format = output_format)
