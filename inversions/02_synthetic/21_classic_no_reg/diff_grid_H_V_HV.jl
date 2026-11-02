using CairoMakie
using JLD2
using Sleipnir

# Works two ways:
#   - Live REPL, right after run_campaign! (`context` in scope): reads straight from
#     context.scenarios / context.scenario_inversions.
#   - Fresh, lightweight session (no ODINN/Vendr campaign re-run needed): reads from
#     the JLD2 snapshot produced by save_snapshot.jl. Only needs `using Sleipnir,
#     CairoMakie, JLD2`.
#
# Produces one combined figure, across all 10 glaciers, comparing the spatial
# diff (inverted - ground truth) for three scenarios that tell the "which loss buys
# you what" story:
#   - H   : Loss H,  no mask   (S1)  -> thickness diff
#   - V   : Loss V,  no mask   (S2)  -> velocity diff
#   - HV  : Loss HV, H masked, V unmasked (S7) -> thickness diff AND velocity diff
# Layout: glaciers as columns, one row per (scenario, field) category.
# Color scale is shared across all :H rows and, separately, across all :V rows, so
# that error magnitude is visually comparable between scenarios for the same field.

results_dir = joinpath(dirname(abspath(@__FILE__)), "outputs", "results_10_glaciers")

if @isdefined(context)
    scenario_meta = context.scenarios
    results_by_scenario = Dict(
        s.id => inversion.results.simulation
        for (s, inversion) in zip(context.scenarios, context.scenario_inversions)
    )
else
    snapshot_path = joinpath(results_dir, "campaign_snapshot.jld2")
    isfile(snapshot_path) || error(
        "No `context` in scope and no snapshot at $(snapshot_path). " *
        "Run save_snapshot.jl once from the live campaign REPL first.",
    )
    snap = JLD2.load(snapshot_path)
    scenario_meta = snap["scenario_meta"]
    results_by_scenario = snap["results_by_scenario"]
end

# Each entry: (scenario_id, field, row_label). field = :H or :V.
categories = [
    ("S1", :H, "a."),
    ("S2", :V, "b."),
    ("S7", :H, "c."),
    ("S7", :V, "d."),
]

output_format = :pdf  # :png or :pdf (PDF keeps text/lines vector; heatmaps stay data-resolution raster either way)
output_path = joinpath(results_dir, "diff_grid_H_V_HV.$(output_format)")

function _scenario_and_results(scenario_id::String, glacier_id::String)
    si = findfirst(s -> s.id == scenario_id, scenario_meta)
    isnothing(si) && error("Scenario $(scenario_id) not found")
    scenario = scenario_meta[si]
    gi = findfirst(==(glacier_id), scenario.rgi_ids)
    isnothing(gi) && error("Glacier $(glacier_id) not found in scenario $(scenario_id)")
    return scenario, results_by_scenario[scenario_id][gi]
end

function _field_diff(results, field::Symbol)
    field_ref = Symbol(field, "_ref")
    pred = Float64.(getproperty(results, field)[end])
    gt   = Float64.(getproperty(results, field_ref)[end])
    mask = Sleipnir.is_in_glacier(Float64.(results.H_ref[end]), 1)
    diff = ifelse.(mask, pred .- gt, NaN)
    return diff, mask
end

_finite_absmax(vs...) = maximum(abs.(filter(isfinite, vcat(vs...))); init = 1.0)

# Wrapped in a function: a top-level `for` loop mutating variables declared outside
# it is ambiguous soft-scope in a script/include'd file (see showcase.jl).
function build_diff_grid(categories, glacier_ids; output_path::String)
    # --- Pass 1: gather all panel data + compute shared color scales per field ---
    panel_data = Matrix{NamedTuple}(undef, length(categories), length(glacier_ids))
    for (row, (sid, field, label)) in enumerate(categories)
        for (col, gid) in enumerate(glacier_ids)
            scenario, results = _scenario_and_results(sid, gid)
            diff, mask = _field_diff(results, field)
            panel_data[row, col] = (sid = sid, field = field, label = label, gid = gid,
                                     scenario = scenario, results = results, diff = diff, mask = mask)
        end
    end

    # vec() each panel's diff before concatenating: glaciers have different grid
    # sizes, so vcat'ing 2D matrices directly would fail on mismatched dimensions.
    h_lim = _finite_absmax((vec(panel_data[row, col].diff) for row in 1:length(categories), col in 1:length(glacier_ids)
                             if panel_data[row, col].field == :H)...)
    v_lim = _finite_absmax((vec(panel_data[row, col].diff) for row in 1:length(categories), col in 1:length(glacier_ids)
                             if panel_data[row, col].field == :V)...)

    n_rows = length(categories)
    n_cols = length(glacier_ids)
    panel_size = 200

    fig = Figure(size = (panel_size * n_cols + 220, panel_size * n_rows + 110))

    # Rows are not grouped contiguously by field (H, V, HV(H), HV(V)), so each row
    # gets its own colorbar rather than one shared bar spanning a row range —
    # H-rows and V-rows still share the same (-h_lim,h_lim) / (-v_lim,v_lim) scale.
    for row in 1:n_rows
        row_hm = nothing
        for col in 1:n_cols
            d = panel_data[row, col]
            lim = d.field == :H ? h_lim : v_lim

            ax = Axis(fig[row, col], aspect = DataAspect(), valign = :top)
            hm = Makie.heatmap!(ax, Sleipnir.reverseForHeatmap(d.diff, d.results.x, d.results.y),
                                 colormap = :RdBu_11, colorrange = (-lim, lim))
            row_hm === nothing && (row_hm = hm)

            nx, ny = size(d.mask)
            ctr = Sleipnir.Contour.contour(collect(1:nx), 1 + ny .- collect(1:ny), d.mask, 0.5)
            outline = filter(curve -> length(curve.vertices) > 15, ctr.lines)
            for curve in outline
                lines!(ax, first.(curve.vertices), last.(curve.vertices), color = :black, linewidth = 1, alpha = 0.4)
            end
            hidedecorations!(ax)
            hidespines!(ax)
        end
        Label(fig[row, 0], categories[row][3]; rotation = pi/2, tellheight = false, fontsize = 16, font = :bold)
        field_label = categories[row][2] == :H ? "Thickness diff (m)" : "Velocity diff (m/yr)"
        Colorbar(fig[row, n_cols + 1], row_hm; label = field_label, width = 22, labelsize = 14, ticklabelsize = 12)
    end

    # Dedicated label row for glacier IDs (rather than Axis titles): titles attached
    # to an aspect-locked Axis inherit its vertical centering within the row, so they
    # land at different heights per glacier. A plain Label row has no such coupling.
    for col in 1:n_cols
        Label(fig[0, col], glacier_ids[col]; fontsize = 15, font = :bold, tellwidth = false)
    end

    rowgap!(fig.layout, 8)
    colgap!(fig.layout, 8)
    resize_to_layout!(fig)

    if output_format == :png
        CairoMakie.save(output_path, fig; px_per_unit = 3)
    else
        CairoMakie.save(output_path, fig)
    end
    println("Saved: ", output_path)
    println()
    println("Caption reference (row -> scenario/field):")
    for (sid, field, label) in categories
        println("  $(label) $(field) diff, scenario $(sid)")
    end
    return fig
end

build_diff_grid(categories, first(scenario_meta).rgi_ids; output_path = output_path)
