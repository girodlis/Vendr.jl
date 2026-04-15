"""
    plotting/summary_grid.jl

Campaign summary and grid plotting utilities.
"""

function _compact_scenario_label(label::AbstractString; max_len::Int = 34)
    str = String(label)
    if ncodeunits(str) <= max_len
        return str
    end
    parts = split(str, " | ")
    if length(parts) >= 3
        first_line = join(parts[1:2], " | ")
        second_line = join(parts[3:end], " | ")
        compact = string(first_line, "\n", second_line)
        return ncodeunits(compact) <= max_len + 12 ? compact : string(first(parts), "\n", last(parts))
    end
    return string(first(str, max_len - 1), "…")
end

function _build_scenario_tick_labels(labels::Vector{String})
    n = length(labels)
    if n <= 10
        return [_compact_scenario_label(lbl; max_len = 38) for lbl in labels], nothing
    end
    short = ["S$(i)" for i in eachindex(labels)]
    legend_lines = ["S$(i): $(labels[i])" for i in eachindex(labels)]
    return short, join(legend_lines, "\n")
end

function _scenario_panel_text(label::AbstractString)
    compact = _compact_scenario_label(label; max_len = 48)
    parts = split(compact, " | ")
    if length(parts) <= 1
        return compact
    end
    return string(parts[1], "\n", join(parts[2:end], " | "))
end

function _max_abs_finite(values)
    limit = 0.0
    found_finite = false
    for value in values
        isfinite(value) || continue
        found_finite = true
        limit = max(limit, abs(value))
    end
    return limit, found_finite
end

function _pick_state(data, idx, context::AbstractString)
    data isa AbstractVector || return data

    n = length(data)
    n > 0 || error("Encountered an empty time series in $(context)")

    selected_idx = isnothing(idx) ? n : idx
    1 <= selected_idx <= n || error("timeIdx=$(selected_idx) is out of bounds for a series of length $(n) in $(context)")
    return data[selected_idx]
end

function _thickness_reference(results::Sleipnir.Results, timeIdx::Union{Nothing, Int64}; prefer_unmasked::Bool = true)
    _ = prefer_unmasked
    return _pick_state(results.H_ref, timeIdx, "plot_thickness_differences_grid")
end

"""
    plot_relative_error_boxplot(; 
        csv_path="outputs/results/inversion_summary.csv", 
        output_path=nothing, 
        log_scale=true, 
        use_absolute_error=false)

Create a boxplot of `relative_error_percent` grouped by scenario label
(`scenario | loss_type | tim=...`) from the inversion summary CSV.
"""
function plot_relative_error_boxplot(;
        csv_path::String = "outputs/results/inversion_summary.csv",
        output_path::Union{Nothing, String} = nothing,
        log_scale::Bool = true,
        use_absolute_error::Bool = false,
    )
    # TODO: make this plot target-aware (A/C) and support field-level metrics for distributed inversions.
    df = CSV.read(csv_path, DataFrame)
    hasproperty(df, :scenario) || error("Missing column: scenario")
    hasproperty(df, :loss_type) || error("Missing column: loss_type")
    hasproperty(df, :use_tim) || error("Missing column: use_tim")
    hasproperty(df, :relative_error_percent) || error("Missing column: relative_error_percent")

    df.scenario_label = string.(df.scenario) .* " | " .* df.loss_type .* " | tim=" .* string.(df.use_tim)
    error_values = use_absolute_error ? abs.(df.relative_error_percent) : df.relative_error_percent

    scenario_labels = unique(df.scenario_label)
    scenario_x = 1:length(scenario_labels)
    scenario_values = [error_values[df.scenario_label .== label] for label in scenario_labels]
    tick_labels, mapping_text = _build_scenario_tick_labels(String.(scenario_labels))

    if log_scale && any(v -> any(<=(0), v), scenario_values)
        error("log_scale=true requires strictly positive values. Use use_absolute_error=true with strictly positive values, or log_scale=false.")
    end

    fig_width = max(1200, 90 * length(scenario_labels))
    fig_height = isnothing(mapping_text) ? 600 : 980
    fig = Figure(size = (fig_width, fig_height))
    ax = Axis(
        fig[1, 1],
        xlabel = "Scenario",
        ylabel = use_absolute_error ? (log_scale ? "Absolute relative error (%) (log10)" : "Absolute relative error (%)") : (log_scale ? "Relative error (%) (log10)" : "Relative error (%)"),
        title = use_absolute_error ? "`A` absolute relative error by scenario" : "`A` relative error by scenario",
        xticks = (scenario_x, tick_labels),
        yscale = log_scale ? log10 : identity,
        xticklabelrotation = π / 4,
        xticklabelalign = (:right, :center),
        xticklabelsize = 10,
    )

    for (idx, values) in enumerate(scenario_values)
        CairoMakie.boxplot!(ax, fill(idx, length(values)), values)
    end

    if !isnothing(mapping_text)
        Label(
            fig[2, 1],
            "Scenario mapping\n" * mapping_text,
            halign = :left,
            valign = :top,
            justification = :left,
            tellwidth = false,
            fontsize = 12,
        )
    end

    if isnothing(output_path)
        output_path = joinpath(
            dirname(csv_path),
            use_absolute_error ? (log_scale ? "temperate_ice_absolute_relative_error_log_boxplot.png" : "temperate_ice_absolute_relative_error_boxplot.png") : (log_scale ? "temperate_ice_relative_error_log_boxplot.png" : "temperate_ice_relative_error_boxplot.png"),
        )
    end

    CairoMakie.save(output_path, fig)
    return output_path
end

"""
    plot_thickness_differences_grid(results_vector, scenario_names; kwargs...) -> Figure

Plot thickness differences (inverted - ground truth) for multiple scenarios in a grid.
"""
function plot_thickness_differences_grid(
        results_vector::Vector{Sleipnir.Results},
        scenario_names::Vector{String};
        figsize::Union{Nothing, Tuple{Int64, Int64}} = nothing,
        scale_text_size::Union{Nothing, Float64} = nothing,
        timeIdx::Union{Nothing, Int64} = nothing,
        plotContour::Bool = false,
        prefer_unmasked_reference::Bool = true,
)
    @assert !isempty(results_vector) "results_vector must not be empty"
    @assert length(results_vector) == length(scenario_names) "Number of results must match number of scenario names"

    num_scenarios = length(results_vector)
    rows = clamp(round(Int, sqrt(num_scenarios)), 1, 3)
    cols = ceil(Int, num_scenarios / rows)

    res1 = results_vector[1]
    lon = res1.lon
    lat = res1.lat
    x = res1.x
    y = res1.y
    rgi_id = res1.rgi_id
    Δx = res1.Δx
    reference_base = _thickness_reference(res1, timeIdx; prefer_unmasked = prefer_unmasked_reference)
    mask = reference_base .> 0.0
    nx, ny = size(mask)

    ctr = plotContour ? Sleipnir.Contour.contour(collect(1:nx), 1 + ny .- collect(1:ny), mask, 0.5) : nothing

    thickness_diffs = Vector{Matrix{Float64}}(undef, num_scenarios)
    color_limit = 0.0
    found_finite_value = false
    for (i, results) in enumerate(results_vector)
        H_current = _pick_state(results.H, timeIdx, "plot_thickness_differences_grid")
        H_ref = _thickness_reference(results, timeIdx; prefer_unmasked = prefer_unmasked_reference)

        size(H_current) == size(H_ref) || error("Scenario $(i): H and H_ref have different sizes ($(size(H_current)) vs $(size(H_ref))).")
        size(H_ref) == (nx, ny) || error("Scenario $(i): H_ref size $(size(H_ref)) does not match first scenario size $((nx, ny)).")

        diff = Float64.(H_current .- H_ref)
        diff[.!mask] .= NaN
        thickness_diffs[i] = diff

        scenario_limit, scenario_has_finite = _max_abs_finite(diff)
        color_limit = max(color_limit, scenario_limit)
        found_finite_value |= scenario_has_finite
    end

    found_finite_value || error("No finite thickness differences were found to plot")
    color_limit > 0 || (color_limit = 1.0)

    base_panel_width = 300
    base_panel_height = 220
    scale_factor = clamp(8.0 / max(cols, 1), 0.7, 1.2)
    panel_width = round(Int, base_panel_width * scale_factor)
    panel_height = round(Int, base_panel_height * scale_factor)

    colorbar_width = 90
    title_height = 80
    spacing_height = 40

    computed_figsize = (cols * panel_width + colorbar_width + 20,
                        rows * panel_height + title_height + spacing_height)
    final_figsize = isnothing(figsize) ? computed_figsize : figsize

    fig = Figure(size = final_figsize)
    gl = fig.layout

    reference_heatmap = nothing

    text_scale = scale_factor
    fontsize_label = max(7, round(Int, 9 * text_scale))
    fontsize_title = max(8, round(Int, 10 * text_scale))

    for i in eachindex(results_vector)
        row = rem(i - 1, rows) + 1
        col = div(i - 1, rows) + 1

        ax = Axis(fig[row, col], aspect = DataAspect())
        diff = thickness_diffs[i]

        hm = Makie.heatmap!(ax,
            Sleipnir.reverseForHeatmap(diff, x, y),
            colormap = :RdBu_11,
            colorrange = (-color_limit, color_limit)
        )
        reference_heatmap === nothing && (reference_heatmap = hm)

        if plotContour && !isnothing(ctr)
            for curve in ctr.lines
                xs = first.(curve.vertices)
                ys = last.(curve.vertices)
                Makie.lines!(ax, xs, ys, color = :black, linewidth = 1, alpha = 0.3)
            end
        end

        ax.xlabel = row == rows ? "Longitude" : ""
        ax.ylabel = col == 1 ? "Latitude" : ""
        ax.xticks = ([round(nx / 2)], ["$(round(lon; digits = 6)) °"])
        ax.yticks = ([round(ny / 2)], ["$(round(lat; digits = 6)) °"])
        ax.yticklabelrotation = π / 2
        ax.ylabelpadding = 5
        ax.yticklabelalign = (:center, :bottom)
        if row != rows
            hidexdecorations!(ax, grid = false)
        end
        if col != 1
            hideydecorations!(ax, grid = false)
        end

        panel_label = _scenario_panel_text(scenario_names[i])
        text!(
            ax,
            panel_label,
            position = (0.98 * nx, 0.98 * ny),
            align = (:right, :top),
            fontsize = fontsize_label,
            color = :black,
        )

        scale_width = 0.10 * nx
        scale_number = round(Δx * scale_width / 1000; digits = 1)
        textsize = isnothing(scale_text_size) ? max(0.3 * scale_width, fontsize_label) : scale_text_size

        poly!(ax, Rect(nx - round(0.15 * nx), round(0.075 * ny), scale_width, scale_width / 10),
            color = :black)
        text!(ax,
            "$scale_number km",
            position = (nx - round(0.15 * nx) + scale_width / 16,
                       round(0.075 * ny) + scale_width / 10),
            fontsize = textsize)
    end

    Colorbar(fig[:, cols + 1], reference_heatmap; label = "Thickness difference (m)")
    fig[0, :] = Label(fig, "Ice Thickness Differences: Inverted - Ground Truth | $rgi_id", fontsize = fontsize_title)

    colsize!(gl, cols + 1, colorbar_width)
    for c in 1:cols
        colsize!(gl, c, panel_width)
    end
    for r in 1:rows
        rowsize!(gl, r, panel_height)
    end
    resize_to_layout!(fig)
    return fig
end