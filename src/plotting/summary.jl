"""
    plotting/summary.jl

Campaign summary and grid plotting utilities.
"""

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

    df.scenario_label = string.(df.loss_type) .* " | tim=" .* string.(df.use_tim) .* 
                            " | H=" .* string.(df.sparsity_H) .* " | Vσ=" .* string.(df.sparsity_V_sigma) .* 
                            " | Vthr=" .* string.(df.sparsity_V_threshold)
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
    # Note: timeIdx and prefer_unmasked_reference are reserved for future time-selection features
    # Currently always uses last time step (results.H[end], results.H_ref[end])

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
    # Use last time step of ground truth observations as stable glacier mask reference
    reference_base = res1.H_ref[end]
    mask = reference_base .> 0.0
    nx, ny = size(mask)

    ctr = plotContour ? Sleipnir.Contour.contour(collect(1:nx), 1 + ny .- collect(1:ny), mask, 0.5) : nothing

    thickness_diffs = Vector{Matrix{Float64}}(undef, num_scenarios)
    color_limit = 0.0
    found_finite_value = false
    for (i, results) in enumerate(results_vector)
        H_current = results.H[end]
        H_ref = results.H_ref[end]

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

"""
    plot_target_difference_grid(inversions_vector, predictions_vector, glacier_idx, target, scenario_names; kwargs...) -> Figure

Plot target parameter differences (inverted - ground truth) for multiple scenarios in a grid.

# Arguments
- `inversions_vector::Vector`: Array of ODINN.Inversion objects, one per scenario
- `predictions_vector::Vector`: Array of ground truth simulations (ODINN.Inversion), one per scenario
- `glacier_idx::Int`: Index of glacier to plot (1-4)
- `target::Symbol`: Target parameter to plot (:A or :C)
- `scenario_names::Vector{String}`: Names for each scenario
- `figsize::Union{Nothing, Tuple{Int,Int}}`: Figure size
- `scale_text_size::Union{Nothing, Float64}`: Scale text size
- `plotContour::Bool`: Plot glacier contour
- `relative_error::Bool`: If true, plot relative error instead of absolute difference

# Returns
Figure object
"""
function plot_target_difference_grid(
        inversions_vector::Vector,
        predictions_vector::Vector,
        glacier_idx::Int,
        target::Symbol,
        scenario_names::Vector{String};
        figsize::Union{Nothing, Tuple{Int64, Int64}} = nothing,
        scale_text_size::Union{Nothing, Float64} = nothing,
        plotContour::Bool = true,
        relative_error::Bool = false,
)
    num_scenarios = length(inversions_vector)
    rows = clamp(round(Int, sqrt(num_scenarios)), 1, 3)
    cols = ceil(Int, num_scenarios / rows)

    # Get coordinates and glacier info from first prediction
    glacier = predictions_vector[1].glaciers[glacier_idx]
    lon = glacier.Coords["lon"][1]
    lat = glacier.Coords["lat"][1]
    x = glacier.Coords["lon"]
    y = glacier.Coords["lat"]
    rgi_id = glacier.rgi_id
    Δx = glacier.Δx
    
    # Get target values from first scenario to determine dimensions (for the specific glacier_idx)
    first_inversion = inversions_vector[1]
    first_prediction = predictions_vector[1]
    first_inv_target = Vendr._target_value(first_inversion, glacier_idx, target; θ=first_inversion.results.stats.θ)
    first_gt_target = Vendr._target_value(first_prediction, glacier_idx, target; θ=nothing)
    first_inv_target = first_inv_target isa Matrix ? first_inv_target : Array(first_inv_target)
    first_gt_target = first_gt_target isa Matrix ? first_gt_target : Array(first_gt_target)
    
    # Determine dimensions based on the specific glacier
    interior_m, interior_n = size(first_gt_target)
    full_m = interior_m + 1 
    full_n = interior_n + 1
    
    # Create mask with full glacier dimensions (using the correct glacier)
    glacier = predictions_vector[1].glaciers[glacier_idx]
    mask = Vendr._valid_domain_mask(glacier, glacier.H₀)
    #mask = glacier.mask
    nx, ny = size(mask)
    
    # Verify mask matches expected full dimensions
    (nx, ny) == (full_m, full_n) || @warn "Expected mask dimensions ($full_m, $full_n) but got ($nx, $ny)"

    ctr = plotContour ? Sleipnir.Contour.contour(collect(1:nx), 1 + ny .- collect(1:ny), mask, 0.5) : nothing

    target_diffs = Vector{Matrix{Float64}}(undef, num_scenarios)
    color_limit = 0.0
    found_finite_value = false
    
    for (i, (inversion, prediction)) in enumerate(zip(inversions_vector, predictions_vector))
        inv_target = Vendr._target_value(inversion, glacier_idx, target; θ=inversion.results.stats.θ)
        gt_target = Vendr._target_value(prediction, glacier_idx, target; θ=nothing)
        
        # Ensure both are 2D matrices
        inv_target = inv_target isa Matrix ? inv_target : Array(inv_target)
        gt_target = gt_target isa Matrix ? gt_target : Array(gt_target)
        
        if size(inv_target) != (nx, ny)
            if size(inv_target) == (interior_m, interior_n)
                padded_inv = fill(NaN, nx, ny)
                padded_inv[1:interior_m, 1:interior_n] = inv_target
                inv_target = padded_inv
            else
                inv_target = inv_target[1:min(size(inv_target, 1), nx), 1:min(size(inv_target, 2), ny)]
            end
        end
        
        if size(gt_target) != (nx, ny)
            if size(gt_target) == (interior_m, interior_n)
                padded_gt = fill(NaN, nx, ny)
                padded_gt[1:interior_m, 1:interior_n] = gt_target
                gt_target = padded_gt
            else
                gt_target = gt_target[1:min(size(gt_target, 1), nx), 1:min(size(gt_target, 2), ny)]
            end
        end

        # Generate comparison plot for this scenario
        target_comp = plot_target_comparison(inv_target, gt_target, inversion.results.simulation[glacier_idx]; colormap=:RdBu_11, plotContour=plotContour)
        save(joinpath("outputs", "results", "target_comparison_$(rgi_id)_scenario_$(i).png"), target_comp)
        
        # Check dimensions match mask
        size(inv_target) == size(mask) || 
            error("Scenario $(i): inv_target size $(size(inv_target)) does not match mask size $(size(mask)).")
        size(gt_target) == size(mask) || 
            error("Scenario $(i): gt_target size $(size(gt_target)) does not match mask size $(size(mask)).")
        
        # Compute difference or relative error
        if relative_error
            valid = mask .& isfinite.(gt_target) .& isfinite.(inv_target) .& (gt_target .!= 0.0)
            diff = similar(gt_target, Float64)
            diff .= 0.0
            diff[valid] .= 100.0 .* ((inv_target[valid] .- gt_target[valid]) ./ gt_target[valid])
        else
            diff = Float64.(inv_target .- gt_target)
        end
        
        # Set values outside mask to NaN
        diff = ifelse.(mask, diff, NaN)
        
        target_diffs[i] = diff
        scenario_limit, scenario_has_finite = Vendr._max_abs_finite(diff)
        color_limit = max(color_limit, scenario_limit)
        found_finite_value |= scenario_has_finite
    end

    found_finite_value || error("No finite target differences were found to plot")
    color_limit > 0 || (color_limit = 1.0)

    # Layout sizing
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

    # Draw panels
    for i in eachindex(inversions_vector)
        row = rem(i - 1, rows) + 1
        col = div(i - 1, rows) + 1

        ax = Axis(fig[row, col], aspect = DataAspect())
        diff = target_diffs[i]

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

        panel_label = Vendr._scenario_panel_text(scenario_names[i])
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

    label_text = relative_error ? "Relative Error (%)" : "Target Difference"
    Colorbar(fig[:, cols + 1], reference_heatmap; label = label_text)
    title_text = "Parameter $target Differences: Inverted - Ground Truth | $rgi_id"
    fig[0, :] = Label(fig, title_text, fontsize = fontsize_title)

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

"""
    plot_target_comparison(inv_target, gt_target, results; colormap=:YlGnBu, kwargs...) -> Figure

Compare inverted vs ground truth target parameter side-by-side with log scale.

# Arguments
- `inv_target::Matrix`: Inverted target values (A or C)
- `gt_target::Matrix`: Ground truth target values
- `results::Results`: Results object for glacier metadata (lon, lat, x, y, rgi_id, Δx, H)
- `colormap`: Colormap to use (default :YlGnBu)
- `plotContour::Bool`: Plot glacier contour (default true)

# Returns
Figure object with two panels (Inverted | Ground Truth)
"""
function plot_target_comparison(
        inv_target::Matrix,
        gt_target::Matrix,
        results::Sleipnir.Results;
        colormap = :YlGnBu,
        plotContour::Bool = true,
)
    lon = results.lon
    lat = results.lat
    x = results.x
    y = results.y
    rgi_id = results.rgi_id
    Δx = results.Δx
    nx, ny = size(results.H[begin])

    # Use stable glacier mask from observations (not optimized initial thickness)
    mask = results.H_ref[end] .> 0.0
    
    # Handle dimension mismatch: pad target data if needed
    # Target parameters are often one dimension smaller than the full domain
    inv_padded = inv_target
    gt_padded = gt_target
    if size(inv_target) != (nx, ny)
        interior_m, interior_n = size(inv_target)
        if interior_m == nx - 1 && interior_n == ny - 1
            # Standard case: target has interior dimensions
            padded_inv = fill(NaN, nx, ny)
            padded_inv[1:interior_m, 1:interior_n] = inv_target
            inv_padded = padded_inv
            
            padded_gt = fill(NaN, nx, ny)
            padded_gt[1:interior_m, 1:interior_n] = gt_target
            gt_padded = padded_gt
        end
    end
    
    # Mask both datasets
    inv_masked = copy(inv_padded)
    gt_masked = copy(gt_padded)
    inv_masked[.!mask] .= NaN
    gt_masked[.!mask] .= NaN
    
    # Compute bounds for log scale
    positive_values = vcat(vec(inv_masked[isfinite.(inv_masked) .& (inv_masked .> 0)]),
                           vec(gt_masked[isfinite.(gt_masked) .& (gt_masked .> 0)]))
    global_min = minimum(positive_values)
    global_max = maximum(positive_values)
    
    ctr = plotContour ? Sleipnir.Contour.contour(collect(1:nx), 1 + ny .- collect(1:ny), mask, 0.5) : nothing
    
    fig = Figure(size = (1000, 450))
    
    # Left: Inverted
    ax1 = Axis(fig[1, 1], aspect = DataAspect())
    hm1 = Makie.heatmap!(ax1, Sleipnir.reverseForHeatmap(inv_masked, x, y), 
                   colormap = colormap, colorrange = (global_min, global_max), colorscale = log10)
    ax1.title = "Inverted"
    ax1.xlabel = "Longitude"
    ax1.ylabel = "Latitude"
    
    if plotContour && !isnothing(ctr)
        for curve in ctr.lines
            lines!(ax1, first.(curve.vertices), last.(curve.vertices), color = :black, linewidth = 1, alpha = 0.3)
        end
    end
    
    # Right: Ground Truth
    ax2 = Axis(fig[1, 2], aspect = DataAspect())
    hm2 = Makie.heatmap!(ax2, Sleipnir.reverseForHeatmap(gt_masked, x, y),
                   colormap = colormap, colorrange = (global_min, global_max), colorscale = log10)
    ax2.title = "Ground Truth"
    ax2.xlabel = "Longitude"
    ax2.ylabel = ""
    hideydecorations!(ax2, grid = false)
    
    if plotContour && !isnothing(ctr)
        for curve in ctr.lines
            lines!(ax2, first.(curve.vertices), last.(curve.vertices), color = :black, linewidth = 1, alpha = 0.3)
        end
    end
    
    Colorbar(fig[1, 3], hm1; label = "log₁₀(value)")
    fig[0, :] = Label(fig, "$rgi_id", fontsize = 14, font = :bold)
    
    return fig
end