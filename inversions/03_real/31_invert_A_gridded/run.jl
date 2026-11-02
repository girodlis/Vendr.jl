# Joint inversion of a gridded rheology A and of the initial thickness H₀ on a real glacier,
# using Millan et al. (2022) surface velocities.
#
# - Model: SIA, no sliding, no mass balance, tspan 2017-2018
# - Loss: LossAvgV (LogSum) + RheologyRegularization on A + InitialThicknessRegularization on H₀
# - H₀ initialized from Farinotti19; Millan22 thickness is only used as reference in the plots
#
# Run from the Vendr.jl root: `julia --project=. inversions/03_real/31_invert_A_gridded/run.jl`
# Figures are written to `outputs/<date>_<rgi_ids>/`, together with a copy of this script
# and the git commit it was run from.

using Revise
using CairoMakie
using SciMLSensitivity
using ODINN
using Vendr
using Random
using Dates

const FileIO = CairoMakie.FileIO

cd(@__DIR__)

rgi_paths = get_rgi_paths()
rgi_ids = [
  "RGI60-11.01450",
  #"RGI60-11.03676",
]

params = Parameters(
    simulation = SimulationParameters(
        use_MB             = false,
        tspan              = (2017.0, 2018.0),
        multiprocessing    = false,
        gridScalingFactor  = 1,
        rgi_paths          = rgi_paths,
        ice_thickness_source = :Millan22,
        velocity_product   = :Millan22,
        #use_glathida_data = true,
        #test_mode = true,
    ),
    hyper = Hyperparameters(
        batch_size = length(rgi_ids),
        epochs     = [400, 300],
        optimizer  = [
            ODINN.Adam(0.1),
            ODINN.LBFGS(linesearch = ODINN.LineSearches.BackTracking(iterations = 15)),
        ],
    ),
    physical = PhysicalParameters(
        minA = 1e-23,
        maxA = 4e-16,
    ),
    UDE = UDEparameters(
        optim_autoAD = ODINN.NoAD(),
        #grad = DiscreteAdjoint(),
        empirical_loss_function = MultiLoss(
            losses = (
                LossAvgV(loss = LogSum(ϵ = 10.0), component = :abs),
                RheologyRegularization(),
                InitialThicknessRegularization(2017.0),
            ),
            λs = (1.0, 3e38, 1e-4),
        ),
        initial_condition_filter = :Zang1980,
    ),
    solver = Huginn.SolverParameters(step = 1/48),
)

glaciers = Sleipnir.initialize_glaciers(rgi_ids, params)

######################################
#### Joint inversion of A and H₀ ####
######################################

trainable_model = GriddedInv(params, glaciers, :A)
# Farinotti19 avoids circularity with V_millan (Millan22 thickness is derived from Millan22 velocities).
# Alternatives: :Millan22, :Farinotti19Random
ic              = InitialCondition(params, glaciers, :Farinotti19)
iceflow         = SIA2Dmodel(params; A = LawA(params; scalar = false))

model = Model(
    iceflow      = iceflow,
    mass_balance = nothing,
    regressors   = (; A = trainable_model, IC = ic),
)

inversion = Inversion(model, glaciers, params)

# θ = inversion.model.trainable_components.θ
# ratio, angle, relerr, (dθ, dθ_FD) = grad_finite_diff(inversion) #, max_params=500)
# println(ratio)
# println(angle)
# println(relerr) 

run!(inversion)



#######################################
####### Results part ########
#######################################

θ_opt = inversion.results.stats.θ

run_name    = "$(Dates.today())_$(join(replace.(rgi_ids, "RGI60-11." => ""), "_"))"
outdir      = mkpath(joinpath(@__DIR__, "outputs", run_name))
cp(@__FILE__, joinpath(outdir, "run.jl"); force = true)
write(joinpath(outdir, "git_commit.txt"), readchomp(`git -C $(@__DIR__) rev-parse HEAD`) * "\n")

norm2(x)    = sqrt(sum(abs2, x))
A_UNIT      = 1e-17
A_temperate = Huginn.polyA_PatersonCuffey()(0.0) # A(0°C) selon Cuffey & Paterson

function glacier_mask_for(H₀::AbstractMatrix, target_size::Tuple{Int, Int}; distance::Int = 1)
    size(H₀) == target_size && return Sleipnir.is_in_glacier(H₀, distance)
    H₀_interior = H₀[1:(end - 1), 1:(end - 1)]
    size(H₀_interior) == target_size && return Sleipnir.is_in_glacier(H₀_interior, distance)
    return trues(target_size)
end

function contour_for(mask::AbstractMatrix, x::Vector{<:AbstractFloat}, y::Vector{<:AbstractFloat})
    m = Sleipnir.reverseForHeatmap(Float64.(mask), x, y)
    nx, ny = size(m)
    return Sleipnir.Contour.contour(collect(1:nx), collect(1:ny), m, 0.5)
end

function overlay_contour!(ax, ctr)
    for curve in ctr.lines
        lines!(ax, first.(curve.vertices), last.(curve.vertices), color = :black, linewidth = 1.5, alpha = 0.5)
    end
end

function render_panel(field, x, y, ctr, nx, ny; scale, colormap, colorrange, outfile)
    fig = Figure(size = (nx * scale, ny * scale), figure_padding = 0)
    ax  = Axis(fig[1, 1], aspect = DataAspect())
    heatmap!(ax, Sleipnir.reverseForHeatmap(field, x, y), colormap = colormap, colorrange = colorrange)
    overlay_contour!(ax, ctr)
    hidedecorations!(ax)
    hidespines!(ax)
    xlims!(ax, 1, nx)  # lock limits so contour vertices overshooting the grid can't letterbox the panel
    ylims!(ax, 1, ny)
    save(outfile, fig)
    return outfile
end

function render_colorbar_h(colormap, colorrange, label; width_px, height_px = 90, outfile)
    fig = Figure(size = (width_px, height_px), figure_padding = 5)
    Colorbar(fig[1, 1], colormap = colormap, limits = colorrange, vertical = false,
        label = label, labelsize = 20, ticklabelsize = 16)
    save(outfile, fig)
    return outfile
end

function place_image!(fig, row, col, path)
    img = FileIO.load(path)
    ax  = Axis(fig[row, col], aspect = DataAspect())
    image!(ax, rotr90(img))
    hidedecorations!(ax)
    hidespines!(ax)
    return ax
end

function save_field_panel(
        path::AbstractString, title::AbstractString, field::AbstractMatrix,
        x::Vector{<:AbstractFloat}, y::Vector{<:AbstractFloat}, ctr;
        colormap, colorrange::Tuple{<:Real, <:Real}, unit_label::AbstractString,
        colorbar_below::Bool = false,
)::Nothing
    fig = Figure(size = colorbar_below ? (550, 650) : (650, 550))
    title_cols = colorbar_below ? 1 : (1:2)
    Label(fig[1, title_cols], title; fontsize = 22, font = :bold, word_wrap = true, tellwidth = false)
    ax = Axis(fig[2, 1], aspect = DataAspect(),
        xlabelsize = 20, ylabelsize = 20, xticklabelsize = 17, yticklabelsize = 17)
    hm = heatmap!(ax, Sleipnir.reverseForHeatmap(field, x, y), colormap = colormap, colorrange = colorrange)
    overlay_contour!(ax, ctr)
    ax.xlabel = "Lon"
    ax.ylabel = "Lat"
    if colorbar_below
        Colorbar(fig[3, 1], hm, label = unit_label, labelsize = 20, ticklabelsize = 17, vertical = false)
    else
        Colorbar(fig[2, 2], hm, label = unit_label, labelsize = 20, ticklabelsize = 17)
    end
    # DataAspect() locks the plotted content's aspect ratio but not the axis's own
    # cell size, so without this the GridLayout can leave the axis cell far larger
    # than the data actually needs (visible as big blank margins around the heatmap).
    # Tie column 1's width to row 2's height via the field's own nx/ny ratio, same fix
    # as the 3-panel A_reg_vs_noreg figure in 02_synthetic/23_.../run.jl.
    nx, ny = size(field)
    colsize!(fig.layout, 1, Aspect(2, nx / ny))
    resize_to_layout!(fig)
    save(path, fig)
    println("Saved: $(basename(path))")
    return nothing
end

######################################
##### Convergence #####
######################################

losses = inversion.results.stats.losses

fig_loss = Figure(size = (600, 400))
ax_loss  = Axis(fig_loss[1, 1],
    title  = "Loss during optimization (batch of $(length(rgi_ids)) glaciers)",
    xlabel = "iteration", ylabel = "loss (log scale)", yscale = log10)
lines!(ax_loss, 1:length(losses), losses, color = :steelblue)

save(joinpath(outdir, "loss_curve_discradj_millan22.png"), fig_loss)
println("Saved: loss_curve_discradj_millan22.png")
println("Loss — début=", losses[1], ", fin=", losses[end])

######################################
############# Per-glacier diagnostics & plots ##################
######################################

real_data_stats = Dict{String, NamedTuple}()

for (glacier_idx, rgi_id) in enumerate(rgi_ids)
    println("\n===== $rgi_id (glacier $glacier_idx/$(length(rgi_ids))) =====")
    glacier        = glaciers[glacier_idx]
    glacier_id_sym = Symbol("$(glacier_idx)")

    t1_vel    = Sleipnir.datetime_to_floatyear(only(glacier.velocityData.date1))
    t2_vel    = Sleipnir.datetime_to_floatyear(only(glacier.velocityData.date2))
    V_millan  = only(glacier.velocityData.vabs)
    H₀_millan = glacier.H₀

    A_opt = Vendr._target_value(inversion, glacier_idx, :A; θ = θ_opt)
    println("A optimisé (min/mean/max) : ", extrema(A_opt), " / ", sum(A_opt) / length(A_opt))

    # ---- Gradient diagnostic: A vs IC ----
    # H₀_inverted reste-t-il proche du prior parce que le signal de fit qui l'atteint est
    # juste faible (H₀ n'influence V qu'indirectement, via un an d'intégration de la PDE,
    # alors que A influence directement la diffusivité SIA) ? On compare les normes des
    # deux blocs de gradient au fil de l'optimisation, stockées dans stats.∇θ_hist.
    ∇θ_hist      = inversion.results.stats.∇θ_hist
    grad_A_norm  = [norm2(g.A[glacier_id_sym]) for g in ∇θ_hist]
    grad_IC_norm = [norm2(g.IC[glacier_id_sym]) for g in ∇θ_hist]

    println("Norme du gradient — A  : dernier=", grad_A_norm[end],  ", max=", maximum(grad_A_norm))
    println("Norme du gradient — IC : dernier=", grad_IC_norm[end], ", max=", maximum(grad_IC_norm))
    println("Ratio ‖∇IC‖/‖∇A‖ (dernier itér.) : ", grad_IC_norm[end] / grad_A_norm[end])

    fig_grad = Figure(size = (600, 400))
    ax_grad  = Axis(fig_grad[1, 1],
        title  = "Gradient norm during optimization: A vs IC — $rgi_id",
        xlabel = "iteration", ylabel = "‖∇θ‖ (log scale)", yscale = log10)
    lines!(ax_grad, 1:length(grad_A_norm), grad_A_norm, color = :steelblue, label = "‖∇A‖")
    lines!(ax_grad, 1:length(grad_IC_norm), grad_IC_norm, color = :orange, label = "‖∇IC‖")
    axislegend(ax_grad, position = :rt)

    save(joinpath(outdir, "grad_norm_A_vs_IC_discradj_millan22_$(rgi_id).png"), fig_grad)
    println("Saved: grad_norm_A_vs_IC_discradj_millan22_$(rgi_id).png")

    # ---- Check d'aliasing : ∇θ_hist / θ_hist stockent-ils des références au même buffer ? ----
    # callback_diagnosis (ODINN/src/simulations/inversions/callback_utils.jl) fait
    # `push!(stats.θ_hist, θ.u)` / `push!(stats.∇θ_hist, θ.grad)` sans `copy(...)`. Déjà
    # vérifié une fois (résultat : pas d'aliasing), on garde le check ici par glacier
    # au cas où le comportement diffère selon le batch.
    θ_hist = inversion.results.stats.θ_hist
    println("Alias check — ∇θ_hist[1].A === ∇θ_hist[end].A : ",
        ∇θ_hist[1].A[glacier_id_sym] === ∇θ_hist[end].A[glacier_id_sym])
    println("Alias check — θ_hist[1].A == θ_hist[end].A : ",
        all(θ_hist[1].A[glacier_id_sym] .== θ_hist[end].A[glacier_id_sym]))

    # ---- Forward run diagnostics ----
    results = inversion.results.simulation[glacier_idx]
    inversion.cache = Sleipnir.init_cache(inversion.model, inversion, glacier_idx, θ_opt)

    Huginn.apply_all_callback_laws!(
        inversion.model.iceflow, inversion.cache.iceflow, inversion, glacier_idx, t1_vel, θ_opt)

    avg_Vx, avg_Vy, avg_V = Huginn.averageV(
        θ_opt, inversion, (t1_vel, t2_vel), 1/12, results.t, results.H,
    )

    # H₀ inverted = initial condition of the forward run (first recorded state, t = tspan[1])
    H₀_inverted = results.H[1]

    # ---- Plot 1+2 combined (poster layout): H₀ (col 1) and V (col 2), ----
    # ---- rows = Millan22 reference / inverted / diff ----
    # Same mask + contour convention as A_reg_vs_noreg in
    # 02_synthetic/23_gridded_A_IC_reg_vs_noreg/run.jl. Built piece-by-piece (see
    # render_panel/render_colorbar_h/place_image! above) and reassembled with Fixed() sizes
    # so the column layout is pixel-exact by construction.
    mask_glacier = glacier_mask_for(H₀_millan, size(H₀_millan))
    ctr = contour_for(mask_glacier, results.x, results.y)
    nx_grid, ny_grid = size(mask_glacier)

    H₀_millan_m   = ifelse.(mask_glacier, H₀_millan, NaN)
    H₀_inverted_m = ifelse.(mask_glacier, H₀_inverted, NaN)
    H₀_diff_m     = ifelse.(mask_glacier, H₀_inverted .- H₀_millan, NaN)
    hmax          = max(maximum(filter(isfinite, H₀_millan_m)), maximum(filter(isfinite, H₀_inverted_m)))
    dmax_H        = maximum(abs.(filter(isfinite, H₀_diff_m)))

    V_millan_m = ifelse.(mask_glacier, V_millan, NaN)
    avg_V_m    = ifelse.(mask_glacier, avg_V, NaN)
    V_diff_m   = ifelse.(mask_glacier, avg_V .- V_millan, NaN)
    vmax       = max(maximum(filter(isfinite, V_millan_m)), maximum(filter(isfinite, avg_V_m)))
    dmax_V     = maximum(abs.(filter(isfinite, V_diff_m)))

    # ---- Summary stats for the report (H₀ and V residuals) ----
    H0_abs_diff = abs.(filter(isfinite, H₀_diff_m))
    V_abs_diff  = abs.(filter(isfinite, V_diff_m))
    H0_mean_abs_diff = sum(H0_abs_diff) / length(H0_abs_diff)
    H0_p95_abs_diff  = sort(H0_abs_diff)[ceil(Int, 0.95 * length(H0_abs_diff))]
    V_mean_abs_diff  = sum(V_abs_diff) / length(V_abs_diff)
    V_rmse           = sqrt(sum(abs2, V_abs_diff) / length(V_abs_diff))

    println("H₀ |inverted − Millan22| — mean=", H0_mean_abs_diff, " m, p95=", H0_p95_abs_diff, " m")
    println("V  |inverted − Millan22| — mean=", V_mean_abs_diff, " m/yr, RMSE=", V_rmse, " m/yr")

    scale   = 8
    panel_w = nx_grid * scale
    panel_h = ny_grid * scale
    cbar_h  = 90

    col_specs = (
        (col = 1, label = "H₀ (m)", gt = H₀_millan_m, inv = H₀_inverted_m, diff = H₀_diff_m,
            colormap = :Blues, range = (0, hmax), dmax = dmax_H, unit = "m"),
        (col = 2, label = "V (m/yr)", gt = V_millan_m, inv = avg_V_m, diff = V_diff_m,
            colormap = :viridis, range = (0, vmax), dmax = dmax_V, unit = "m/yr"),
    )

    mktempdir() do tmp
        fig_cmp = Figure(size = (2 * panel_w, 3 * panel_h + 2 * cbar_h), fontsize = 26)

        for spec in col_specs
            c = spec.col
            Label(fig_cmp[0, c], spec.label, fontsize = 30, font = :bold, tellwidth = false)

            f_gt, f_inv, f_diff = joinpath(tmp, "c$(c)_gt.png"), joinpath(tmp, "c$(c)_inv.png"), joinpath(tmp, "c$(c)_diff.png")
            f_cb, f_cbd = joinpath(tmp, "c$(c)_cb.png"), joinpath(tmp, "c$(c)_cbd.png")

            render_panel(spec.gt, results.x, results.y, ctr, nx_grid, ny_grid;
                scale, colormap = spec.colormap, colorrange = spec.range, outfile = f_gt)
            render_panel(spec.inv, results.x, results.y, ctr, nx_grid, ny_grid;
                scale, colormap = spec.colormap, colorrange = spec.range, outfile = f_inv)
            render_panel(spec.diff, results.x, results.y, ctr, nx_grid, ny_grid;
                scale, colormap = :RdBu, colorrange = (-spec.dmax, spec.dmax), outfile = f_diff)
            render_colorbar_h(spec.colormap, spec.range, spec.unit;
                width_px = panel_w, height_px = cbar_h, outfile = f_cb)
            render_colorbar_h(:RdBu, (-spec.dmax, spec.dmax), "diff ($(spec.unit))";
                width_px = panel_w, height_px = cbar_h, outfile = f_cbd)

            place_image!(fig_cmp, 1, c, f_gt)
            place_image!(fig_cmp, 2, c, f_inv)
            place_image!(fig_cmp, 4, c, f_diff)
            place_image!(fig_cmp, 3, c, f_cb)
            place_image!(fig_cmp, 5, c, f_cbd)
        end

        Label(fig_cmp[1, 0], "Millan22\n(reference)", fontsize = 26, rotation = pi / 2, tellheight = false)
        Label(fig_cmp[2, 0], "Inverted", fontsize = 26, rotation = pi / 2, tellheight = false)
        Label(fig_cmp[4, 0], "Diff\n(inverted − GT)", fontsize = 26, rotation = pi / 2, tellheight = false)

        colsize!(fig_cmp.layout, 1, Fixed(panel_w))
        colsize!(fig_cmp.layout, 2, Fixed(panel_w))
        rowsize!(fig_cmp.layout, 1, Fixed(panel_h))
        rowsize!(fig_cmp.layout, 2, Fixed(panel_h))
        rowsize!(fig_cmp.layout, 4, Fixed(panel_h))
        rowsize!(fig_cmp.layout, 3, Fixed(cbar_h))
        rowsize!(fig_cmp.layout, 5, Fixed(cbar_h))
        colgap!(fig_cmp.layout, 10)
        rowgap!(fig_cmp.layout, 8)
        resize_to_layout!(fig_cmp)

        save(joinpath(outdir, "compare_H0_V_A_IC_reg_millan22_discradj_$(rgi_id).png"), fig_cmp)
        println("Saved: compare_H0_V_A_IC_reg_millan22_discradj_$(rgi_id).png")
    end

    # ---- Same H₀/V comparison, as 6 individual single-panel PNGs ----
    # gt and inv reuse spec.range (fixed above, from gt+inv extrema, before plotting), so
    # each field's two images stay on the same color scale.
    for spec in col_specs
        prefix = spec.col == 1 ? "H0" : "V"
        save_field_panel(joinpath(outdir, "$(prefix)_ground_truth_$(rgi_id).png"),
            "$(spec.label) — ground truth (Millan22)", spec.gt, results.x, results.y, ctr;
            colormap = spec.colormap, colorrange = spec.range, unit_label = spec.unit, colorbar_below = true)
        save_field_panel(joinpath(outdir, "$(prefix)_inverted_$(rgi_id).png"),
            "$(spec.label) — inverted", spec.inv, results.x, results.y, ctr;
            colormap = spec.colormap, colorrange = spec.range, unit_label = spec.unit, colorbar_below = true)
        save_field_panel(joinpath(outdir, "$(prefix)_diff_$(rgi_id).png"),
            "$(spec.label) — diff (inverted − GT)", spec.diff, results.x, results.y, ctr;
            colormap = :RdBu, colorrange = (-spec.dmax, spec.dmax), unit_label = "diff ($(spec.unit))", colorbar_below = true)
    end

    # ---- Plot 3: carte de A inversé ----
    # A vit sur la grille "staggered" (nx-1, ny-1), un cran plus petite que H₀/V (nx, ny).
    glacier_mask_A = glacier_mask_for(H₀_millan, size(A_opt))
    x_A = (results.x[1:end-1] .+ results.x[2:end]) ./ 2
    y_A = (results.y[1:end-1] .+ results.y[2:end]) ./ 2

    # A est de l'ordre de 1e-17 : Makie/PlotUtils ne trouve pas de ticks "ronds" sur une
    # plage aussi petite et n'affiche aucune valeur sur la colorbar. On réexprime donc A
    # en unités de A_UNIT pour rester dans une plage lisible.
    A_masked = ifelse.(glacier_mask_A, A_opt, NaN) ./ A_UNIT

    # ---- Summary stats for the report (A field, in units of A_UNIT) ----
    A_valid     = filter(isfinite, A_masked)
    A_high_frac = count(>(5.0), A_valid) / length(A_valid)
    println("A optimisé (unités ×1e-17) — mean=", sum(A_valid) / length(A_valid),
        ", fraction > 5e-17 : ", 100 * A_high_frac, "%")

    real_data_stats[rgi_id] = (
        H0_mean_abs_diff = H0_mean_abs_diff, H0_p95_abs_diff = H0_p95_abs_diff,
        V_mean_abs_diff = V_mean_abs_diff, V_rmse = V_rmse,
        A_mean = sum(A_valid) / length(A_valid), A_high_frac = A_high_frac,
    )

    ctr_A = contour_for(glacier_mask_A, x_A, y_A)

    fig_a = Figure(size = (500, 400))

    ax_a = Axis(fig_a[1, 1], aspect = DataAspect(), title = "Inverted A — $rgi_id")
    hm_a = heatmap!(ax_a, Sleipnir.reverseForHeatmap(A_masked, x_A, y_A),
        colormap = :YlGnBu)
    overlay_contour!(ax_a, ctr_A)
    Colorbar(fig_a[1, 2], hm_a, label = "A (×10⁻¹⁷ Pa⁻³ yr⁻¹)")

    save(joinpath(outdir, "map_A_inverted_reg_millan22_discradj_$(rgi_id).png"), fig_a)
    println("Saved: map_A_inverted_reg_millan22_discradj_$(rgi_id).png")

    # ---- Plot 4: distribution de A inversé vs référence glace tempérée ----
    bar_color    = "#228BE6"  # RGB(34, 139, 230)
    temp_color   = "#F35424"  # RGB(243, 84, 36)
    millan_color = "#228B22"  # RGB(34, 139, 34), placeholder green, no exact RGB was given

    fig_hist = Figure(size = (600, 400))
    ax_hist  = Axis(fig_hist[1, 1],
        xlabel = "A (×10⁻¹⁷ Pa⁻³ yr⁻¹)", ylabel = "Density",
        xlabelsize = 16, ylabelsize = 16, xticklabelsize = 13, yticklabelsize = 13)

    # Fine, fixed-width bins spanning the full prior range so that any pile-up
    # right at minA/maxA (parameter saturation) is visible as a narrow spike
    # at the edge rather than being smeared into a wide default bin.
    hist_bins = range(params.physical.minA / A_UNIT, params.physical.maxA / A_UNIT, length = 81)
    hist!(ax_hist, vec(A_opt[glacier_mask_A]) ./ A_UNIT, normalization = :pdf,
        bins = hist_bins, color = (bar_color, 0.75), strokewidth = 0.5, strokecolor = :black)

    vlines!(ax_hist, [A_temperate] ./ A_UNIT,
        color = temp_color, linewidth = 2, label = "A(0°C) Cuffey & Paterson")
    vlines!(ax_hist, [4.10e-17] ./ A_UNIT,
        color = millan_color, linewidth = 2, linestyle = :dot, label = "A Millan et al., 2022")

    axislegend(ax_hist, position = :rt, labelsize = 12, patchsize = (16, 10),
        framevisible = false, padding = (4, 4, 4, 4), rowgap = 2)

    # Bins above still span the full prior range so saturation at either bound stays
    # visible if it occurs, but with a wide prior (e.g. maxA = 4e-16) most of that range
    # is empty and the actual distribution ends up squeezed into a sliver on the left.
    # Zoom the displayed axis to the data (and the two reference lines) instead; if the
    # data genuinely extends near a bound, that bound naturally stays in view since it's
    # then part of `data_vals`'s own range.
    data_vals = vec(A_opt[glacier_mask_A]) ./ A_UNIT
    x_hi = max(maximum(data_vals), A_temperate / A_UNIT, 4.10e-17 / A_UNIT) * 1.15
    xlims!(ax_hist, 0, x_hi)

    # With a wide prior, the max bound sits far outside this zoomed view, so a vline
    # for it would draw nothing while still adding an empty-looking legend entry.
    # State the bounds as text instead, in the empty lower-right corner of the plot.
    prior_lo_raw = params.physical.minA / A_UNIT
    prior_lo_str = prior_lo_raw < 0.01 ? "0" : string(round(prior_lo_raw, sigdigits = 2))
    prior_hi_str = string(round(params.physical.maxA / A_UNIT, sigdigits = 2))
    text!(ax_hist, 0.98, 0.05, text = "prior bounds: $(prior_lo_str) to $(prior_hi_str)",
        space = :relative, align = (:right, :bottom), fontsize = 11, color = :gray40)

    save(joinpath(outdir, "hist_A_vs_temperate_reg_millan22_discradj_$(rgi_id).png"), fig_hist)
    println("Saved: hist_A_vs_temperate_reg_millan22_discradj_$(rgi_id).png")
end
