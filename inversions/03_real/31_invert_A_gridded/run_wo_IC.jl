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
  "RGI60-11.03676",
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
    ),
    hyper = Hyperparameters(
        batch_size = length(rgi_ids),
        epochs     = [5, 300],
        optimizer  = [
            ODINN.Adam(0.005),
            ODINN.LBFGS(linesearch = ODINN.LineSearches.BackTracking(iterations = 5)),
        ],
    ),
    physical = PhysicalParameters(
        minA = 1e-18,
        maxA = 1e-16,
    ),
    UDE = UDEparameters(
        optim_autoAD = ODINN.NoAD(),
        #grad = DiscreteAdjoint(),
        empirical_loss_function = MultiLoss(
            losses = (
                LossAvgV(loss = LogSum(ϵ = 10.0), component = :abs),
                RheologyRegularization(),
               # InitialThicknessRegularization(2017.0),
            ),
            λs = (1.0, 3e37),
        ),
        initial_condition_filter = :Zang1980,
    ),
    solver = Huginn.SolverParameters(step = 1/52),
)

glacier_params = Sleipnir.Parameters(; physical = params.physical, simulation = params.simulation)
glaciers = Sleipnir.initialize_glaciers(rgi_ids, glacier_params)

######################################
#### Joint inversion of A and H₀ ####
######################################

trainable_model = GriddedInv(params, glaciers, :A)
#ic              = InitialCondition(params, glaciers, :Millan22) #Farinotti19Random
iceflow         = SIA2Dmodel(params; A = LawA(params; scalar = false))

model = Model(
    iceflow      = iceflow,
    mass_balance = nothing,
    regressors   = (; A = trainable_model),
)

inversion = Inversion(model, glaciers, params)
run!(inversion)

θ_opt = inversion.results.stats.θ

run_name    = "$(Dates.today())_wo_IC_$(join(replace.(rgi_ids, "RGI60-11." => ""), "_"))"
outdir      = mkpath(joinpath(@__DIR__, "outputs", run_name))
cp(@__FILE__, joinpath(outdir, "run_wo_IC.jl"); force = true)
write(joinpath(outdir, "git_commit.txt"), readchomp(`git -C $(@__DIR__) rev-parse HEAD`) * "\n")
norm2(x)    = sqrt(sum(abs2, x))
A_UNIT      = 1e-17
A_temperate = Huginn.polyA_PatersonCuffey()(0.0) # A(0°C) from Cuffey & Paterson

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
    Label(fig[1, 1:2], title; fontsize = 22, font = :bold, word_wrap = true, tellwidth = false)
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
println("Loss — start=", losses[1], ", end=", losses[end])

######################################
############# Per-glacier diagnostics & plots ##################
######################################

for (glacier_idx, rgi_id) in enumerate(rgi_ids)
    println("\n===== $rgi_id (glacier $glacier_idx/$(length(rgi_ids))) =====")
    glacier        = glaciers[glacier_idx]
    glacier_id_sym = Symbol("$(glacier_idx)")

    t1_vel    = Sleipnir.datetime_to_floatyear(only(glacier.velocityData.date1))
    t2_vel    = Sleipnir.datetime_to_floatyear(only(glacier.velocityData.date2))
    V_millan  = only(glacier.velocityData.vabs)
    H₀_millan = glacier.H₀

    A_opt = Vendr._target_value(inversion, glacier_idx, :A; θ = θ_opt)
    println("Optimized A (min/mean/max): ", extrema(A_opt), " / ", sum(A_opt) / length(A_opt))

    # ---- Gradient diagnostic: A ----
    ∇θ_hist     = inversion.results.stats.∇θ_hist
    grad_A_norm = [norm2(g.A[glacier_id_sym]) for g in ∇θ_hist]

    println("Gradient norm — A : last=", grad_A_norm[end], ", max=", maximum(grad_A_norm))

    fig_grad = Figure(size = (600, 400))
    ax_grad  = Axis(fig_grad[1, 1],
        title  = "Gradient norm during optimization: A — $rgi_id",
        xlabel = "iteration", ylabel = "‖∇A‖ (log scale)", yscale = log10)
    lines!(ax_grad, 1:length(grad_A_norm), grad_A_norm, color = :steelblue)

    save(joinpath(outdir, "grad_norm_A_discradj_millan22_$(rgi_id).png"), fig_grad)
    println("Saved: grad_norm_A_discradj_millan22_$(rgi_id).png")

    # ---- Aliasing check: do ∇θ_hist / θ_hist store references to the same buffer? ----
    # callback_diagnosis (ODINN/src/simulations/inversions/callback_utils.jl) does
    # `push!(stats.θ_hist, θ.u)` / `push!(stats.∇θ_hist, θ.grad)` without `copy(...)`. Already
    # checked once (result: no aliasing); the check is kept here per glacier in case the
    # behaviour differs depending on the batch.
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

    # ---- Plot 1: V comparison (poster layout), rows = Millan22 reference / inverted / diff ----
    # Same mask + contour convention as A_reg_vs_noreg in
    # 02_synthetic/23_gridded_A_IC_reg_vs_noreg/run.jl. Built piece-by-piece (see
    # render_panel/render_colorbar_h/place_image! above) and reassembled with Fixed() sizes
    # so the layout is pixel-exact by construction.
    mask_glacier = glacier_mask_for(H₀_millan, size(H₀_millan))
    ctr = contour_for(mask_glacier, results.x, results.y)
    nx_grid, ny_grid = size(mask_glacier)

    V_millan_m = ifelse.(mask_glacier, V_millan, NaN)
    avg_V_m    = ifelse.(mask_glacier, avg_V, NaN)
    V_diff_m   = ifelse.(mask_glacier, avg_V .- V_millan, NaN)
    vmax       = max(maximum(filter(isfinite, V_millan_m)), maximum(filter(isfinite, avg_V_m)))
    dmax_V     = maximum(abs.(filter(isfinite, V_diff_m)))

    scale   = 8
    panel_w = nx_grid * scale
    panel_h = ny_grid * scale
    cbar_h  = 90

    spec = (col = 1, label = "V (m/yr)", gt = V_millan_m, inv = avg_V_m, diff = V_diff_m,
        colormap = :viridis, range = (0, vmax), dmax = dmax_V, unit = "m/yr")

    mktempdir() do tmp
        fig_cmp = Figure(size = (panel_w, 3 * panel_h + 2 * cbar_h), fontsize = 26)

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

        Label(fig_cmp[1, 0], "Millan22\n(reference)", fontsize = 26, rotation = pi / 2, tellheight = false)
        Label(fig_cmp[2, 0], "Inverted", fontsize = 26, rotation = pi / 2, tellheight = false)
        Label(fig_cmp[4, 0], "Diff\n(inverted − GT)", fontsize = 26, rotation = pi / 2, tellheight = false)

        colsize!(fig_cmp.layout, 1, Fixed(panel_w))
        rowsize!(fig_cmp.layout, 1, Fixed(panel_h))
        rowsize!(fig_cmp.layout, 2, Fixed(panel_h))
        rowsize!(fig_cmp.layout, 4, Fixed(panel_h))
        rowsize!(fig_cmp.layout, 3, Fixed(cbar_h))
        rowsize!(fig_cmp.layout, 5, Fixed(cbar_h))
        colgap!(fig_cmp.layout, 10)
        rowgap!(fig_cmp.layout, 8)
        resize_to_layout!(fig_cmp)

        save(joinpath(outdir, "compare_V_A_reg_millan22_discradj_$(rgi_id).png"), fig_cmp)
        println("Saved: compare_V_A_reg_millan22_discradj_$(rgi_id).png")
    end

    # ---- Same V comparison, as 3 individual single-panel PNGs ----
    # gt and inv reuse spec.range (fixed above, from gt+inv extrema, before plotting), so
    # both images stay on the same color scale.
    save_field_panel(joinpath(outdir, "V_ground_truth_$(rgi_id).png"),
        "$(spec.label) — ground truth (Millan22)", spec.gt, results.x, results.y, ctr;
        colormap = spec.colormap, colorrange = spec.range, unit_label = spec.unit, colorbar_below = true)
    save_field_panel(joinpath(outdir, "V_inverted_$(rgi_id).png"),
        "$(spec.label) — inverted", spec.inv, results.x, results.y, ctr;
        colormap = spec.colormap, colorrange = spec.range, unit_label = spec.unit, colorbar_below = true)
    save_field_panel(joinpath(outdir, "V_diff_$(rgi_id).png"),
        "$(spec.label) — diff (inverted − GT)", spec.diff, results.x, results.y, ctr;
        colormap = :RdBu, colorrange = (-spec.dmax, spec.dmax), unit_label = "diff ($(spec.unit))", colorbar_below = true)

    # ---- Plot 3: map of inverted A ----
    # A lives on the "staggered" grid (nx-1, ny-1), one cell smaller than H₀/V (nx, ny).
    glacier_mask_A = glacier_mask_for(H₀_millan, size(A_opt))
    x_A = (results.x[1:end-1] .+ results.x[2:end]) ./ 2
    y_A = (results.y[1:end-1] .+ results.y[2:end]) ./ 2

    # A is of order 1e-17: Makie/PlotUtils finds no "round" ticks on such a small range
    # and shows no value on the colorbar, so A is expressed in units of A_UNIT to keep
    # a readable range.
    A_masked = ifelse.(glacier_mask_A, A_opt, NaN) ./ A_UNIT

    ctr_A = contour_for(glacier_mask_A, x_A, y_A)

    fig_a = Figure(size = (500, 400))

    ax_a = Axis(fig_a[1, 1], aspect = DataAspect(), title = "Inverted A — $rgi_id")
    hm_a = heatmap!(ax_a, Sleipnir.reverseForHeatmap(A_masked, x_A, y_A),
        colormap = :YlGnBu)
    overlay_contour!(ax_a, ctr_A)
    Colorbar(fig_a[1, 2], hm_a, label = "A (×10⁻¹⁷ Pa⁻³ yr⁻¹)")

    save(joinpath(outdir, "map_A_inverted_reg_millan22_discradj_$(rgi_id).png"), fig_a)
    println("Saved: map_A_inverted_reg_millan22_discradj_$(rgi_id).png")

    # ---- Plot 4: distribution of inverted A vs temperate ice reference ----
    fig_hist = Figure(size = (600, 400))
    ax_hist  = Axis(fig_hist[1, 1],
        title  = "Distribution of inverted A vs temperate ice — $rgi_id",
        xlabel = "A (×10⁻¹⁷ Pa⁻³ yr⁻¹)", ylabel = "density")

    hist!(ax_hist, vec(A_opt[glacier_mask_A]) ./ A_UNIT, normalization = :pdf,
        color = (:steelblue, 0.6), strokewidth = 1, strokecolor = :black)

    vlines!(ax_hist, [params.physical.minA, params.physical.maxA] ./ A_UNIT,
        color = :gray, linestyle = :dash, label = "prior bounds (min/max A)")
    vlines!(ax_hist, [A_temperate] ./ A_UNIT,
        color = :red, linewidth = 2, label = "A(0°C) Cuffey & Paterson")
    vlines!(ax_hist, [4.10e-17] ./ A_UNIT,
        color = :darkgreen, linewidth = 2, linestyle = :dot, label = "A Millan et al., 2022")

    axislegend(ax_hist, position = :rt)

    save(joinpath(outdir, "hist_A_vs_temperate_reg_millan22_discradj_$(rgi_id).png"), fig_hist)
    println("Saved: hist_A_vs_temperate_reg_millan22_discradj_$(rgi_id).png")
end
