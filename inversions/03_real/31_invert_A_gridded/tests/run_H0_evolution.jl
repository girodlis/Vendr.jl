using Revise
using CairoMakie
using SciMLSensitivity
using ODINN
using Vendr
using Random

const FileIO = CairoMakie.FileIO

cd(@__DIR__)

# ODINN's DiscreteAdjoint gradient path calls `get_mb_model(mass_balance, glacier_idx)`
# unconditionally (ML_utils.jl, VJPs.jl), but the function was never defined anywhere in
# the package (added in a074a23, alongside per-glacier MB calibration, without its
# definition). We don't use per-glacier MB (mass_balance = nothing below), so just return
# the model unchanged — matching the pre-a074a23 behaviour.
Core.eval(ODINN, :(get_mb_model(mb_model, glacier_idx) = mb_model))

rgi_paths = get_rgi_paths()
rgi_ids = [
  "RGI60-11.02773",
]

params = Parameters(
    simulation = SimulationParameters(
        use_MB             = false,
        tspan              = (2017.0, 2018.0),
        multiprocessing    = false,
        gridScalingFactor  = 3,
        rgi_paths          = rgi_paths,
        ice_thickness_source = :Farinotti19,
        velocity_product   = :Millan22,
    ),
    hyper = Hyperparameters(
        batch_size = length(rgi_ids),
        epochs     = [200, 300],
        optimizer  = [
            ODINN.Adam(0.0005),
            ODINN.LBFGS(linesearch = ODINN.LineSearches.BackTracking(iterations = 5)),
        ],
    ),
    physical = PhysicalParameters(
        minA = 1e-18,
        maxA = 1e-16,
    ),
    UDE = UDEparameters(
        optim_autoAD = ODINN.NoAD(),
        empirical_loss_function = MultiLoss(
            losses = (
                LossAvgV(loss = LogSum(ϵ = 10.0), component = :abs),
                RheologyRegularization(),
                InitialThicknessRegularization(2017.0),
            ),
            λs = (1.0, 3e37, 1e-40),
        ),
        initial_condition_filter = :Zang1980,
    ),
    solver = Huginn.SolverParameters(step = 1/24),
)

# `_build_glacier` builds glacier.velocityData when `velocity_product = :Millan22` above,
# OR when the loss function is auto-detected as needing it via `Sleipnir.velocityProduct`
# — but that auto-detect has no implementation for ODINN's loss types and is checked
# eagerly whenever `params.UDE` is set, so it errors regardless of `velocity_product`.
# Build glaciers with a UDE-less Parameters (same simulation/physical) so only the
# explicit `velocity_product` is used, never the broken auto-detect.
glacier_params = Sleipnir.Parameters(; physical = params.physical, simulation = params.simulation)
glaciers = Sleipnir.initialize_glaciers(rgi_ids, glacier_params)

######################################
#### Joint inversion of A and H₀ ####
######################################

trainable_model = GriddedInv(params, glaciers, :A)
ic              = InitialCondition(params, glaciers, :Farinotti19) #Farinotti19Random
iceflow         = SIA2Dmodel(params; A = LawA(params; scalar = false))

model = Model(
    iceflow      = iceflow,
    mass_balance = nothing,
    regressors   = (; A = trainable_model, IC = ic),
)

inversion = Inversion(model, glaciers, params)
run!(inversion)

θ_opt = inversion.results.stats.θ

outdir = @__DIR__
println("Loss — start=", inversion.results.stats.losses[1], ", end=", inversion.results.stats.losses[end])

# ---- Does the gradient actually reach θ.IC, or is it structurally zero? ----
# Direct empirical test (independent of reading the code): if ‖∇IC‖ is zero or NaN at
# EVERY iteration, the gradient does not propagate back to H₀ (wiring bug). If it is
# non-zero but small compared to ‖∇A‖, it is a weak signal (indirect H₀ → V coupling
# through the forward integration), not a break.
norm2(x) = sqrt(sum(abs2, x))
for (glacier_idx, rgi_id) in enumerate(rgi_ids)
    glacier_id_sym = Symbol("$(glacier_idx)")
    ∇θ_hist  = inversion.results.stats.∇θ_hist
    grad_A   = [norm2(g.A[glacier_id_sym])  for g in ∇θ_hist]
    grad_IC  = [norm2(g.IC[glacier_id_sym]) for g in ∇θ_hist]

    n_zero_IC = count(iszero, grad_IC)
    n_nan_IC  = count(isnan, grad_IC)

    println("--- $rgi_id: gradient A vs IC over $(length(∇θ_hist)) iterations ---")
    println("  ‖∇A‖  — first=", grad_A[1],  ", last=", grad_A[end],  ", max=", maximum(grad_A))
    println("  ‖∇IC‖ — first=", grad_IC[1], ", last=", grad_IC[end], ", max=", maximum(grad_IC))
    println("  ‖∇IC‖ zero at $(n_zero_IC)/$(length(grad_IC)) iterations")
    println("  ‖∇IC‖ NaN  at $(n_nan_IC)/$(length(grad_IC)) iterations")
    println("  Mean ratio ‖∇IC‖/‖∇A‖: ", sum(grad_IC ./ grad_A) / length(grad_A))
end

# H₀ lives on the (nx, ny) grid; is_in_glacier needs a mask on that same grid — see
# 02_synthetic/23_gridded_A_IC_reg_vs_noreg/run.jl for the same helper.
function glacier_mask_for(H₀::AbstractMatrix, target_size::Tuple{Int, Int}; distance::Int = 1)
    size(H₀) == target_size && return Sleipnir.is_in_glacier(H₀, distance)
    H₀_interior = H₀[1:(end - 1), 1:(end - 1)]
    size(H₀_interior) == target_size && return Sleipnir.is_in_glacier(H₀_interior, distance)
    return trues(target_size)
end

# Glacier-outline contour (marching squares on the mask), same convention as
# Sleipnir.plot_glacier_heatmaps(; plotContour=true).
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

# ---- Piece-by-piece rendering (see 02_synthetic/23_gridded_A_IC_reg_vs_noreg/run.jl and
# the pre-existing panels in this same directory) ----
# Makie's GridLayout can tie a colorbar's width to an axis's width across columns via
# Aspect-ratio negotiation (colsize!/rowsize!(Aspect(...))), but that path proved unreliable
# on real (irregular, contour-heavy) glacier data. Sidestepping it: render each panel/colorbar
# to its own PNG at an exact, chosen pixel size, then reassemble with Fixed() layout sizes.
function render_panel(field, x, y, ctr, nx, ny; scale, colormap, colorrange, outfile)
    fig = Figure(size = (nx * scale, ny * scale), figure_padding = 0)
    ax  = Axis(fig[1, 1], aspect = DataAspect())
    heatmap!(ax, Sleipnir.reverseForHeatmap(field, x, y), colormap = colormap, colorrange = colorrange)
    overlay_contour!(ax, ctr)
    hidedecorations!(ax)
    hidespines!(ax)
    xlims!(ax, 1, nx)
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

######################################
##### Evolution of H₀ (IC) along the inversion iterations #####
######################################
# θ.IC (initialized from Farinotti19) is optimized just like A: callback_diagnosis
# pushes a snapshot of θ at EVERY iteration into stats.θ_hist
# (ODINN/src/simulations/inversions/callback_utils.jl). For each snapshot, the effective
# H₀ field is `evaluate_H₀(θ, glacier, filter, glacier_id)` (ODINN/.../InitialCondition_utils.jl),
# which applies the same positivity filter (:Zang1980 here) as the forward run.
θ_hist   = inversion.results.stats.θ_hist
n_adam   = params.hyper.epochs[1]
ic_filter = params.UDE.initial_condition_filter

for (glacier_idx, rgi_id) in enumerate(rgi_ids)
    println("\n===== $rgi_id (glacier $glacier_idx/$(length(rgi_ids))) =====")
    glacier      = glaciers[glacier_idx]
    H₀_farinotti = glacier.H₀

    results = inversion.results.simulation[glacier_idx]
    mask_glacier = glacier_mask_for(H₀_farinotti, size(H₀_farinotti))
    ctr          = contour_for(mask_glacier, results.x, results.y)
    nx_grid, ny_grid = size(mask_glacier)

    n_snapshots = 5
    snap_idx    = unique(round.(Int, range(1, length(θ_hist), length = n_snapshots)))

    H_masked = [
        ifelse.(mask_glacier, ODINN.evaluate_H₀(θ_hist[i], glacier, ic_filter, glacier_idx), NaN)
        for i in snap_idx
    ]
    hmax = maximum(maximum(filter(isfinite, H)) for H in H_masked)

    scale   = 8
    panel_w = nx_grid * scale
    panel_h = ny_grid * scale
    cbar_h  = 90
    n       = length(snap_idx)

    mktempdir() do tmp
        fig_ic = Figure(size = (n * panel_w, panel_h + cbar_h + 40), fontsize = 22)

        for (col, i) in enumerate(snap_idx)
            f_panel = joinpath(tmp, "it$(col).png")
            render_panel(H_masked[col], results.x, results.y, ctr, nx_grid, ny_grid;
                scale, colormap = :Blues, colorrange = (0, hmax), outfile = f_panel)
            place_image!(fig_ic, 2, col, f_panel)
            phase = i <= n_adam ? "Adam" : "LBFGS"
            Label(fig_ic[1, col], "iter $(i) ($(phase))", fontsize = 20, tellwidth = false)
            colsize!(fig_ic.layout, col, Fixed(panel_w))
        end

        f_cb = joinpath(tmp, "cb.png")
        render_colorbar_h(:Blues, (0, hmax), "H₀ (m)"; width_px = panel_w, height_px = cbar_h, outfile = f_cb)
        place_image!(fig_ic, 3, cld(n, 2), f_cb)

        rowsize!(fig_ic.layout, 2, Fixed(panel_h))
        rowsize!(fig_ic.layout, 3, Fixed(cbar_h))
        colgap!(fig_ic.layout, 10)
        rowgap!(fig_ic.layout, 8)
        resize_to_layout!(fig_ic)

        save(joinpath(outdir, "H0_evolution_iterations_$(rgi_id).png"), fig_ic)
        println("Saved: H0_evolution_iterations_$(rgi_id).png")
    end
end
