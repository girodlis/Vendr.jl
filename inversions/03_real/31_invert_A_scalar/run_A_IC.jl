using Revise
using CairoMakie
using SciMLSensitivity
using ODINN
using Vendr

rgi_paths = get_rgi_paths()
rgi_ids   = ["RGI60-11.02773"]

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
        epochs     = [200, 30],
        optimizer  = [
            ODINN.Adam(0.01),
            ODINN.LBFGS(linesearch = ODINN.LineSearches.BackTracking(iterations = 20)),
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
                LossAvgV(component = :abs),
                InitialThicknessRegularization(2017.0),
            ),
            λs = (1.0, 1e-4),
        ),
        initial_condition_filter = :Zang1980,
    ),
    solver = Huginn.SolverParameters(step = 1/12),
)

glaciers = Sleipnir.initialize_glaciers(rgi_ids, params)

# Reference data from Millan22
t1_vel    = Sleipnir.datetime_to_floatyear(only(glaciers[1].velocityData.date1))
t2_vel    = Sleipnir.datetime_to_floatyear(only(glaciers[1].velocityData.date2))
V_millan  = only(glaciers[1].velocityData.vabs)
H₀_millan = glaciers[1].H₀

######################################
#### Joint inversion of A and H₀ ####
######################################

trainable_model = GlacierWideInv(params, glaciers, :A)
ic              = InitialCondition(params, glaciers, :Millan22)
iceflow         = SIA2Dmodel(params; A = LawA(params; scalar = true))

model = Model(
    iceflow      = iceflow,
    mass_balance = nothing,
    regressors   = (; A = trainable_model, IC = ic),
)

inversion = Inversion(model, glaciers, params)
run!(inversion)

A_opt = only(Vendr._target_value(inversion, 1, :A; θ = inversion.results.stats.θ))
println("A optimisé : ", A_opt)

######################################
############# Diagnostics ############
######################################

results = inversion.results.simulation[1]
θ_opt   = inversion.results.stats.θ
inversion.cache = Sleipnir.init_cache(inversion.model, inversion, 1, θ_opt)

Huginn.apply_all_callback_laws!(
    inversion.model.iceflow, inversion.cache.iceflow, inversion, 1, t1_vel, θ_opt)

avg_Vx, avg_Vy, avg_V = Huginn.averageV(
    θ_opt, inversion, (t1_vel, t2_vel), 1/12, results.t, results.H,
)

# H₀ inverted = initial condition of the forward run (first recorded state, t = tspan[1])
H₀_inverted = results.H[1]
glacier_mask = H₀_millan .> 0

######################################
############# Plots ##################
######################################

outdir = @__DIR__

# ---- Plot 1: IC check — H₀ Millan22 vs H₀ inversé ----
H₀_diff = ifelse.(glacier_mask, H₀_inverted .- H₀_millan, NaN)
hmax    = max(maximum(filter(isfinite, H₀_millan)), maximum(filter(isfinite, H₀_inverted)))
dmax_H  = maximum(abs.(filter(isfinite, H₀_diff)))

fig_ic = Figure(size = (1300, 400))

ax_ic1 = Axis(fig_ic[1, 1], aspect = DataAspect(), title = "H₀ Millan22 (prior)")
hm_ic1 = heatmap!(ax_ic1, Sleipnir.reverseForHeatmap(H₀_millan, results.x, results.y),
    colormap = :Blues, colorrange = (0, hmax))
Colorbar(fig_ic[1, 2], hm_ic1, label = "m")

ax_ic2 = Axis(fig_ic[1, 3], aspect = DataAspect(), title = "H₀ inversé")
hm_ic2 = heatmap!(ax_ic2, Sleipnir.reverseForHeatmap(H₀_inverted, results.x, results.y),
    colormap = :Blues, colorrange = (0, hmax))
Colorbar(fig_ic[1, 4], hm_ic2, label = "m")

ax_ic3 = Axis(fig_ic[1, 5], aspect = DataAspect(), title = "diff H₀ (inversé − Millan22)")
hm_ic3 = heatmap!(ax_ic3, Sleipnir.reverseForHeatmap(H₀_diff, results.x, results.y),
    colormap = :RdBu, colorrange = (-dmax_H, dmax_H))
Colorbar(fig_ic[1, 6], hm_ic3, label = "m")

save(joinpath(outdir, "compare_H0_A_IC.png"), fig_ic)
println("Saved: compare_H0_A_IC.png")

# ---- Plot 2: Velocity comparison ----
V_diff = ifelse.(V_millan .> 0, avg_V .- V_millan, NaN)
vmax   = max(maximum(filter(isfinite, V_millan)), maximum(filter(isfinite, avg_V)))
dmax_V = maximum(abs.(filter(isfinite, V_diff)))

fig_v = Figure(size = (1300, 400))

ax_v1 = Axis(fig_v[1, 1], aspect = DataAspect(), title = "V Millan22")
hm_v1 = heatmap!(ax_v1, Sleipnir.reverseForHeatmap(V_millan, results.x, results.y),
    colormap = :viridis, colorrange = (0, vmax))
Colorbar(fig_v[1, 2], hm_v1, label = "m/yr")

ax_v2 = Axis(fig_v[1, 3], aspect = DataAspect(), title = "avg_V modèle (A + H₀ inversés)")
hm_v2 = heatmap!(ax_v2, Sleipnir.reverseForHeatmap(avg_V, results.x, results.y),
    colormap = :viridis, colorrange = (0, vmax))
Colorbar(fig_v[1, 4], hm_v2, label = "m/yr")

ax_v3 = Axis(fig_v[1, 5], aspect = DataAspect(), title = "diff V (modèle − Millan22)")
hm_v3 = heatmap!(ax_v3, Sleipnir.reverseForHeatmap(V_diff, results.x, results.y),
    colormap = :RdBu, colorrange = (-dmax_V, dmax_V))
Colorbar(fig_v[1, 6], hm_v3, label = "m/yr")

save(joinpath(outdir, "compare_V_A_IC.png"), fig_v)
println("Saved: compare_V_A_IC.png")
