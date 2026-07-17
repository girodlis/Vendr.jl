using Revise
#using Vendr

using CairoMakie
using SciMLSensitivity

using ODINN 

rgi_paths = get_rgi_paths()

rgi_ids = ["RGI60-11.02773"]

params = Parameters(
        simulation = SimulationParameters(
            use_MB = false,
            tspan = (2017.0, 2018.0),
            multiprocessing = false,
            rgi_paths = rgi_paths,
            ice_thickness_source = :Millan22,
            velocity_product = :Millan22,
        ),
        hyper = Hyperparameters(
            batch_size = length(rgi_ids),
            epochs = [100, 20],
            optimizer = [
                ODINN.Adam(0.01),
                ODINN.LBFGS(linesearch = ODINN.LineSearches.BackTracking(iterations = 20)),
            ],
            # epochs = 50,
            # #optimizer = ODINN.Adam(0.09),
            # optimizer = ODINN.LBFGS(linesearch = ODINN.LineSearches.BackTracking(iterations = 5)),
        ),
        physical = PhysicalParameters(
            minA = 1e-18,
            maxA = 1e-16, 
        ),
        UDE = UDEparameters(
            #### Adjoint auto
            #grad = SciMLSensitivityAdjoint(),
            #optim_autoAD = ODINN.Optimization.AutoZygote(),
            #sensealg = InterpolatingAdjoint(autojacvec = SciMLSensitivity.EnzymeVJP()),

            #grad =  DiscreteAdjoint(),
            optim_autoAD = ODINN.NoAD(),
            
            empirical_loss_function = LossAvgV(component = :abs),
            # empirical_loss_function = MultiLoss(
            #     losses = (
            #         LossAvgV(),
            #         InitialThicknessRegularization(t₀ = 2017.0),
            #     ),
            #     λs = (1.0, 1.0),
            # ),
            #target = :A, # default
            #initial_condition_filter = :Zang1980
        ),
        solver = Huginn.SolverParameters(
            step = 1/12,
        ),
    )

glaciers = Sleipnir.initialize_glaciers(rgi_ids, params)

######################################
######## Inversion of A ##############
######################################

trainable_model = GlacierWideInv(params, glaciers, :A) 
A_law = LawA(params; scalar = true)
iceflow = SIA2Dmodel(params; A = A_law)

model = Model(
    iceflow = iceflow,
    mass_balance = nothing,
    regressors = (; A = trainable_model),
)

inversion = Inversion(model, glaciers, params)
run!(inversion)

A_opt = only(Vendr._target_value(inversion, 1, :A; θ = inversion.results.stats.θ))
println("A optimisé : ", A_opt)

######################################
######## Inversion of H0 ##############
######################################

ic = InitialCondition(params, glaciers, :Millan22)
iceflow2 = SIA2Dmodel(params; A = ConstantA(A_opt))

model2 = Model(
    iceflow = iceflow2,
    mass_balance = nothing,
    regressors = (; IC = ic),   # seulement IC trainable
)

inversion2 = Inversion(model2, glaciers, params)
run!(inversion2)

##################### Results

# Optimized A values per glacier
# A_values = [
#     only(Vendr._target_value(inversion, i, :A; θ = inversion.results.stats.θ))
#     for i in eachindex(inversion.glaciers)
# ]

results = inversion.results.simulation[1]
fig = Sleipnir.plot_glacier(results, "evolution difference", [:H];
               tspan = results.tspan, metrics = ["difference"])
# path = "/Users/girodli/.julia/dev/Vendr.jl/inversions/03_real/31_invert_A_scalar/fig_H_evol.png"
# save(path, fig)

fig = Sleipnir.plot_glacier(results, "heatmaps", [:V]; timeIdx = 12)
# path = "/Users/girodli/.julia/dev/Vendr.jl/inversions/03_real/31_invert_A_scalar/fig_V_heatmap.png"
# save(path, fig)

################# Averaging

θ = inversion.results.stats.θ
glacier_idx = 1
inversion.cache = Sleipnir.init_cache(inversion.model, inversion, glacier_idx, θ)  

avg_Vx, avg_Vy, avg_V = Huginn.averageV(
    θ, inversion,
    (results.date1_Vref[1], results.date2_Vref[1]),
    1/12,
    results.t, results.H,
)

V_millan = results.V_ref[1]
diff = avg_V .- V_millan

################# Plots

vmax = max(maximum(filter(isfinite, V_millan)), maximum(filter(isfinite, avg_V)))
dmax = maximum(abs.(filter(isfinite, diff)))

fig = Figure(size = (1300, 400))

ax1 = Axis(fig[1, 1], aspect = DataAspect(), title = "V Millan22")
hm1 = heatmap!(ax1, Sleipnir.reverseForHeatmap(V_millan, results.x, results.y),
    colormap = :viridis, colorrange = (0, vmax))
Colorbar(fig[1, 2], hm1)

ax2 = Axis(fig[1, 3], aspect = DataAspect(), title = "avg_V (modèle)")
hm2 = heatmap!(ax2, Sleipnir.reverseForHeatmap(avg_V, results.x, results.y),
    colormap = :viridis, colorrange = (0, vmax))
Colorbar(fig[1, 4], hm2)

ax3 = Axis(fig[1, 5], aspect = DataAspect(), title = "diff (modèle - Millan)")
hm3 = heatmap!(ax3, Sleipnir.reverseForHeatmap(diff, results.x, results.y),
    colormap = :RdBu, colorrange = (-dmax, dmax))
Colorbar(fig[1, 6], hm3)
save("compare_V.png", fig)

