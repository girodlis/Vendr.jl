using Revise
using Vendr
using ODINN 
using CairoMakie
using SciMLSensitivity

rgi_paths = get_rgi_paths()

rgi_ids = ["RGI60-11.02773"]

params = Parameters(
        simulation = SimulationParameters(
            use_MB = false,
            tspan = (2017.0, 2018.0),
            test_mode = false,
            multiprocessing = false,
            #workers = 4,
            use_glathida_data = true,
            gridScalingFactor = 1,
            rgi_paths = rgi_paths,
            ice_thickness_source = :Millan22,
        ),
        hyper = Hyperparameters(
            batch_size = length(rgi_ids),
            #epochs = [20, 20],
            # optimizer = [
            #     ODINN.Adam(0.09),
            #     ODINN.LBFGS(linesearch = ODINN.LineSearches.BackTracking(iterations = 5)),
            # ],
            epochs = 20,
            optimizer = ODINN.Adam(0.09),
        ),
        physical = PhysicalParameters(
            #minA = 8e-21,
            #maxA = 8e-17, 
            minA = 1e-18, 
            maxA = 3e-16, 
        ),
        UDE = UDEparameters(
            # Adjoint 
            #grad = SciMLSensitivityAdjoint(),
            #optim_autoAD = ODINN.Optimization.AutoZygote(),
            #sensealg = InterpolatingAdjoint(autojacvec = SciMLSensitivity.EnzymeVJP()),
            grad =  DiscreteAdjoint(),
            optim_autoAD = ODINN.NoAD(),
            
            empirical_loss_function = MultiLoss(
                losses = (
                    LossAvgV(),
                    InitialThicknessRegularization(t₀ = 2017.5),
                ),
                λs = (1.0, 1e-4),
            ),
            target = :A,
            initial_condition_filter = :Zang1980
        ),
        solver = Huginn.SolverParameters(
            step = 1/12,
        ),
    )

glaciers = Sleipnir.initialize_glaciers(rgi_ids, params)
glaciers = Sleipnir.generate_ground_truth(glaciers, params)

trainable_model = GriddedInv(params, glaciers, :A) 

ic = InitialCondition(params, glaciers, :Millan22)

A_law = LawA(params; scalar = false)

iceflow = SIA2Dmodel(params; A = A_law)
model = Model(
    iceflow = iceflow,
    mass_balance = nothing,
    regressors = (; A = trainable_model, IC = ic),
)

inversion = Inversion(model, glaciers, params)

run!(inversion)

# Optimized A values per glacier
A_values = [
    only(Vendr._target_value(inversion, i, :A; θ = inversion.results.stats.θ))
    for i in eachindex(inversion.glaciers)
]
# 1.268613475279205e-17

results = inversion.results.simulation[1]
fig = Sleipnir.plot_glacier(results, "evolution difference", [:H];
               tspan = results.tspan, metrics = ["difference"])
path = "/Users/girodli/.julia/dev/Vendr.jl/inversions/03_real/31_invert_A_gridded/fig_H_diff.png"
save(path, fig)