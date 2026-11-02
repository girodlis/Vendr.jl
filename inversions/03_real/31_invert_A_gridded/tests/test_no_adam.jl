using Revise
using CairoMakie
using SciMLSensitivity
using ODINN
using Vendr
using Random

cd(@__DIR__)

# Same monkey-patch as run.jl (get_mb_model was never defined upstream for the
# per-glacier MB path; we don't use per-glacier MB here, mass_balance = nothing).
Core.eval(ODINN, :(get_mb_model(mb_model, glacier_idx) = mb_model))

rgi_paths = get_rgi_paths()
rgi_ids = [
  "RGI60-11.01450",
]

# Same as run.jl except the optimizer: single-stage LBFGS(m=400, StrongWolfe(c_2=0.1))
# instead of the two-stage Adam(0.02)+LBFGS(BackTracking). This mirrors exactly what
# Vendr.build_parameters uses for a GRIDDED A law (models/parameters.jl:31-35) — the
# Adam+BackTracking pattern in run.jl is the one build_parameters reserves for SCALAR
# A laws only (models/parameters.jl:17-29), and its own comment there already flags
# Adam(0.02) as prone to a stable limit cycle even in that scalar case. run.jl builds
# its Parameters by hand (doesn't go through build_parameters), so it never got this
# gridded-vs-scalar distinction. Purpose: check whether the loss decreases
# monotonically with the "correct" optimizer, instead of the up-then-down excursion
# seen with epochs = [300, 30] + Adam (loss 0.167 -> 0.394 over Adam, back down to
# 0.167 over 4 L-BFGS iterations).
params = Parameters(
    simulation = SimulationParameters(
        use_MB             = false,
        tspan              = (2017.0, 2018.0),
        multiprocessing    = false,
        gridScalingFactor  = 1,
        rgi_paths          = rgi_paths,
        ice_thickness_source = :Farinotti19,
        velocity_product   = :Millan22,
    ),
    hyper = Hyperparameters(
        batch_size = length(rgi_ids),
        epochs     = 300,
        optimizer  = ODINN.LBFGS(m = 400, linesearch = ODINN.LineSearches.StrongWolfe(c_2 = 0.1)),
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
            λs = (1.0, 3e37, 1e-4),
        ),
        initial_condition_filter = :Zang1980,
    ),
    solver = Huginn.SolverParameters(step = 1/208),
)

glacier_params = Sleipnir.Parameters(; physical = params.physical, simulation = params.simulation)
glaciers = Sleipnir.initialize_glaciers(rgi_ids, glacier_params)

trainable_model = GriddedInv(params, glaciers, :A)
ic              = InitialCondition(params, glaciers, :Farinotti19)
iceflow         = SIA2Dmodel(params; A = LawA(params; scalar = false))

model = Model(
    iceflow      = iceflow,
    mass_balance = nothing,
    regressors   = (; A = trainable_model, IC = ic),
)

inversion = Inversion(model, glaciers, params)
run!(inversion)

losses = inversion.results.stats.losses
println()
println("========== NO-ADAM TEST RESULTS ==========")
println("N iterations recorded : ", length(losses))
println("loss[1]                : ", losses[1])
println("loss[end]               : ", losses[end])
println("min(loss) @ iter        : ", argmin(losses), " -> ", minimum(losses))
println("monotone non-increasing? ", all(diff(losses) .<= 1e-9))
println("relative change (1->end): ", round(100*(losses[1]-losses[end])/losses[1], digits=2), " %")
println()
println("Full loss trace:")
for (i, l) in enumerate(losses)
    println("  iter $i: $l")
end
println("============================================")
