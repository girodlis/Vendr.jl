using Revise
using ODINN
using Vendr
using Statistics

Core.eval(ODINN, :(get_mb_model(mb_model, glacier_idx) = mb_model))

rgi_paths = get_rgi_paths()
rgi_ids = [
  "RGI60-11.01450",
]

sim_params    = SimulationParameters(
    use_MB             = false,
    tspan              = (2017.0, 2018.0),
    multiprocessing    = false,
    gridScalingFactor  = 1,
    rgi_paths          = rgi_paths,
    ice_thickness_source = :Farinotti19,
    velocity_product   = :Millan22,
)
phys_params   = PhysicalParameters(minA = 1e-18, maxA = 1e-16)
solver_params = Huginn.SolverParameters(step = 1/208)

glacier_params = Sleipnir.Parameters(; physical = phys_params, simulation = sim_params)
glaciers = Sleipnir.initialize_glaciers(rgi_ids, glacier_params)

function build_inversion_for_loss(loss_fn; epochs = 1)
    params = Parameters(
        simulation = sim_params,
        hyper = Hyperparameters(
            batch_size = length(rgi_ids), epochs = epochs,
            optimizer = ODINN.LBFGS(m = 400, linesearch = ODINN.LineSearches.StrongWolfe(c_2 = 0.1)),
        ),
        physical = phys_params,
        UDE = UDEparameters(
            optim_autoAD = ODINN.NoAD(),
            empirical_loss_function = loss_fn,
            initial_condition_filter = :Zang1980,
        ),
        solver = solver_params,
    )
    trainable_model = GriddedInv(params, glaciers, :A)
    ic              = InitialCondition(params, glaciers, :Farinotti19)
    iceflow         = SIA2Dmodel(params; A = LawA(params; scalar = false))
    model = Model(iceflow = iceflow, mass_balance = nothing, regressors = (; A = trainable_model, IC = ic))
    return Inversion(model, glaciers, params)
end

full_loss = MultiLoss(
    losses = (
        LossAvgV(loss = LogSum(ϵ = 10.0), component = :abs),
        RheologyRegularization(),
        InitialThicknessRegularization(2017.0),
    ),
    λs = (1.0, 3e37, 1e-4),
)

println("========== Running 15 real LBFGS iterations (gridded-A config, to reach the plateau) ==========")
inv_full = build_inversion_for_loss(full_loss; epochs = 15)
run!(inv_full)
θ_after = inv_full.results.stats.θ
losses  = inv_full.results.stats.losses
println("loss[1] = ", losses[1], "   loss[end] = ", losses[end])

println()
println("========== LOSS DECOMPOSITION AT θ_after (post 15 LBFGS iterations) ==========")
total_after = ODINN.loss_iceflow_transient(θ_after, inv_full, map)
println("Total loss (3 terms, θ_after) : ", total_after)

terms = (
    ("LossAvgV (fit vitesse)",          LossAvgV(loss = LogSum(ϵ = 10.0), component = :abs), 1.0),
    ("RheologyRegularization",           RheologyRegularization(),                            3e37),
    ("InitialThicknessRegularization",   InitialThicknessRegularization(2017.0),              1e-4),
)

println()
println("---- Per-term breakdown at θ_after (raw value, then × λ used in run.jl) ----")
for (name, single_loss, λ) in terms
    inv_i = build_inversion_for_loss(MultiLoss(losses = (single_loss,), λs = (1.0,)))
    raw   = ODINN.loss_iceflow_transient(θ_after, inv_i, map)
    println(rpad(name, 32), " raw=", raw, "   ×λ=", λ, " -> ", raw * λ)
end

A_opt = Vendr._target_value(inv_full, 1, :A; θ = θ_after)
println()
println("A_opt extrema : ", extrema(A_opt), "   (bounds: ", phys_params.minA, " to ", phys_params.maxA, ")")
println("A_opt std/mean : ", std(vec(A_opt)) / (sum(A_opt)/length(A_opt)))
println("============================================================================")
