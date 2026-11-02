using Revise
using ODINN
using Vendr

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

full_loss = MultiLoss(
    losses = (
        LossAvgV(loss = LogSum(ϵ = 10.0), component = :abs),
        RheologyRegularization(),
        InitialThicknessRegularization(2017.0),
    ),
    λs = (1.0, 3e37, 1e-4),
)

params = Parameters(
    simulation = sim_params,
    hyper = Hyperparameters(
        batch_size = length(rgi_ids), epochs = 1,
        optimizer = ODINN.LBFGS(m = 400, linesearch = ODINN.LineSearches.StrongWolfe(c_2 = 0.1)),
    ),
    physical = phys_params,
    UDE = UDEparameters(
        optim_autoAD = ODINN.NoAD(),
        empirical_loss_function = full_loss,
        initial_condition_filter = :Zang1980,
    ),
    solver = solver_params,
)
trainable_model = GriddedInv(params, glaciers, :A)
ic              = InitialCondition(params, glaciers, :Farinotti19)
iceflow         = SIA2Dmodel(params; A = LawA(params; scalar = false))
model = Model(iceflow = iceflow, mass_balance = nothing, regressors = (; A = trainable_model, IC = ic))
inv = Inversion(model, glaciers, params)

θ0 = deepcopy(model.trainable_components.θ)

println("========== 1 LBFGS iteration, just to get a real adjoint gradient at θ0 ==========")
run!(inv)
println("loss[1] = ", inv.results.stats.losses[1], "   (expect 0.16713503275518585 == loss at θ0)")

g0  = inv.results.stats.∇θ_hist[1]
gid = only(keys(θ0.A))

L(θ) = ODINN.loss_iceflow_transient(θ, inv, map)

A_grad  = g0.A[gid]
IC_grad = g0.IC[gid]
iA  = argmax(vec(abs.(A_grad)))
iIC = argmax(vec(abs.(IC_grad)))

println()
println("Chosen components (largest |adjoint gradient|):")
println("  A  idx=$iA   adjoint_grad=", vec(A_grad)[iA])
println("  IC idx=$iIC  adjoint_grad=", vec(IC_grad)[iIC])

function fd_check(θbase, block::Symbol, gid, flat_idx, ε)
    θp = deepcopy(θbase); θm = deepcopy(θbase)
    getproperty(θp, block)[gid][flat_idx] += ε
    getproperty(θm, block)[gid][flat_idx] -= ε
    Lp, Lm = L(θp), L(θm)
    return (Lp - Lm) / (2ε), Lp, Lm
end

println()
println("========== Finite-difference check (central diff) ==========")
fdA, LpA, LmA = fd_check(θ0, :A, gid, iA, 0.5)
println("A  : adjoint=", vec(A_grad)[iA], "   finite_diff=", fdA, "   L(+ε)=", LpA, "  L(-ε)=", LmA)

fdIC, LpIC, LmIC = fd_check(θ0, :IC, gid, iIC, 20.0)
println("IC : adjoint=", vec(IC_grad)[iIC], "   finite_diff=", fdIC, "   L(+ε)=", LpIC, "  L(-ε)=", LmIC)
println("============================================================================")
