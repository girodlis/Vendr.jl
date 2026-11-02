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

println("========== Running 5 real LBFGS iterations (plateau already reached by iter 2-3) ==========")
inv_full = build_inversion_for_loss(full_loss; epochs = 5)
run!(inv_full)
θ_after = inv_full.results.stats.θ
losses  = inv_full.results.stats.losses
println("loss[1] = ", losses[1], "   loss[end] = ", losses[end])

norm2(x) = sqrt(sum(abs2, x))
gid = only(keys(θ_after.A))

println()
println("========== Gradient at θ_after (plateau point) ==========")
# grad_loss_iceflow! uses raw Zygote.gradient internally, a different differentiation
# path than the ContinuousAdjoint/EnzymeVJP+Mooncake adjoint actually used by run!
# (see "Optimizing with custom ContinuousAdjoint..." in the log) — calling it directly
# crashes on something unrelated (NetCDF/CommonDataModel internals). Reuse the last
# gradient already computed by the real optimizer instead (∇θ_hist[end]), which is
# known-good (it's what LBFGS itself used).
g = inv_full.results.stats.∇θ_hist[end]
println("‖∇θ‖ total : ", norm2(g))
println("‖∇A‖  : ", norm2(g.A[gid]),  "   (‖A‖  = ", norm2(θ_after.A[gid]),  ")")
println("‖∇IC‖ : ", norm2(g.IC[gid]), "   (‖IC‖ = ", norm2(θ_after.IC[gid]), ")")

println()
println("========== Manual line search from θ_after along -∇θ (bypassing LBFGS's own line search) ==========")
L0 = ODINN.loss_iceflow_transient(θ_after, inv_full, map)
println("L(θ_after) = ", L0)
for α in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10)
    θ_try = θ_after .- α .* g
    L_try = ODINN.loss_iceflow_transient(θ_try, inv_full, map)
    println("  α=", α, "  L(θ_after - α∇θ) = ", L_try, "   ΔL = ", L_try - L0)
end
println("============================================================================")
