using Revise
using ODINN
using Vendr

# Same monkey-patch as run.jl.
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

# Builds a fresh Inversion (own GriddedInv/IC θ, both initialized to their constant
# starting value — Cuffey-Paterson A, Farinotti19 H₀ — same as run.jl before any
# optimizer step) for a given empirical_loss_function, WITHOUT calling run! (no
# optimization happens: we just want to evaluate the loss once at θ0).
function build_inversion_for_loss(loss_fn)
    params = Parameters(
        simulation = sim_params,
        hyper = Hyperparameters(batch_size = length(rgi_ids), epochs = 1, optimizer = ODINN.LBFGS()),
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

println("========== LOSS DECOMPOSITION AT θ0 (before any optimizer step) ==========")

# 1) Full 3-term loss (same weights as run.jl) — sanity check: should reproduce
# losses[1] from the original run (0.16713503275518585).
full_loss = MultiLoss(
    losses = (
        LossAvgV(loss = LogSum(ϵ = 10.0), component = :abs),
        RheologyRegularization(),
        InitialThicknessRegularization(2017.0),
    ),
    λs = (1.0, 3e37, 1e-4),
)
inv_full = build_inversion_for_loss(full_loss)
θ0_full  = inv_full.model.trainable_components.θ
total0   = ODINN.loss_iceflow_transient(θ0_full, inv_full, map)
println("Total loss (3 terms, θ0) : ", total0, "   [run.jl reported losses[1] = 0.16713503275518585]")

# 2) Each term in isolation (λ=1.0, so we read the RAW unweighted value), then
# manually apply the real λ used in run.jl to see the actual weighted contribution.
terms = (
    ("LossAvgV (fit vitesse)",          LossAvgV(loss = LogSum(ϵ = 10.0), component = :abs), 1.0),
    ("RheologyRegularization",           RheologyRegularization(),                            3e37),
    ("InitialThicknessRegularization",   InitialThicknessRegularization(2017.0),              1e-4),
)

println()
println("---- Per-term breakdown (raw value, then × λ used in run.jl) ----")
for (name, single_loss, λ) in terms
    inv_i = build_inversion_for_loss(MultiLoss(losses = (single_loss,), λs = (1.0,)))
    θi    = inv_i.model.trainable_components.θ
    raw   = ODINN.loss_iceflow_transient(θi, inv_i, map)
    println(rpad(name, 32), " raw=", raw, "   ×λ=", λ, " -> ", raw * λ)
end
println("============================================================================")
