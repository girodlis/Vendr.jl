using JLD2

# Run this ONCE in the live REPL (right after run_campaign!, while `context` still
# holds scenario_inversions/scenario_predictions) before killing the Julia session.
#
# It dumps just what post-hoc plotting scripts need — the per-glacier Sleipnir.Results
# (H, V, x, y, ...) for every scenario, plus plain scenario metadata — into a single
# JLD2 file. Sleipnir.Results is a plain struct of arrays/numbers/strings, so it
# serializes cleanly; we deliberately don't dump the full ODINN Inversion objects
# (optimizer state, autodiff caches, ...), which are heavier and unnecessary for
# plotting.
#
# Once this file exists, plotting scripts (see diff_grid_H_V_HV.jl) can load it in a
# fresh, lightweight Julia session — `using Sleipnir, CairoMakie, JLD2` — with no
# need to `using ODINN` or re-run the campaign.

snapshot_path = joinpath(context.results_dir, "campaign_snapshot.jld2")

scenario_meta = [
    (id = s.id, loss_type = s.loss_type, use_tim = s.use_tim,
     sparsity_H = s.sparsity_H, sparsity_V_sigma = s.sparsity_V_sigma,
     sparsity_V_threshold = s.sparsity_V_threshold, rgi_ids = s.rgi_ids)
    for s in context.scenarios
]

results_by_scenario = Dict(
    s.id => inversion.results.simulation
    for (s, inversion) in zip(context.scenarios, context.scenario_inversions)
)

JLD2.jldsave(snapshot_path; scenario_meta = scenario_meta, results_by_scenario = results_by_scenario)

println("Saved snapshot: ", snapshot_path)
println("You can now safely kill this Julia session — future plots only need:")
println("  using Sleipnir, CairoMakie, JLD2")
