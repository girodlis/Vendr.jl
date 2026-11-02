using JLD2
using Sleipnir
using Statistics
using DataFrames
using CSV

# Standalone: no `context`, no ODINN/Vendr needed. Reads the snapshot produced by
# save_snapshot.jl and computes, for every scenario x glacier x field (H, V), the
# mean absolute relative error (%) over the glacier's valid pixels
# (Sleipnir.is_in_glacier mask, eroded by 1 px).
#
# Writes a long-format CSV (one row per scenario/glacier/field) plus prints a wide
# pivot table (glaciers x scenarios) per field to the console.

results_dir = joinpath(dirname(abspath(@__FILE__)), "outputs", "results_10_glaciers")
snapshot_path = joinpath(results_dir, "campaign_snapshot.jld2")
isfile(snapshot_path) || error("No snapshot at $(snapshot_path). Run save_snapshot.jl first (needs a live campaign REPL).")

snap = JLD2.load(snapshot_path)
# Keep only the scenarios used in diff_grid_H_V_HV.png (S1=H no mask, S2=V no mask,
# S7=HV with H masked/V unmasked), so this summary stays consistent with that figure.
scenario_ids = ["S1", "S2", "S7"]
scenario_meta = filter(s -> s.id in scenario_ids, snap["scenario_meta"])
results_by_scenario = snap["results_by_scenario"]

function mean_abs_rel_error(results, field::Symbol)
    field_ref = Symbol(field, "_ref")
    pred = Float64.(getproperty(results, field)[end])
    gt   = Float64.(getproperty(results, field_ref)[end])
    mask = Sleipnir.is_in_glacier(Float64.(results.H_ref[end]), 1)
    valid = mask .& (gt .!= 0) .& isfinite.(pred) .& isfinite.(gt)
    any(valid) || return NaN
    rel_err = 100.0 .* abs.(pred[valid] .- gt[valid]) ./ abs.(gt[valid])
    return mean(rel_err)
end

rows = NamedTuple[]
for scenario in scenario_meta
    results_vec = results_by_scenario[scenario.id]
    for (gi, rgi_id) in enumerate(scenario.rgi_ids)
        results = results_vec[gi]
        for field in (:H, :V)
            push!(rows, (
                scenario = scenario.id,
                loss_type = scenario.loss_type,
                use_tim = scenario.use_tim,
                sparsity_H = scenario.sparsity_H,
                sparsity_V_sigma = scenario.sparsity_V_sigma,
                sparsity_V_threshold = scenario.sparsity_V_threshold,
                rgi_id = rgi_id,
                field = string(field),
                abs_relative_error_pct = mean_abs_rel_error(results, field),
            ))
        end
    end
end

df = DataFrame(rows)
csv_path = joinpath(results_dir, "abs_relative_error_summary_S1_S2_S7.csv")
CSV.write(csv_path, df)
println("Saved: ", csv_path)

for field in ("H", "V")
    sub = filter(:field => ==(field), df)
    pivoted = unstack(sub, :rgi_id, :scenario, :abs_relative_error_pct)
    println("\nAbs. relative error (%) — field $(field)")
    show(pivoted, allrows = true, allcols = true)
    println()
end
