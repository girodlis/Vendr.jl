using Revise
cd(@__DIR__)

using Vendr

# Run this in the same REPL session where `run.jl` was `include`d, after
# `run_campaign!(context)` has finished — it reuses the live `context`
# (scenario_inversions/scenario_predictions) instead of recomputing anything.
#
# The per-glacier maps (thickness_differences_grid, velocity_norm_differences_grid,
# thickness/velocity_predictions_with_gt, target_differences_grid) are already
# generated automatically by run_campaign! -> save_comparison_grids!. This script
# only adds the synthesis views that need the full campaign context: the `A`
# error table and the faceted iteration curves.

@assert @isdefined(context) "`context` not found in this session: run `include(\"run.jl\")` first."
@assert length(context.scenario_inversions) == length(context.scenarios) "Campaign not finished yet: $(length(context.scenario_inversions))/$(length(context.scenarios)) scenarios completed."

# Table: A relative error (%), one row per glacier, one column per scenario.
# Reads campaign_summary.csv directly, so it also works offline / after the fact.
plot_A_relative_error_heatmap(context.csv_file)

# Iteration curves, faceted by loss_type (H / V / HV):
# - loss evolution (doubles as the H or V empirical error for pure-H / pure-V scenarios)
# - A relative error evolution, computed cheaply from θ_hist (no forward re-simulation)
plot_loss_curves_faceted(context)
plot_A_error_curves_faceted(context)

println("Analysis figures written to $(context.results_dir)")
