using Revise
cd(@__DIR__)

using Vendr
using ODINN
using CairoMakie

# Campaign root is the current directory (path of this script)
campaign_root = dirname(abspath(@__FILE__))

# Load campaign config and scenarios from campaign_root/config/
context = build_campaign_run_context(campaign_root)

run_campaign!(context)

save_epoch_counts_csv(context)

save_comparison_grids!(
    rgi_ids = context.campaign.rgi_ids,
    results_dir = context.results_dir,
    scenarios = context.scenarios,
    scenario_inversions = context.scenario_inversions,
    scenario_predictions = context.scenario_predictions,
)

print_summary(context.csv_file)

plot_relative_error_boxplot(
    csv_path = context.csv_file,
    output_path = joinpath(context.results_dir, "temperate_ice_absolute_relative_error_log_boxplot.png"),
    log_scale = true,
    use_absolute_error = true,
)

# ============ Plots Ground Truth + All Scenarios ============
# Labels adaptés pour mettre en avant la régularisation
scenario_labels = [
    if scenario.regularization_weight > 0.0
        "$(scenario_label(1, scenario)) | λ=$(scenario.regularization_weight)"
    else
        scenario_label(1, scenario)  # Fallback au label standard
    end
    for (i, scenario) in enumerate(context.scenarios)
]

for (glacier_idx, rgi_id) in enumerate(context.campaign.rgi_ids)
    fig = plot_all_scenarios_with_ground_truth(
        context.scenario_inversions,
        context.scenario_predictions,
        glacier_idx,
        :A,
        scenario_labels;
        plotContour = true,
        colormap = :YlGnBu
    )
    save(joinpath(context.results_dir, "all_scenarios_A_$(rgi_id).png"), fig)
    println("Saved: all_scenarios_A_$(rgi_id).png")
end