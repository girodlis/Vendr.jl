using Revise
cd(@__DIR__)

using Vendr
using ODINN

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