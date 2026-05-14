using Revise
cd(@__DIR__)

using Vendr
using ODINN  # Already reexports Sleipnir

# Campaign root is the current directory (path of this script)
campaign_root = dirname(abspath(@__FILE__))

# Load campaign config and scenarios from campaign_root/config/
context = build_campaign_run_context(campaign_root)

run_campaign!(context)

print_summary(context.csv_file)

plot_relative_error_boxplot(
    csv_path = context.csv_file,
    output_path = joinpath(context.results_dir, "temperate_ice_absolute_relative_error_log_boxplot.png"),
    log_scale = true,
    use_absolute_error = true,
)
