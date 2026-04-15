export plot_losses, plot_relative_error_boxplot, plot_thickness_differences_grid, plot_velocity_differences_grid

using Plots
using CSV
using DataFrames
using CairoMakie

# Loss plotting (candidate for future migration to Sleipnir)
include("loss.jl")

# Campaign-level reporting plots (summary + grids)
include("summary_grid.jl")
