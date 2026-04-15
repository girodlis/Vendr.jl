module Vendr

# ODINN libraries
using Sleipnir
using Muninn
using Huginn
using ODINN


# Some utils libraries
using JLD2
using Revise
using Statistics
using CSV
using DataFrames
using TOML

include("setup/setup.jl")
include("campaign/campaign.jl")
include("plotting/plotting.jl")


end