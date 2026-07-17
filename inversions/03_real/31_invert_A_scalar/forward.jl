using ODINN

# Define the working directory
#working_dir = joinpath(ODINN.root_dir, "demos")

# Ensure the working directory exists
#mkpath(working_dir)

# Define which glacier RGI IDs we want to work with
rgi_ids = ["RGI60-11.02773"]
rgi_paths = get_rgi_paths()

# Create the necessary parameters
params = Parameters(
    simulation = SimulationParameters(
    working_dir = working_dir,
    tspan = (2017.0, 2018.0),
    multiprocessing = false,
    #workers = 4,
    rgi_paths = rgi_paths,
    ice_thickness_source = :Millan22,
)
)

#A_law = TemperateA()
A_law = ConstantA(7.6e-17)

model = Model(
    iceflow = SIA2Dmodel(params; A = A_law),
    mass_balance = TImodel1(params; DDF = 6.0 / 1000.0, acc_factor = 1.2 / 1000.0)
)

glaciers = initialize_glaciers(rgi_ids, params)

prediction = Prediction(model, glaciers, params)

run!(prediction)

# Then we can visualize the results of the simulation, e.g. the difference in ice thickness
# between 2017 and 2018
#plot_glacier(prediction.results[1], "evolution difference", [:H]; metrics = ["difference"])

fig = plot_glacier(prediction.results[1], "heatmaps", [:V]; timeIdx = 12)
path = "/Users/girodli/.julia/dev/Vendr.jl/inversions/03_real/31_invert_A_scalar/fig_test.png"
save(path, fig)