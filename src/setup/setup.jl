export spinup_historical_forward_simulation!

using ODINN

"""
    spinup_historical_forward_simulation!(glaciers, params; spinup_tspan=(2009.0, 2010.0))

Run a short historical forward simulation to initialize glacier state
before inversion.
"""
function spinup_historical_forward_simulation!(
        glaciers,
        params;
        spinup_tspan = (2009.0, 2010.0),
    )
    spinup_params = Parameters(
        simulation = SimulationParameters(
            tspan = spinup_tspan,
            multiprocessing = params.simulation.multiprocessing,
            workers = params.simulation.workers,
            working_dir = params.simulation.working_dir,
            rgi_paths = params.simulation.rgi_paths,
            use_MB = params.simulation.use_MB,
        ),
    )

    model = Model(
        iceflow = SIA2Dmodel(params),
        mass_balance = TImodel1(params; DDF = 6.0 / 1000.0, acc_factor = 1.2 / 1000.0),
    )

    spinup_prediction = Prediction(model, glaciers, spinup_params)
    run!(spinup_prediction)

    for (glacier, result) in zip(glaciers, spinup_prediction.results)
        final_H = result.H[end]
        glacier.H₀ .= final_H
        glacier.S .= glacier.B .+ final_H

        if !isempty(result.V)
            glacier.V .= result.V[end]
            glacier.Vx .= result.Vx[end]
            glacier.Vy .= result.Vy[end]
        end
    end

    return glaciers, spinup_prediction
end
