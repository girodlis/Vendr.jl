function _build_spinup_parameters(params, spinup_tspan)
    params.simulation.tspan == spinup_tspan && return params

    return Parameters(
        simulation = SimulationParameters(
            tspan = spinup_tspan,
            multiprocessing = params.simulation.multiprocessing,
            workers = params.simulation.workers,
            working_dir = params.simulation.working_dir,
            rgi_paths = params.simulation.rgi_paths,
            use_MB = params.simulation.use_MB,
            gridScalingFactor = params.simulation.gridScalingFactor,
        ),
    )
end

"""
    build_ground_truth(
        campaign::CampaignConfig,
        scenario::ScenarioConfig,
        params::Sleipnir.Parameters,
    ) -> Simulation

Generate reference (ground truth) forward simulation for comparison.

Runs a forward simulation using specified flow laws (law_A, law_C) to create
ground truth targets for inversion. Includes optional spinup phase.

# Arguments
- `campaign::CampaignConfig`: Campaign with law and time settings
- `scenario::ScenarioConfig`: Scenario configuration
- `params::Sleipnir.Parameters`: Simulation parameters

# Returns
Completed forward simulation with glacier states and results

# See Also
- [`spinup_glacier_state!`](@ref): Pre-spinup phase
"""
function build_ground_truth(
    campaign::CampaignConfig,
    scenario::ScenarioConfig,
    params::Sleipnir.Parameters,
)

    if campaign.spinup
        spinup_params = _build_spinup_parameters(params, campaign.spinup_tspan)
        spinup_glaciers = initialize_glaciers(scenario.rgi_ids, spinup_params)

        spinup_MB_model = scenario.use_tim ?
            mass_balance = TImodel1(
                spinup_params; DDF = 6.0/1000.0,
                acc_factor = 1.2/1000.0
            ) :
            nothing

        spinup_model = Model(
            iceflow = SIA2Dmodel(spinup_params),
            mass_balance = spinup_MB_model,
        )
        
        # spinup generation with monthly timestep
        spinup_prediction = Prediction(spinup_model, spinup_glaciers, spinup_params)
        run!(spinup_prediction)

        glaciers = initialize_glaciers(scenario.rgi_ids, params)
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
    else
        glaciers = initialize_glaciers(scenario.rgi_ids, params)
    end

    A_gt = resolve_law_A(campaign, params; mode = :ground_truth)
    C_gt = resolve_law_C(campaign, params)

     MB_model = scenario.use_tim ?
        mass_balance = TImodel1(
            params; DDF = 6.0/1000.0,
            acc_factor = 1.2/1000.0
        ) :
        nothing
        
    truth_model = isnothing(C_gt) ?
        Model(
            iceflow = SIA2Dmodel(params; A = A_gt),
            mass_balance = MB_model,
        ) :
        Model(
            iceflow = SIA2Dmodel(params; A = A_gt, C = C_gt),
            mass_balance = MB_model,
        )

    t0, t1 = campaign.tspan
    gt_solver_step = something(campaign.step_gt, campaign.step)
    gt_tstops = collect(range(t0, t1, step = gt_solver_step))
    t1 ∉ gt_tstops && push!(gt_tstops, t1)

    # Temporarily use step_gt for GT simulation
    original_step = params.solver.step
    try
        params.solver.step = gt_solver_step
        return generate_ground_truth_prediction(glaciers, params, truth_model, gt_tstops)
    finally
        params.solver.step = original_step
    end
end

