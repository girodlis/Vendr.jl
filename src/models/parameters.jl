"""    build_parameters(campaign::CampaignConfig, scenario::ScenarioConfig) -> Sleipnir.Parameters

Construct full Parameters object with simulation, hyperparameter, physical,
UDE and solver settings.

# Arguments
- `campaign::CampaignConfig`: Campaign configuration with epoch and time settings
- `scenario::ScenarioConfig`: Scenario with loss type and masking settings

# Returns
`Sleipnir.Parameters` with all simulation and optimization parameters
"""
function build_parameters(campaign::CampaignConfig, scenario::ScenarioConfig)::Sleipnir.Parameters
    rgi_paths = get_rgi_paths()

    return Parameters(
        simulation = SimulationParameters(
            use_MB = scenario.use_tim,
            step_MB = 1.0,
            tspan = campaign.tspan,
            test_mode = false,
            multiprocessing = true,
            workers = 4,
            use_glathida_data = true, # /!\ Do a first run without downscale to download the grid and then run with downscale
            gridScalingFactor = campaign.gridScalingFactor,
            rgi_paths = rgi_paths,
        ),
        hyper = Hyperparameters(
            batch_size = length(scenario.rgi_ids),

            #### Scalar
            epochs = [campaign.epochs_adam, campaign.epochs_linesearch],
            optimizer = [
               ODINN.Adam(0.01),
               ODINN.LBFGS(linesearch = ODINN.LineSearches.BackTracking(iterations = 5)),
            ],

            #### Gridded
            # epochs = campaign.epochs_linesearch,
            # optimizer = ODINN.LBFGS(m = 400, linesearch = ODINN.LineSearches.StrongWolfe(c_2=0.1))
        ),
        physical = PhysicalParameters(
            minA = 1e-18, #8e-19,
            maxA = 3e-16, #8e-15,
        ),
        UDE = resolve_UDE_params(scenario),
        solver = Huginn.SolverParameters(
            step = campaign.step,
        ),
    )
end

"""    resolve_loss(scenario::ScenarioConfig) -> LossFunction

Construct the empirical loss function for one scenario.

# Arguments
- `scenario::ScenarioConfig`: Configuration specifying loss type and sparsity

# Returns
Loss function object (LossH, LossV, or LossHV)

# Throws
- `ErrorException`: If loss_type is not one of "H", "V", "HV"
"""
function resolve_loss(scenario::ScenarioConfig)::ODINN.AbstractLoss
    #h_distance = scenario.sparsity_H ? 0 : 3
    h_loss = LossH(loss = L2Sum(distance = 0))

    empirical = if scenario.loss_type == "H"
        h_loss
    elseif scenario.loss_type == "V"
        LossV()
    elseif scenario.loss_type == "HV"
        LossHV(hLoss = h_loss, vLoss = LossV())
    else
        error("Unsupported loss_type=$(scenario.loss_type).")
    end

    if scenario.regularization_weight == 0.0
        return empirical
    else
        return ODINN.MultiLoss(
            losses = (empirical, ODINN.RheologyRegularization(),),
            λs = (1.0, scenario.regularization_weight,)
        )
    end
end


function resolve_UDE_params(scenario::ScenarioConfig)
    if scenario.use_optim_autoAD
        return UDEparameters(
            optim_autoAD = ODINN.Optimization.AutoZygote(),
            grad = ODINN.SciMLSensitivityAdjoint(),
            empirical_loss_function = resolve_loss(scenario),
        )
    else
        return UDEparameters(
            optim_autoAD = ODINN.NoAD(),
            # Discrete for gridded
            #grad = DiscreteAdjoint(),
            empirical_loss_function = resolve_loss(scenario),
        )
    end
end

