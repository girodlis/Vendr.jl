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
            tspan = campaign.tspan,
            test_mode = false,
            multiprocessing = true,
            workers = 4,
            use_glathida_data = true,
            #gridScalingFactor = 4,
            rgi_paths = rgi_paths,
        ),
        hyper = Hyperparameters(
            batch_size = length(scenario.rgi_ids),
            epochs = [campaign.epochs_adam, campaign.epochs_linesearch],
            optimizer = [
                ODINN.Adam(0.01),
                ODINN.LBFGS(linesearch = ODINN.LineSearches.BackTracking(iterations = 5)),
            ],
        ),
        physical = PhysicalParameters(
            minA = 8e-19,
            maxA = 8e-15,
        ),
        UDE = resolve_autoAD(scenario),
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
function resolve_loss(scenario::ScenarioConfig)::Union{LossH, LossV, LossHV}
    h_distance = scenario.sparsity_H ? 0 : 3
    h_loss = LossH(loss = L2Sum(distance = h_distance))

    if scenario.loss_type == "H"
        return h_loss
    elseif scenario.loss_type == "V"
        return LossV()
    elseif scenario.loss_type == "HV"
        return LossHV(hLoss = h_loss, vLoss = LossV())
    else
        error("Unsupported loss_type=$(scenario.loss_type).")
    end
end

function resolve_autoAD(scenario::ScenarioConfig)
    if scenario.use_optim_autoAD
        return UDEparameters(
            # ADTypes.AutoEnzyme()
            empirical_loss_function = resolve_loss(scenario),
        )
    else
        return UDEparameters(
            optim_autoAD = ODINN.NoAD(),
            empirical_loss_function = resolve_loss(scenario),
        )
    end
end

