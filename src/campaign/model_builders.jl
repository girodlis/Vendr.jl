"""
    campaign/model_builders.jl

Functions for constructing Parameters, laws, models, and related setup for inversion training.
"""

using ODINN, Huginn, Sleipnir

scenario_epochs_adam(scenario::ScenarioConfig, campaign::CampaignConfig) = 
    something(scenario.epochs_adam, campaign.epochs_adam)

scenario_epochs_linesearch(scenario::ScenarioConfig, campaign::CampaignConfig) = 
    something(scenario.epochs_linesearch, campaign.epochs_linesearch)

"""
    _resolve_ground_truth_A_law(campaign::CampaignConfig)

Instantiate the flow law for ground truth generation.
"""
function _resolve_ground_truth_A_law(campaign::CampaignConfig)
    if campaign.law_A == :TemperateA
        return Huginn.TemperateA()
    elseif campaign.law_A == :CuffeyPaterson
        return CuffeyPaterson(scalar = true)
    else
        error("Unsupported law_A=$(campaign.law_A).")
    end
end

"""
    _resolve_fixed_C_law(campaign::CampaignConfig, params)

Instantiate the fixed basal friction law for ground truth generation.
"""
function _resolve_fixed_C_law(campaign::CampaignConfig, params)
    if campaign.law_C == :SyntheticC
        return SyntheticC(params)
    elseif campaign.law_C == :None
        return nothing
    else
        error("Unsupported law_C=$(campaign.law_C).")
    end
end

"""
    _resolve_loss(scenario::ScenarioConfig) -> Union{LossH, LossV, LossHV}

Construct the loss function for a scenario.

- "H": Thickness mismatch only; optionally sparse
- "V": Velocity mismatch only
- "HV": Combined thickness and velocity mismatch
"""
function _resolve_loss(scenario::ScenarioConfig)
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

"""
    _build_parameters(campaign::CampaignConfig, scenario::ScenarioConfig) -> Parameters

Construct full Parameters object with simulation, hyperparameter, physical, UDE, and solver settings.

Merges campaign-level defaults with scenario-specific overrides.
"""
#TODO: add the possibility to override on the run file
function _build_parameters(campaign::CampaignConfig, scenario::ScenarioConfig)
    rgi_paths = get_rgi_paths()
    epochs_adam = scenario_epochs_adam(scenario, campaign)
    epochs_linesearch = scenario_epochs_linesearch(scenario, campaign)

    return Parameters(
        simulation = SimulationParameters(
            use_MB = scenario.use_tim,
            # TODO: move tspan to campaign config and support transient runs with configurable tstops.
            tspan = (2010.0, 2011.0),
            test_mode = false,
            multiprocessing = true,
            workers = 4,
            use_glathida_data = true,
            rgi_paths = rgi_paths,
        ),
        hyper = Hyperparameters(
            batch_size = length(scenario.rgi_ids),
            epochs = [epochs_adam, epochs_linesearch],
            optimizer = [
                ODINN.Adam(0.01),
                ODINN.LBFGS(linesearch = ODINN.LineSearches.BackTracking(iterations = 5))
            ],
        ),
        physical = PhysicalParameters(
            minA = 8e-19,
            maxA = 8e-15,
        ),
        UDE = UDEparameters(
            optim_autoAD = ODINN.NoAD(), #TODO: possibly switch to AutoZygote 
            empirical_loss_function = _resolve_loss(scenario),
        ),
        solver = Huginn.SolverParameters(
            step = 2.0, #TODO: possibly switch to transient
        ),
    )
end

"""
    _spinup_if_needed(glaciers, params, campaign::CampaignConfig) -> Tuple{Vector{Glacier}, Union{Nothing, Any}}

Run spinup historical forward simulation if enabled in campaign config.
"""
function _spinup_if_needed(glaciers, params, campaign::CampaignConfig)
    if campaign.spinup
        return spinup_historical_forward_simulation!(glaciers, params; spinup_tspan = campaign.spinup_tspan)
    end
    return glaciers, nothing
end

"""
    _build_ground_truth_prediction(campaign::CampaignConfig, scenario::ScenarioConfig, params)

Generate reference ground truth by forward simulation.

Includes both A and C laws (if specified in campaign), even though inversion only optimizes A.
"""
function _build_ground_truth_prediction(campaign::CampaignConfig, scenario::ScenarioConfig, params)
    validate_campaign(campaign)

    glaciers = initialize_glaciers(scenario.rgi_ids, params)
    glaciers, _ = _spinup_if_needed(glaciers, params, campaign)

    A_gt = _resolve_ground_truth_A_law(campaign)
    C_gt = _resolve_fixed_C_law(campaign, params)

    truth_model = isnothing(C_gt) ?
        Model(
            iceflow = SIA2Dmodel(params; A = A_gt),
            mass_balance = TImodel1(params; DDF = 6.0 / 1000.0, acc_factor = 1.2 / 1000.0),
        ) :
        Model(
            iceflow = SIA2Dmodel(params; A = A_gt, C = C_gt),
            mass_balance = TImodel1(params; DDF = 6.0 / 1000.0, acc_factor = 1.2 / 1000.0),
        )

    # TODO: read tstops from config for transient runs instead of fixed two-step synthetic setup.
    tstops = collect(2010:1.0:2011)
    return generate_ground_truth_prediction(glaciers, params, truth_model, tstops)
end

"""
    _build_inversion_model(campaign::CampaignConfig, scenario::ScenarioConfig, params, glaciers) -> Model

Construct the trainable inversion model.

Currently uses A-only inversion to avoid Mooncake AD issues with SyntheticC.
C is kept in ground truth generation but not in the inversion.
"""
function _build_inversion_model(campaign::CampaignConfig, scenario::ScenarioConfig, params, glaciers)
    # TODO: select inversion target from campaign.invert_target and branch A/C model+regressor creation.
    A_law = LawA(params; scalar = true) #TODO: non scalar A
    trainable_model = GlacierWideInv(params, glaciers, :A) #TODO: trainable C, gridded A/C

    iceflow = SIA2Dmodel(params; A = A_law)

    return Model(
        iceflow = iceflow,
        mass_balance = TImodel1(params; DDF = 6.0 / 1000.0, acc_factor = 1.2 / 1000.0),
        regressors = (; A = trainable_model), #TODO: trainable C
    )
end
