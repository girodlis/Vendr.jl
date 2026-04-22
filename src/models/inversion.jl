"""
    build_inversion_model(
        campaign::CampaignConfig,
        scenario::ScenarioConfig,
        params::Sleipnir.Parameters,
        glaciers,
    ) -> Model

Create trainable inversion model for parameter estimation.

Constructs a neural network-based inversion model for the specified parameter
(A or C). Automatically chooses between scalar (GlacierWideInv) and gridded
(GriddedInv) model types based on campaign.law_A configuration.

# Arguments
- `campaign::CampaignConfig`: Campaign with invert_target and law_A
- `scenario::ScenarioConfig`: Scenario configuration
- `params::Sleipnir.Parameters`: Simulation parameters
- `glaciers`: Glacier objects for parameter dimension calculation

# Returns
Model object ready for training via ODINN.jl

# See Also
- [`resolve_law_A`](@ref): Determine law type (scalar vs gridded)
"""
function build_inversion_model(
    campaign::CampaignConfig,
    scenario::ScenarioConfig,
    params::Sleipnir.Parameters,
    glaciers,
)
    _ = scenario
    A_law = resolve_law_A(campaign, params; mode = :inversion)
    trainable_model = _resolve_inversion_trainable_model(campaign, params, glaciers)

    iceflow = SIA2Dmodel(params; A = A_law)

    MB_model = scenario.use_tim ?
        mass_balance = TImodel1(params; DDF = 6.0/1000.0, acc_factor = 1.2/1000.0) :
        nothing

    return Model(
        iceflow = iceflow,
        mass_balance = MB_model,
        regressors = (; A = trainable_model), #TODO: adapt for C inversion
    )
end

