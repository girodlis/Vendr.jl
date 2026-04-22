"""
    resolve_law_A(campaign::CampaignConfig, params; mode::Symbol=:inversion)

Instantiate law A for inversion or ground-truth forward simulation.
"""
function resolve_law_A(campaign::CampaignConfig, params; mode::Symbol = :inversion)
    if mode == :ground_truth
        if campaign.law_A == :TemperateA
            return Huginn.TemperateA()
        elseif campaign.law_A == :CuffeyPatersonScalar
            return CuffeyPaterson(scalar = true)
        elseif campaign.law_A == :CuffeyPatersonGridded
            return CuffeyPaterson(scalar = false)
        else
            error("Unsupported law_A=$(campaign.law_A).")
        end
    end

    if campaign.law_A == :TemperateA || campaign.law_A == :CuffeyPatersonScalar
        return LawA(params; scalar = true)
    elseif campaign.law_A == :CuffeyPatersonGridded
        return LawA(params; scalar = false)
    else
        error("Unsupported law_A=$(campaign.law_A).")
    end
end

"""
    resolve_law_C(campaign::CampaignConfig, params)

Instantiate fixed law C for ground truth generation.
"""
function resolve_law_C(campaign::CampaignConfig, params)
    if campaign.law_C == :SyntheticC
        return SyntheticC(params)
    elseif campaign.law_C == :None
        return nothing
    else
        error("Unsupported law_C=$(campaign.law_C).")
    end
end

function _resolve_inversion_trainable_model(campaign::CampaignConfig, params, glaciers)
    if campaign.law_A == :TemperateA || campaign.law_A == :CuffeyPatersonScalar
        return GlacierWideInv(params, glaciers, :A)
    elseif campaign.law_A == :CuffeyPatersonGridded
        return GriddedInv(params, glaciers, :A)
    else
        error("Unsupported law_A=$(campaign.law_A).")
    end
end

