"""
    apply_masks(scenario::ScenarioConfig, glaciers) -> Tuple

Apply observational data masks (thickness and velocity) to glaciers.

Masks sparsify glaciers' observational data according to scenario configuration.
Saves unmasked references for restoration after inversion.

# Arguments
- `scenario::ScenarioConfig`: Scenario with sparsity settings
- `glaciers`: Glacier objects with velocity and thickness data

# Returns
Tuple (glaciers, unmasked_references) where references are NamedTuple or nothing
"""
function apply_masks(scenario::ScenarioConfig, glaciers)::Tuple
    unmasked_velocity_ref = Dict{String, Any}()
    unmasked_thickness_ref = Dict{String, Any}()
    applied_velocity_mask = false
    applied_thickness_mask = false

    if scenario.sparsity_V_sigma > 0.0 && scenario.sparsity_V_threshold > 0.0
        for glacier in glaciers
            vdata = glacier.velocityData
            if isnothing(vdata)
                continue
            end
            unmasked_velocity_ref[glacier.rgi_id] = (
                vabs = deepcopy(vdata.vabs),
                vx = deepcopy(vdata.vx),
                vy = deepcopy(vdata.vy),
            )
            ntstops = length(vdata.vabs)
            base_mask = .!glacier.mask
            vel_mask = Sleipnir.random_spatially_coherent_mask(base_mask;
                sigma = scenario.sparsity_V_sigma,
                threshold = scenario.sparsity_V_threshold,
            )
            vel_mask_f = Float64.(vel_mask)
            for it in 1:ntstops
                glacier.velocityData.vx[it] .*= vel_mask_f
                glacier.velocityData.vy[it] .*= vel_mask_f
                glacier.velocityData.vabs[it] .*= vel_mask_f
            end
            applied_velocity_mask = true
        end
    end

    if scenario.sparsity_H
        for glacier in glaciers
            hdata = glacier.thicknessData
            isnothing(hdata) && continue
            unmasked_thickness_ref[glacier.rgi_id] = deepcopy(hdata.H)
            ntstops = length(hdata.H)
            glathida_mask = glacier.H_glathida .> 0.0
            for it in 1:ntstops
                glacier.thicknessData.H[it] .*= glathida_mask
            end
            applied_thickness_mask = true
        end
    end

    if !applied_velocity_mask && !applied_thickness_mask
        return glaciers, nothing
    end

    unmasked_references = (
        thickness = applied_thickness_mask ? unmasked_thickness_ref : nothing,
        velocity = applied_velocity_mask ? unmasked_velocity_ref : nothing,
    )

    return glaciers, unmasked_references
end

"""
    prepare_scenario(campaign::CampaignConfig, scenario::ScenarioConfig)

Prepare all components needed for one scenario run.
"""
function prepare_scenario(campaign::CampaignConfig, scenario::ScenarioConfig)
    params = build_parameters(campaign, scenario)
    prediction = build_ground_truth(campaign, scenario, params)
    glaciers = prediction.glaciers
    glaciers, unmasked_references = apply_masks(scenario, glaciers)
    inv_model = build_inversion_model(campaign, scenario, params, glaciers)
    return inv_model, glaciers, params, unmasked_references, prediction
end

