function restore_unmasked_references!(glacier_result, unmasked_references)
    isnothing(unmasked_references) && return glacier_result

    rgi_id = glacier_result.rgi_id

    if !isnothing(unmasked_references.thickness) && haskey(unmasked_references.thickness, rgi_id)
        glacier_result.H_ref = deepcopy(unmasked_references.thickness[rgi_id])
    end

    if !isnothing(unmasked_references.velocity) && haskey(unmasked_references.velocity, rgi_id)
        velocity_ref = unmasked_references.velocity[rgi_id]
        glacier_result.V_ref = deepcopy(velocity_ref.vabs)
    end

    return glacier_result
end
