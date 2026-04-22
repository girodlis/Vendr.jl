"""
    _target_value(simulation, glacier_idx::Int, target::Symbol; θ=nothing) -> Union{Array{Float64, 0}, Array{Float64, 2}}

Compute target law value (A or C parameter) for a glacier.

Automatically detects and handles both scalar laws (returning 0-D array) and
matrix laws (returning 2-D array) based on cache dimensionality.

# Arguments
- `simulation`: Simulation object with `model.iceflow` containing law definitions
- `glacier_idx::Int`: Glacier index (1-based) in glaciers list
- `target::Symbol`: Parameter to evaluate (:A or :C)
- `θ=nothing`: Trained parameters; if nothing uses cache defaults

# Returns
Array value (scalar: 0-D, matrix: 2-D) representing the law value

# Throws
- `ErrorException`: If law or cache field does not exist
"""
function _target_value(
    simulation,
    glacier_idx::Int,
    target::Symbol;
    θ=nothing,
)::Union{Array{Float64, 0}, Array{Float64, 2}}
    law = getproperty(simulation.model.iceflow, target)
    
    # Create cache to identify dimensionality
    iceflow_cache = init_cache(simulation.model.iceflow, simulation, glacier_idx, θ)
    cache_value = getproperty(iceflow_cache, target).value
    
    # Detect if scalar or matrix
    is_scalar = ndims(cache_value) == 0
    #TODO: pas gestion scalair vs matrix mais use eval law quand inversion && pas scalaire
    
    if is_scalar && θ === nothing
        return cache_value
    else
        t_ref = first(simulation.parameters.simulation.tspan)
        input_values = generate_inputs(inputs(law), simulation, glacier_idx, t_ref)
        return eval_law(law, simulation, glacier_idx, input_values, θ)
    end
end

"""
    _valid_domain_mask(glacier, values::AbstractArray) -> BitMatrix

Compute mask for valid domain cells in glacier grid.

Matches grid size to glacier domain. Handles both full grids and interior grids
(common in staggered finite difference schemes).

# Returns
Boolean mask where `true` indicates cells with positive ice thickness (H > 0)
"""
function _valid_domain_mask(glacier, values::AbstractArray)::BitMatrix
    h0 = glacier.H₀
    if size(values) == size(h0)
        return h0 .> 0.0
    end

    interior_h0 = h0[1:(end - 1), 1:(end - 1)]
    if size(values) == size(interior_h0)
        return interior_h0 .> 0.0
    end

    return trues(size(values))
end

"""
    _relative_error_percent(pred_value, gt_value, glacier) -> Float64

Compute percent relative error between prediction and ground truth.

Handles both scalar and spatial fields. For scalar values, returns simple percent error.
For spatial fields, returns mean error across valid domain cells (H > 0, finite values).

# Arguments
- `pred_value`: Predicted value (Number or AbstractArray)
- `gt_value`: Ground truth value (Number or AbstractArray)
- `glacier`: Glacier object with H₀ (ice thickness) field

# Returns  
Percent relative error: 100 * (pred - gt) / gt

# Throws
- `ErrorException`: If gt_value contains zeros or no valid cells found
"""
function _relative_error_percent(
    pred_value,
    gt_value,
    glacier,
)::Float64
    if pred_value isa Number || gt_value isa Number ||
       (pred_value isa AbstractArray && length(pred_value) == 1) ||
       (gt_value isa AbstractArray && length(gt_value) == 1)

        pred_scalar = pred_value isa Number ? Float64(pred_value) : Float64(first(pred_value))
        gt_scalar = gt_value isa Number ? Float64(gt_value) : Float64(first(gt_value))
        gt_scalar == 0.0 && error(
            "Cannot compute relative error: ground truth value is zero."
        )
        rel_err = 100.0 * ((pred_scalar - gt_scalar) / gt_scalar)
        return rel_err
    end

    pred_arr = pred_value isa AbstractArray ? pred_value : Array([pred_value])
    gt_arr = gt_value isa AbstractArray ? gt_value : Array([gt_value])
    size(pred_arr) == size(gt_arr) || error("Shape mismatch for relative error: pred=$(size(pred_arr)) gt=$(size(gt_arr))")

    domain_mask = _valid_domain_mask(glacier, gt_arr)
    valid = domain_mask .& isfinite.(gt_arr) .& isfinite.(pred_arr) .& (gt_arr .!= 0.0)
    any(valid) || error(
        "No valid cells found to compute relative error."
    )

    rel = @. 100.0 * ((pred_arr[valid] - gt_arr[valid]) / gt_arr[valid])
    mean_err = mean(rel)
    return mean_err
end

"""
    compute_relative_error(
        inversion,
        prediction,
        glacier_idx::Int;
        θ=nothing,
        target::Symbol=:A,
    ) -> Float64

Compute percent relative error of inverted parameter versus ground truth.

Compares the inversion result (with trained parameters θ) to the ground truth
prediction across all valid domain cells. Automatically handles scalar and matrix laws.

# Arguments
- `inversion`: Completed inversion with `results.stats.θ` trained parameters
- `prediction`: Ground truth forward simulation
- `glacier_idx::Int`: Glacier index to evaluate
- `θ=nothing`: Trained parameters (if nothing, uses inversion.results.stats.θ)
- `target::Symbol`: Parameter (:A or :C)

# Returns
Percent relative error, or scalar value for single-value laws

# Throws
- `ErrorException`: If θ is not found or no valid cells exist
"""
function compute_relative_error(
    inversion,
    prediction,
    glacier_idx::Int;
    θ=nothing,
    target::Symbol=:A,
)::Float64
    θ_value = isnothing(θ) ? inversion.results.stats.θ : θ
    isnothing(θ_value) && error("No trained parameters (θ) found in inversion.results.stats.θ")

    gt_value = _target_value(prediction, glacier_idx, target; θ=nothing)
    pred_value = _target_value(inversion, glacier_idx, target; θ=θ_value)
    glacier = prediction.glaciers[glacier_idx]

    return _relative_error_percent(pred_value, gt_value, glacier)
end