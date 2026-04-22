"""
    _format_scenario_label(label::AbstractString; max_len::Int = 34)

Compact and wrap long scenario labels for plot readability.
"""
function _format_scenario_label(label::AbstractString; max_len::Int = 34)
    str = String(label)
    if ncodeunits(str) <= max_len
        return str
    end

    parts = split(str, " | ")
    if length(parts) >= 3
        first_line = join(parts[1:2], " | ")
        second_line = join(parts[3:end], " | ")
        compact = string(first_line, "\n", second_line)
        return ncodeunits(compact) <= max_len + 12 ? compact : string(first(parts), "\n", last(parts))
    end

    return string(first(str, max_len - 1), "...")
end

function _build_scenario_tick_labels(labels::Vector{String})
    n = length(labels)
    if n <= 10
        return [_format_scenario_label(lbl; max_len = 38) for lbl in labels], nothing
    end

    short = ["S$(i)" for i in eachindex(labels)]
    legend_lines = ["S$(i): $(labels[i])" for i in eachindex(labels)]
    return short, join(legend_lines, "\n")
end

function _scenario_panel_text(label::AbstractString)
    compact = _format_scenario_label(label; max_len = 48)
    parts = split(compact, " | ")
    if length(parts) <= 1
        return compact
    end
    return string(parts[1], "\n", join(parts[2:end], " | "))
end

function _max_abs_finite(values)
    limit = 0.0
    found_finite = false
    for value in values
        isfinite(value) || continue
        found_finite = true
        limit = max(limit, abs(value))
    end
    return limit, found_finite
end

function _pick_state(data, idx, context::AbstractString)
    data isa AbstractVector || return data

    n = length(data)
    n > 0 || error("Encountered an empty time series in $(context)")

    selected_idx = isnothing(idx) ? n : idx
    1 <= selected_idx <= n || error("timeIdx=$(selected_idx) is out of bounds for a series of length $(n) in $(context)")
    return data[selected_idx]
end

function _thickness_reference(results::Sleipnir.Results, timeIdx::Union{Nothing, Int64}; prefer_unmasked::Bool = true)
    _ = prefer_unmasked
    return _pick_state(results.H_ref, timeIdx, "plot_thickness_differences_grid")
end
