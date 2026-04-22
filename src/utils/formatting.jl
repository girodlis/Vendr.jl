"""
    scenario_banner(i::Int, n::Int, scenario::ScenarioConfig) -> String

Format a header string for scenario execution.
"""
function scenario_banner(i::Int, n::Int, scenario::ScenarioConfig)
    return "=== Scenario $(i)/$(n) : $(scenario.id) | loss=$(scenario.loss_type) tim=$(scenario.use_tim) H=$(scenario.sparsity_H) V=$(scenario.sparsity_V_sigma), $(scenario.sparsity_V_threshold)"
end

"""
    scenario_label(i::Int, scenario::ScenarioConfig) -> String

Format a human-readable label for a scenario (used in plots and tables).
"""
function scenario_label(i::Int, scenario::ScenarioConfig)
    return "$(scenario.id) | loss=$(scenario.loss_type) | tim=$(scenario.use_tim) | H=$(scenario.sparsity_H) | Vσ=$(scenario.sparsity_V_sigma) | Vthr=$(scenario.sparsity_V_threshold)"
end