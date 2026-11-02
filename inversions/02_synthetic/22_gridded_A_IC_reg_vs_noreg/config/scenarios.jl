# Generate config/scenarios.toml

# Single setup (loss = V, dense) with vs. without rheology regularization
# on A, to compare "A inverted without regularization" vs. "A inverted
# with regularization" against the same ground truth.
scenarios = [
    ("S1_noreg", 0.0),
    ("S2_reg", 3e37),
]

out = joinpath(@__DIR__, "scenarios.toml")

open(out, "w") do io
    println(io, "# Generated scenarios: V loss (dense), reg. vs. no reg. on A")
    println(io)
    for (id, reg_weight) in scenarios
        println(io, "[[scenarios]]")
        println(io, "id = \"$(id)\"")
        println(io, "loss_type = \"V\"")
        println(io, "use_tim = false")
        println(io, "sparsity_H = false")
        println(io, "sparsity_V_sigma = 0.0")
        println(io, "sparsity_V_threshold = 0.0")
        println(io, "regularization_weight = $(reg_weight)")
        println(io)
    end
end
