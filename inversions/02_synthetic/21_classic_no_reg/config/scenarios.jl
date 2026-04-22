# Generate 24 scenarios in config/scenarios.toml
# 3 (loss_type) x 2 (use_tim) x 2 (sparsity_H) x 2 (sparsity_V)

loss_types = ["H", "V", "HV"]
use_tim_values = [false, true]
sparsity_H_values = [false, true]

sparsity_V_levels = [
    (0.0, 0.0),   # no sparsity
    (0.7, 0.02),  # (sigma, threshold)
]

out = "config/scenarios.toml"

open(out, "w") do io
    println(io, "# Auto-generated 24 scenarios")
    println(io)

    sid = 1
    for use_tim in use_tim_values
        for sparsity_H in sparsity_H_values
            for (sigma, threshold) in sparsity_V_levels
                for loss_type in loss_types
                    println(io, "[[scenarios]]")
                    println(io, "id = \"S$(sid)\"")
                    println(io, "loss_type = \"$(loss_type)\"")
                    println(io, "use_tim = $(use_tim)")
                    println(io, "sparsity_H = $(sparsity_H)")
                    println(io, "sparsity_V_sigma = $(sigma)")
                    println(io, "sparsity_V_threshold = $(threshold)")
                    println(io)
                    sid += 1
                end
            end
        end
    end
end