# Generate scenarios in config/scenarios.toml
cd(@__DIR__)

loss_types = ["H", "V", "HV"]
use_tim_values = [false]
sparsity_H_values = [false, true]
sparsity_V_levels = [
    (0.0, 0.0),   # no sparsity
    (0.7, 0.02),  # (sigma, threshold)
]

#regularization_weight = 5.29e37 # H best
#regularization_weight = range(1.29e38, 2.15e38; length=10) # V test 1
#regularization_weight = 1.38e38 # V best
regularization_weight = 0.0
out = "scenarios.toml"

open(out, "w") do io
    println(io, "# Auto-generated scenarios")
    println(io)

    sid = 1
    for use_tim in use_tim_values
        for sparsity_H in sparsity_H_values
            for (sigma, threshold) in sparsity_V_levels
                for reg_weight in regularization_weight
                    for loss_type in loss_types
                        println(io, "[[scenarios]]")
                        println(io, "id = \"S$(sid)\"")
                        println(io, "loss_type = \"$(loss_type)\"")
                        println(io, "use_tim = $(use_tim)")
                        println(io, "sparsity_H = $(sparsity_H)")
                        println(io, "sparsity_V_sigma = $(sigma)")
                        println(io, "sparsity_V_threshold = $(threshold)")
                        println(io, "regularization_weight = $(reg_weight)")
                        println(io)
                        sid += 1
                    end
                end
            end
        end
    end
end