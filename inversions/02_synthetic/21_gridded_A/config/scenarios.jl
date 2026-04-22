loss_types = ["H", "V"]
use_tim_values = [false, true]
sparsity_H_values = [false]
use_optim_autoAD_values = [true, false] 


sparsity_V_levels = [
    (0.0, 0.0),   # no sparsity
]

out = "config/scenarios.toml"

open(out, "w") do io
    println(io, "# Auto-generated scenarios")
    println(io)

    sid = 1
    for loss_type in loss_types
        for use_tim in use_tim_values
            for sparsity_H in sparsity_H_values
                for (sigma, threshold) in sparsity_V_levels
                    for use_optim_autoAD in use_optim_autoAD_values
                        println(io, "[[scenarios]]")
                        println(io, "id = \"S$(sid)\"")
                        println(io, "loss_type = \"$(loss_type)\"")
                        println(io, "use_tim = $(use_tim)")
                        println(io, "sparsity_H = $(sparsity_H)")
                        println(io, "sparsity_V_sigma = $(sigma)")
                        println(io, "sparsity_V_threshold = $(threshold)")
                        println(io, "use_optim_autoAD = $(use_optim_autoAD)")
                        println(io)
                        sid += 1
                    end
                end
            end
        end
    end
end