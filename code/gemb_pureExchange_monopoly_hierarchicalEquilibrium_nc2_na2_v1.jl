using GeneralEquilibriumModeling
using Optim

s1 = 200.0
s2 = 100.0

# u(x1, x2) = (x1^(-1) + x2^(-1))^(-1)
const preference = GEMB.CESSpec(
    [1.0, 1.0];
    es=0.5,
)

function solve_competitive_equilibrium(q, s2)
    model = GEMB.GEMBModel(
        [
            GEMB.CommoditySpec(
                :commodity_1;
                price_lower_bound=1.0e-8,
            ),
            :commodity_2,
        ];
        numeraire=:commodity_2,
    )

    GEMB.add_agent!(
        model,
        preference;
        demands=[:commodity_1, :commodity_2],
        endowments=:commodity_1,
        endowment_quantities=q,
        activity_start=100.0,
        name=:consumer_1,
    )

    GEMB.add_agent!(
        model,
        preference;
        demands=[:commodity_1, :commodity_2],
        endowments=:commodity_2,
        endowment_quantities=s2,
        activity_start=100.0,
        name=:consumer_2,
    )

    result = GEMB.solve(
        model;
        silent=true,
    )

    u1 = result.agent_variable_values[1][1]
    u2 = result.agent_variable_values[2][1]

    return (
        model=model,
        result=result,
        u1=u1,
        u2=u2,
    )
end

function monopoly_objective(q)
    equilibrium = solve_competitive_equilibrium(q, s2)
    return -equilibrium.u1
end

# Prevent inner equilibrium breakdown at q = 0
q_lower = 0.01 * s1

optimization = optimize(
    monopoly_objective,
    q_lower,
    s1,
    Brent();
    rel_tol=1.0e-8,
    abs_tol=1.0e-8,
)

q_star = Optim.minimizer(optimization)

baseline = solve_competitive_equilibrium(s1, s2)
monopoly = solve_competitive_equilibrium(q_star, s2)

println("========== Baseline competitive equilibrium ==========")
println("Supply of consumer 1 = ", s1)
println("Prices = ", baseline.result.prices)
println("Utility of consumer 1 = ", baseline.u1)
println("Utility of consumer 2 = ", baseline.u2)

println("\n========== Monopoly equilibrium ==========")
println("Optimal supply of consumer 1 = ", q_star)
println("Prices = ", monopoly.result.prices)
println("Utility of consumer 1 = ", monopoly.u1)
println("Utility of consumer 2 = ", monopoly.u2)
println("Outer optimization converged = ", Optim.converged(optimization))

GEMB.print_equilibrium_statistics(
    monopoly.model,
    monopoly.result,
)