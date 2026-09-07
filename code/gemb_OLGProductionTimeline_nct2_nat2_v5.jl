using GeneralEquilibriumModeling
using Plots

ng = 50
np = ng + 2

alpha = 2.0
beta_firm = 0.2
beta_consumer = [0.1, 0.1, 0.8]

population_growth = 0.0
labor_first = [50.0, 50.0, 0.0]
initial_product = 8.0

product(t) = GEMB.CommodityRef(:product; period=t)
labor(t) = GEMB.CommodityRef(:labor; period=t)

model = GEMB.GEMBModel(
    [
        GEMB.CommoditySpec(
            :product;
            axes=(period=1:np,),
            price_lower_bound=1.0e-100,
        ),
        GEMB.CommoditySpec(
            :labor;
            axes=(period=1:(np - 1),),
            price_lower_bound=1.0e-100,
        ),
    ];
    numeraire=labor(1),
)

for t in 1:(np - 1)
    GEMB.add_agent!(
        model,
        GEMB.CESSpec(
            [beta_firm, 1 - beta_firm];
            es=1.0,
            alpha=alpha,
        );
        outputs=product(t + 1),
        demands=[product(t), labor(t)],
        activity_start=100.0,
        name=Symbol("firm_$t"),
    )
end

for g in 1:ng
    scale = (1 + population_growth)^(g - 1)

    consumer_demands = [
        product(g),
        product(g + 1),
        product(g + 2),
    ]

    consumer_endowments = [
        labor(g),
        labor(g + 1),
    ]

    consumer_endowment_quantities = labor_first[1:2] .* scale

    if g == 1
        consumer_endowments = vcat(
            [product(1)],
            consumer_endowments,
        )
        consumer_endowment_quantities = vcat(
            [initial_product],
            consumer_endowment_quantities,
        )
    end

    GEMB.add_agent!(
        model,
        GEMB.CESSpec(
            beta_consumer;
            es=1.0,
            alpha=1.0,
        );
        demands=consumer_demands,
        endowments=consumer_endowments,
        endowment_quantities=consumer_endowment_quantities,
        name=Symbol("consumer_$g"),
    )
end

gamma_p = -4.0 + 2.0 * sqrt(6.0)

steady_labor =
    labor_first[1] +
    labor_first[2] / (1 + population_growth)

steady_output =
    alpha^(1 / (1 - beta_firm)) *
    (beta_firm * gamma_p)^(beta_firm / (1 - beta_firm)) *
    steady_labor

p0 = ones(2 * np - 1)

result = GEMB.solve(
    model;
    p0=p0,
    residual_tol=1.0e-9,
    silent=true,
)

result.solved || error("Equilibrium solve failed.")

firm_output = [
    result.agent_variable_values[t][1]
    for t in 1:(np - 1)
]

prices = result.prices
product_prices = prices[1:np]
labor_prices = prices[(np + 1):(2 * np - 1)]

mid = div(np - 1, 2)
window = max(1, mid - 2):min(np - 1, mid + 2)

println()
println("Finite-horizon OLG production timeline")
println("--------------------------------------")
println("Solved: ", result.solved)
println("Generations: ", ng)
println("Periods: ", np)
println("Steady-state price-growth factor: ", gamma_p)
println("Steady-state output: ", steady_output)

println()
println("Firm output around the middle of the timeline:")
for t in window
    println("firm_", t, ": ", firm_output[t])
end

println()
println("Last seven firm outputs:")
for t in max(1, np - 7):(np - 1)
    println("firm_", t, ": ", firm_output[t])
end

println()
println("Middle price-growth factors:")
println(
    "product_", mid + 1, " / product_", mid, ": ",
    product_prices[mid + 1] / product_prices[mid],
)
println(
    "labor_", mid + 1, " / labor_", mid, ":     ",
    labor_prices[mid + 1] / labor_prices[mid],
)

output_plot = plot(
    1:length(firm_output),
    firm_output;
    xlabel="Period",
    ylabel="Output",
    label="Output",
    legend=:bottomright,
    marker=:circle,
    grid=true,
    gridalpha=1.0,
)

hline!(
    output_plot,
    [steady_output];
    label="Steady-state output",
    linestyle=:dash,
)

display(output_plot)
