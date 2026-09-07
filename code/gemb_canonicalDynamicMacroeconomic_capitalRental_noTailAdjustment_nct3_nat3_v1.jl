using GeneralEquilibriumModeling

np = 100

discount_factor = 0.97
depreciation_rate = 0.06
capital_share = 0.35
consumption_share = 0.4

labor_endowment = 100.0
initial_capital = 286.6341

technology = ones(np - 1)

product(t) = GEMB.CommodityRef(:product; period=t)
labor(t) = GEMB.CommodityRef(:labor; period=t)
capital_rental(t) = GEMB.CommodityRef(:capital_rental; period=t)

intertemporal_weights = discount_factor .^ (0:(np - 2))

model = GEMB.GEMBModel(
    [
        GEMB.CommoditySpec(
            :product;
            axes=(period=1:np,),
            price_lower_bound=1.0e-10,
        ),
        GEMB.CommoditySpec(
            :labor;
            axes=(period=1:(np - 1),),
            price_lower_bound=1.0e-10,
        ),
        GEMB.CommoditySpec(
            :capital_rental;
            axes=(period=1:(np - 1),),
            price_lower_bound=1.0e-10,
        ),
    ];
    numeraire=labor(1),
)

rental_spec = GEMB.ActivityDemandSpec(
    (activity, prices) -> [activity],
)

for t in 1:(np - 1)
    GEMB.add_agent!(
        model,
        GEMB.CESSpec(
            [capital_share, 1 - capital_share];
            es=1.0,
            alpha=technology[t],
        );
        outputs=product(t + 1),
        demands=[
            capital_rental(t),
            labor(t),
        ],
        activity_start=100.0,
        name=Symbol("firm_$t"),
    )

    GEMB.add_agent!(
        model,
        rental_spec;
        outputs=[
            capital_rental(t),
            product(t + 1),
        ],
        output_coefficients=[
            1.0,
            1 - depreciation_rate,
        ],
        demands=product(t),
        activity_start=100.0,
        name=Symbol("rental_$t"),
    )
end

product_weights =
    consumption_share .* intertemporal_weights

labor_weights =
    (1 - consumption_share) .* intertemporal_weights

utility_weights =
    vcat(product_weights, labor_weights)
utility_weights ./= sum(utility_weights)

consumer_demands = vcat(
    [product(t) for t in 2:np],
    [labor(t) for t in 1:(np - 1)],
)

consumer_endowments = vcat(
    [product(1)],
    [labor(t) for t in 1:(np - 1)],
)

consumer_endowment_quantities = vcat(
    [initial_capital],
    fill(labor_endowment, np - 1),
)

GEMB.add_agent!(
    model,
    GEMB.CESSpec(
        utility_weights;
        es=1.0,
        alpha=1.0,
    );
    demands=consumer_demands,
    endowments=consumer_endowments,
    endowment_quantities=consumer_endowment_quantities,
    name=:consumer,
)

gross_return = 1 / discount_factor
target_mpk =
    gross_return - (1 - depreciation_rate)

capital_labor_ratio =
    (capital_share / target_mpk)^(1 / (1 - capital_share))

steady_mpl =
    (1 - capital_share) *
    capital_labor_ratio^capital_share

product_price_1 =
    1 / (discount_factor * steady_mpl)

product_p0 =
    product_price_1 .* discount_factor .^ (0:(np - 1))

labor_p0 =
    discount_factor .^ (0:(np - 2))

rental_p0 =
    product_p0[1:(np - 1)] .-
    (1 - depreciation_rate) .* product_p0[2:np]

p0 = vcat(
    product_p0,
    labor_p0,
    rental_p0,
)

result = GEMB.solve(
    model;
    p0=p0,
    residual_tol=1.0e-9,
    silent=true,
)

result.solved || error("Equilibrium solve failed.")

new_output = [
    result.agent_variable_values[2 * t - 1][1]
    for t in 1:(np - 1)
]

capital = [
    result.agent_variable_values[2 * t][1]
    for t in 1:(np - 1)
]

labor_input = [
    -result.agent_net_supplies[2 * t - 1][3]
    for t in 1:(np - 1)
]

consumer_supply = result.agent_net_supplies[end]

consumption =
    -consumer_supply[1:(np - 1)]

labor_supply =
    consumer_supply[np:(2 * np - 2)]

leisure =
    labor_endowment .- labor_supply

total_output =
    new_output .+
    (1 - depreciation_rate) .* capital

steady_labor =
    labor_endowment / (
        1 +
        ((1 - consumption_share) / consumption_share) *
        (
            capital_labor_ratio^capital_share -
            depreciation_rate * capital_labor_ratio
        ) /
        steady_mpl
    )

steady_capital =
    capital_labor_ratio * steady_labor

steady_new_output =
    steady_capital^capital_share *
    steady_labor^(1 - capital_share)

steady_consumption =
    steady_new_output -
    depreciation_rate * steady_capital

println()
println("Canonical dynamic macroeconomic equilibrium")
println("-------------------------------------------")
println("Solved: ", result.solved)
println("Periods: ", np)
println("Tail adjustment: false")

println()
println("Steady-state benchmark")
println("Capital:     ", steady_capital)
println("Labor input: ", steady_labor)
println("New output:  ", steady_new_output)
println("Consumption: ", steady_consumption)

println()
println("Equilibrium paths")
println("Capital:      ", capital)
println("Labor input:  ", labor_input)
println("New output:   ", new_output)
println("Consumption:  ", consumption)
println("Leisure:      ", leisure)
println("Total output: ", total_output)

prices = result.prices

product_prices = prices[1:np]
labor_prices = prices[(np + 1):(2 * np - 1)]
rental_prices = prices[(2 * np):(3 * np - 2)]

println("Product prices:  ", product_prices)
println("Labor prices:    ", labor_prices)
println("Rental prices:   ", rental_prices)

println()

