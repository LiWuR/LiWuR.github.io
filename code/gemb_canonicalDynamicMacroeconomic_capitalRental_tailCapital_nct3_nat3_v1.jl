using GeneralEquilibriumModeling

np = 20

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

consumer_commodities = vcat(
    [product(t) for t in 1:np],
    [labor(t) for t in 1:(np - 1)],
)

consumer_net_supply = function (
    local_variables,
    local_prices,
    observed_values=Any[],
)
    terminal_weight = local_variables[1]

    product_prices = local_prices[1:np]
    labor_prices = local_prices[(np + 1):(2 * np - 1)]

    product_weights =
        consumption_share .* intertemporal_weights
    product_weights[end] += terminal_weight

    labor_weights =
        (1 - consumption_share) .* intertemporal_weights

    beta = vcat(product_weights, labor_weights)
    beta ./= sum(beta)

    income =
        product_prices[1] * initial_capital +
        labor_endowment * sum(labor_prices)

    demand_prices = vcat(
        product_prices[2:np],
        labor_prices,
    )

    demand = GEMB.CES_demand(
        beta,
        income,
        demand_prices;
        es=1.0,
    )

    T = promote_type(
        eltype(local_variables),
        eltype(local_prices),
    )
    supply = zeros(T, 2 * np - 1)

    supply[1] = initial_capital
    supply[2:np] .= -demand[1:(np - 1)]
    supply[(np + 1):(2 * np - 1)] .=
        labor_endowment .- demand[np:(2 * np - 2)]

    return supply
end

tail_condition = function (
    local_variables,
    local_prices,
    net_supply,
    observed_values,
)
    return [observed_values[1] - observed_values[2]]
end

GEMB.add_net_supply_agent!(
    model,
    consumer_net_supply;
    commodities=consumer_commodities,
    variable_names=[:terminal_weight],
    variable_lower_bounds=[1.0e-8],
    variable_upper_bounds=[Inf],
    variable_start=[1.0],
    observed_variables=[
        GEM.AgentVariableRef(
            Symbol("rental_$(np - 2)"),
            :activity,
        ),
        GEM.AgentVariableRef(
            Symbol("rental_$(np - 1)"),
            :activity,
        ),
    ],
    condition_rule=GEM.ExplicitAgentConditions(
        tail_condition,
    ),
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

prices = result.prices

product_prices = prices[1:np]
labor_prices = prices[(np + 1):(2 * np - 1)]
rental_prices = prices[(2 * np):(3 * np - 2)]

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

product_demand =
    -consumer_supply[2:np]

labor_supply =
    consumer_supply[(np + 1):(2 * np - 1)]

leisure =
    labor_endowment .- labor_supply

terminal_weight =
    result.agent_variable_values[end][1]

ordinary_last_weight =
    consumption_share * intertemporal_weights[end]

ordinary_consumption = copy(product_demand)
ordinary_consumption[end] *=
    ordinary_last_weight /
    (ordinary_last_weight + terminal_weight)

terminal_product_demand =
    product_demand[end] - ordinary_consumption[end]

total_output =
    new_output .+
    (1 - depreciation_rate) .* capital

println()
println("Canonical dynamic macroeconomic equilibrium")
println("-------------------------------------------")
println("Solved: ", result.solved)
println("Terminal weight:       ", terminal_weight)
println(
    "Terminal weight ratio: ",
    terminal_weight / intertemporal_weights[end],
)
println(
    "Last-capital difference: ",
    capital[end - 1] - capital[end],
)

println()
println("Equilibrium paths")
println("Capital:              ", capital)
println("Labor input:          ", labor_input)
println("New output:           ", new_output)
println("Ordinary consumption: ", ordinary_consumption)
println("Terminal demand:       ", terminal_product_demand)
println("Leisure:              ", leisure)

println()

