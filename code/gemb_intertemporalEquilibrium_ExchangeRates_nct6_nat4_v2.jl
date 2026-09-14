using GeneralEquilibriumModeling.GEM
using GeneralEquilibriumModeling.GEMB

# Parameters
n_p = 3
n_prod = n_p - 1

alpha = ones(n_prod)
labor_endowment = fill(100.0, n_prod)
money_endowment = 1.0

beta_A = [8.0 / 15.0, 4.0 / 15.0]
beta_m_A = 1.0 / 5.0

beta_B = [27.0 / 40.0, 9.0 / 40.0]
beta_m_B = 1.0 / 10.0

lambda = [
    3.0 / 4.0  1.0 / 4.0
    1.0 / 4.0  3.0 / 4.0
]

# Commodities
model = GEMBModel(
    [
        CommoditySpec(
            :productA;
            axes=(period=2:n_p,),
            price_lower_bound=1.0e-10,
        ),
        CommoditySpec(
            :productB;
            axes=(period=2:n_p,),
            price_lower_bound=1.0e-10,
        ),
        CommoditySpec(
            :laborA;
            axes=(period=1:n_prod,),
            price_lower_bound=1.0e-10,
        ),
        CommoditySpec(
            :laborB;
            axes=(period=1:n_prod,),
            price_lower_bound=1.0e-10,
        ),
        CommoditySpec(
            :moneyA;
            axes=(period=1:n_p,),
            price_lower_bound=1.0e-10,
        ),
        CommoditySpec(
            :moneyB;
            axes=(period=1:n_p,),
            price_lower_bound=1.0e-10,
        ),
    ];
    numeraire=CommodityRef(:productA; period=2),
    numeraire_value=1.0,
)

# Endogenous interest rates
rate_names_A = [Symbol("rA_$i") for i in 1:n_prod]
rate_names_B = [Symbol("rB_$i") for i in 1:n_prod]
rate_names = [rate_names_A; rate_names_B]

add_agent!(
    model,
    ConditionAgentSpec(
        (rates, p_money) -> [
            [
                rates[i] -
                p_money[i] / sum(p_money[(i + 1):n_p])
                for i in 1:n_prod
            ];
            [
                rates[n_prod + i] -
                p_money[n_p + i] /
                sum(p_money[(n_p + i + 1):(2 * n_p)])
                for i in 1:n_prod
            ];
        ],
    );
    variable_names=rate_names,
    variable_start=[
        fill(0.8, n_prod);
        fill(1.5, n_prod);
    ],
    variable_lower_bounds=zeros(2 * n_prod),
    variable_upper_bounds=Inf,
    observed_variables=[
        [PriceVariableRef(Symbol("moneyA_$i")) for i in 1:n_p];
        [PriceVariableRef(Symbol("moneyB_$i")) for i in 1:n_p];
    ],
    name=:interestRateDeterminer,
)

rate_refs = [
    agent_variable_ref(:interestRateDeterminer, name)
    for name in rate_names
]
rate_refs_A = rate_refs[1:n_prod]
rate_refs_B = rate_refs[(n_prod + 1):(2 * n_prod)]

# Firms
function make_firm_spec(alpha_i)
    return ActivityDemandSpec(
        (activity, prices, observed_values) -> begin
            p_labor, p_money = prices
            r = observed_values[1]

            return [
                activity / alpha_i,
                activity * p_labor * r /
                (alpha_i * p_money),
            ]
        end,
    )
end

for (country, rates) in ((:A, rate_refs_A), (:B, rate_refs_B))
    for i in 1:n_prod
        add_agent!(
            model,
            make_firm_spec(alpha[i]);
            outputs=CommodityRef(Symbol("product$country"); period=i + 1),
            output_coefficients=1.0,
            demands=[
                CommodityRef(Symbol("labor$country"); period=i),
                CommodityRef(Symbol("money$country"); period=i),
            ],
            observed_variables=[rates[i]],
            activity_start=100.0,
            name=Symbol("firm$(country)_$i"),
        )
    end
end

# Consumers
product_refs_A = [
    CommodityRef(:productA; period=i)
    for i in 2:n_p
]
product_refs_B = [
    CommodityRef(:productB; period=i)
    for i in 2:n_p
]
labor_refs_A = [
    CommodityRef(:laborA; period=i)
    for i in 1:n_prod
]
labor_refs_B = [
    CommodityRef(:laborB; period=i)
    for i in 1:n_prod
]
money_refs_A = [
    CommodityRef(:moneyA; period=i)
    for i in 1:n_p
]
money_refs_B = [
    CommodityRef(:moneyB; period=i)
    for i in 1:n_p
]

consumer_demands = [
    product_refs_A;
    product_refs_B;
    money_refs_A;
    money_refs_B;
]

function consumer_demand(income, prices, rates, lambda_A, lambda_B)
    rates_A = rates[1:n_prod]
    rates_B = rates[(n_prod + 1):(2 * n_prod)]

    p_product_A = prices[1:n_prod]
    p_product_B = prices[(n_prod + 1):(2 * n_prod)]
    p_money_A = prices[(2 * n_prod + 1):(2 * n_prod + n_p)]
    p_money_B = prices[(2 * n_prod + n_p + 1):(2 * n_prod + 2 * n_p)]

    product_demand_A = [
        lambda_A * beta_A[i] * income /
        ((1.0 + rates_A[i]) * p_product_A[i])
        for i in 1:n_prod
    ]

    product_demand_B = [
        lambda_B * beta_B[i] * income /
        ((1.0 + rates_B[i]) * p_product_B[i])
        for i in 1:n_prod
    ]

    money_demand_A = [
        lambda_A * beta_A[i] * income * rates_A[i] /
        ((1.0 + rates_A[i]) * p_money_A[i])
        for i in 1:n_prod
    ]
    push!(
        money_demand_A,
        lambda_A * beta_m_A * income / p_money_A[end],
    )

    money_demand_B = [
        lambda_B * beta_B[i] * income * rates_B[i] /
        ((1.0 + rates_B[i]) * p_money_B[i])
        for i in 1:n_prod
    ]
    push!(
        money_demand_B,
        lambda_B * beta_m_B * income / p_money_B[end],
    )

    return [
        product_demand_A;
        product_demand_B;
        money_demand_A;
        money_demand_B;
    ]
end

function make_consumer_spec(lambda_A, lambda_B)
    return MarshallDemandConsumerSpec(
        (income, prices, rates) ->
            consumer_demand(income, prices, rates, lambda_A, lambda_B),
    )
end

for (
    name,
    lambda_A,
    lambda_B,
    labor_refs,
    money_refs,
) in (
    (:consumerA, lambda[1, 1], lambda[2, 1], labor_refs_A, money_refs_A),
    (:consumerB, lambda[1, 2], lambda[2, 2], labor_refs_B, money_refs_B),
)
    add_agent!(
        model,
        make_consumer_spec(lambda_A, lambda_B);
        demands=consumer_demands,
        endowments=[labor_refs; money_refs],
        endowment_quantities=[
            labor_endowment;
            fill(money_endowment, n_p);
        ],
        observed_variables=rate_refs,
        name=name,
    )
end

# Starting values
p0 = [
    1.0, 0.55,        # product A
    0.80, 0.30,       # product B
    0.55, 0.23,       # labor A
    0.30, 0.10,       # labor B
    140.0, 80.0, 70.0,# money A
    210.0, 80.0, 40.0 # money B
]

# Solve
result = solve(
    model;
    p0=p0,
    residual_tol=1.0e-8,
    silent=true,
)

result.solved || error("Equilibrium solve failed.")
result.all_markets_clear || error(
    "PATH found a complementarity solution, but not all commodity markets clear."
)

# Results
rates = result.agent_variable_values[1]
rates_A = rates[1:n_prod]
rates_B = rates[(n_prod + 1):(2 * n_prod)]

firm_activities = [
    result.agent_variable_values[1 + i][1]
    for i in 1:(2 * n_prod)
]
firm_activities_A = firm_activities[1:n_prod]
firm_activities_B = firm_activities[(n_prod + 1):(2 * n_prod)]

p = result.prices

p_product_A = p[1:n_prod]
p_product_B = p[(n_prod + 1):(2 * n_prod)]
p_labor_A = p[(2 * n_prod + 1):(3 * n_prod)]
p_labor_B = p[(3 * n_prod + 1):(4 * n_prod)]
p_money_A = p[(4 * n_prod + 1):(4 * n_prod + n_p)]
p_money_B = p[(4 * n_prod + n_p + 1):(4 * n_prod + 2 * n_p)]

income_A =
    sum(p_labor_A .* labor_endowment) +
    money_endowment * sum(p_money_A)

income_B =
    sum(p_labor_B .* labor_endowment) +
    money_endowment * sum(p_money_B)

currency_value_A = [
    sum(p_money_A[i:end])
    for i in 1:n_p
]
currency_value_B = [
    sum(p_money_B[i:end])
    for i in 1:n_p
]

exchange_rates = currency_value_A ./ currency_value_B
exchange_rate_growth = exchange_rates[2:end] ./ exchange_rates[1:(end - 1)]
interest_rate_ratio = (1.0 .+ rates_B) ./ (1.0 .+ rates_A)

consumer_prices = [
    p_product_A;
    p_product_B;
    p_money_A;
    p_money_B;
]

demand_A = consumer_demand(
    income_A,
    consumer_prices,
    rates,
    lambda[1, 1],
    lambda[2, 1],
)

demand_B = consumer_demand(
    income_B,
    consumer_prices,
    rates,
    lambda[1, 2],
    lambda[2, 2],
)

consumption_AA = demand_A[1:n_prod]
consumption_BA = demand_A[(n_prod + 1):(2 * n_prod)]
consumption_AB = demand_B[1:n_prod]
consumption_BB = demand_B[(n_prod + 1):(2 * n_prod)]

println()
println("========== GEMB endogenous-interest exchange-rate equilibrium ==========")
println("Solved:                        ", result.solved)
println("All markets clear:             ", result.all_markets_clear)
println("Interest rates A:              ", rates_A)
println("Interest rates B:              ", rates_B)
println("Income A:                      ", income_A)
println("Income B:                      ", income_B)
println("Product prices A:              ", p_product_A)
println("Product prices B:              ", p_product_B)
println("Labor prices A:                ", p_labor_A)
println("Labor prices B:                ", p_labor_B)
println("Currency-service prices A:     ", p_money_A)
println("Currency-service prices B:     ", p_money_B)
println("Firm activities A:             ", firm_activities_A)
println("Firm activities B:             ", firm_activities_B)
println("Consumption AA:                ", consumption_AA)
println("Consumption AB:                ", consumption_AB)
println("Consumption BA:                ", consumption_BA)
println("Consumption BB:                ", consumption_BB)
println("Currency values A:             ", currency_value_A)
println("Currency values B:             ", currency_value_B)
println("Exchange rates B per A:        ", exchange_rates)
println("Exchange-rate growth:          ", exchange_rate_growth)
println("Interest-rate ratio:           ", interest_rate_ratio)
println("Max market residual:           ", result.max_market_residual)
println("Max natural residual:          ", result.max_natural_residual)
println("========================================================================")
