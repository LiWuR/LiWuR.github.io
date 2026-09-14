using GeneralEquilibriumModeling.GEMB

n_p = 3
n_finite = n_p - 1

omega = 100.0
beta_A = [8.0 / 27.0, 4.0 / 27.0, 2.0 / 27.0]
beta_B = [9.0 / 27.0, 3.0 / 27.0, 1.0 / 27.0]

model = GEMBModel(
    [
        CommoditySpec(:laborA; axes=(period=1:n_finite,), price_lower_bound=1.0e-10),
        CommoditySpec(:laborB; axes=(period=1:n_finite,), price_lower_bound=1.0e-10),
        CommoditySpec(:moneyA; axes=(period=1:n_p,), price_lower_bound=1.0e-10),
        CommoditySpec(:moneyB; axes=(period=1:n_p,), price_lower_bound=1.0e-10),
    ];
    numeraire=CommodityRef(:laborA; period=1),
    numeraire_value=1.0,
)

labor_refs_A = [CommodityRef(:laborA; period=i) for i in 1:n_finite]
labor_refs_B = [CommodityRef(:laborB; period=i) for i in 1:n_finite]
money_refs_A = [CommodityRef(:moneyA; period=i) for i in 1:n_p]
money_refs_B = [CommodityRef(:moneyB; period=i) for i in 1:n_p]

consumer_demands = [labor_refs_A; labor_refs_B; money_refs_A; money_refs_B]

interest_rates(p_money) = [
    p_money[i] / sum(p_money[(i + 1):end])
    for i in 1:n_finite
]

function consumer_demand(income, prices)
    p_labor_A = prices[1:n_finite]
    p_labor_B = prices[(n_finite + 1):(2 * n_finite)]
    p_money_A = prices[(2 * n_finite + 1):(2 * n_finite + n_p)]
    p_money_B = prices[(2 * n_finite + n_p + 1):end]

    r_A = interest_rates(p_money_A)
    r_B = interest_rates(p_money_B)

    labor_A = [
        beta_A[i] * income / ((1.0 + r_A[i]) * p_labor_A[i])
        for i in 1:n_finite
    ]
    labor_B = [
        beta_B[i] * income / ((1.0 + r_B[i]) * p_labor_B[i])
        for i in 1:n_finite
    ]

    money_A = [
        [
            beta_A[i] * income * r_A[i] /
            ((1.0 + r_A[i]) * p_money_A[i])
            for i in 1:n_finite
        ];
        beta_A[end] * income / p_money_A[end]
    ]

    money_B = [
        [
            beta_B[i] * income * r_B[i] /
            ((1.0 + r_B[i]) * p_money_B[i])
            for i in 1:n_finite
        ];
        beta_B[end] * income / p_money_B[end]
    ]

    return [labor_A; labor_B; money_A; money_B]
end

consumer_spec = MarshallDemandConsumerSpec(consumer_demand)

for (name, labor_refs, money_refs) in (
    (:consumer1, labor_refs_A, money_refs_A),
    (:consumer2, labor_refs_B, money_refs_B),
)
    add_agent!(
        model,
        consumer_spec;
        demands=consumer_demands,
        endowments=[labor_refs; money_refs],
        endowment_quantities=[fill(omega, n_finite); ones(n_p)],
        name=name,
    )
end

p0 = [
    1.0, 0.60,
    0.80, 0.30,
    90.0, 55.0, 45.0,
    140.0, 55.0, 30.0,
]

result = solve(model; p0=p0, residual_tol=1.0e-8, silent=true)

result.solved || error("Equilibrium solve failed.")
result.all_markets_clear || error(
    "PATH found a complementarity solution, but not all commodity markets clear."
)

p = result.prices

p_labor_A = [p[1:n_finite]; 0.0]
p_labor_B = [p[(n_finite + 1):(2 * n_finite)]; 0.0]
p_money_A = p[(2 * n_finite + 1):(2 * n_finite + n_p)]
p_money_B = p[(2 * n_finite + n_p + 1):end]

r_A = interest_rates(p_money_A)
r_B = interest_rates(p_money_B)

income_1 = omega * sum(p_labor_A[1:n_finite]) + sum(p_money_A)
income_2 = omega * sum(p_labor_B[1:n_finite]) + sum(p_money_B)

currency_value_A = [sum(p_money_A[i:end]) for i in 1:n_p]
currency_value_B = [sum(p_money_B[i:end]) for i in 1:n_p]

exchange_rates = currency_value_A ./ currency_value_B
exchange_rate_growth = exchange_rates[2:end] ./ exchange_rates[1:(end - 1)]
interest_rate_ratio = (1.0 .+ r_B) ./ (1.0 .+ r_A)

println()
println("========== GEMB pure-exchange interest-rate and exchange-rate equilibrium ==========")
println("Interest rates A:              ", r_A)
println("Interest rates B:              ", r_B)
println("Income consumer 1:             ", income_1)
println("Income consumer 2:             ", income_2)
println("Labor prices A:                ", p_labor_A)
println("Labor prices B:                ", p_labor_B)
println("Currency-service prices A:     ", p_money_A)
println("Currency-service prices B:     ", p_money_B)
println("Currency values A:             ", currency_value_A)
println("Currency values B:             ", currency_value_B)
println("Exchange rates B per A:        ", exchange_rates)
println("Exchange-rate growth:          ", exchange_rate_growth)
println("Interest-rate ratio:           ", interest_rate_ratio)
println("Max market residual:           ", result.max_market_residual)
println("Max natural residual:          ", result.max_natural_residual)
println("====================================================================================")
