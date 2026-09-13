using GeneralEquilibriumModeling.GEM
using GeneralEquilibriumModeling.GEMB

# Parameters
n_p = 4
n_prod = n_p - 1

theta = 0.5
alpha = ones(n_prod)

initial_capital = 62.0
omega = [62.0, 32.0, 20.0]

# Consumption shares for products 2,...,n_p
beta = [352.0, 228.0, 120.0] ./ 775.0

# Share spent on terminal money in period n_p
beta_terminal_money = 75.0 / 775.0

@assert isapprox(
    sum(beta) + beta_terminal_money,
    1.0;
    atol=1e-12,
)

# Commodities
model = GEMBModel(
    [
        CommoditySpec(
            :product;
            axes=(period=1:n_p,),
            price_lower_bound=1e-10,
        ),
        CommoditySpec(
            :labor;
            axes=(period=1:n_prod,),
            price_lower_bound=1e-10,
        ),
        CommoditySpec(
            :money;
            axes=(period=1:n_p,),
            price_lower_bound=1e-10,
        ),
    ];
    numeraire=CommodityRef(:product; period=2),
    numeraire_value=1.0,
)

# Endogenous interest rates
#
#     r_i = p_money_i / sum_{j=i+1}^{n_p} p_money_j,
#     i = 1,...,n_p-1
rate_names = [Symbol("r$i") for i in 1:n_prod]

add_agent!(
    model,
    ConditionAgentSpec(
        (r, p_m) -> [
            r[i] - p_m[i] / sum(p_m[(i + 1):n_p])
            for i in 1:n_prod
        ],
    );
    variable_names=rate_names,
    variable_start=ones(n_prod),
    variable_lower_bounds=zeros(n_prod),
    variable_upper_bounds=Inf,
    observed_variables=[
        PriceVariableRef(Symbol("money_$i"))
        for i in 1:n_p
    ],
    name=:interestRateDeterminer,
)

rate_refs = [
    agent_variable_ref(:interestRateDeterminer, rate_names[i])
    for i in 1:n_prod
]

# Production activities of the intertemporal firm
#
#     y_i = alpha_i * k_i^theta * l_i^(1-theta)
#
# Period i uses product i as capital and produces product i+1.
#
# In period 1, both capital and labor are market purchases:
#
#     effective prices = [(1+r_1)p_1, (1+r_1)p_l1]
#
# From period 2 onward, capital is carried internally:
#
#     effective prices = [p_i, (1+r_i)p_li]
#
activity_start = [60.0, 25.0, 10.0]

for i in 1:n_prod
    firm = ActivityDemandSpec(
        (y, p, obs) -> begin
            p_k, p_l, p_m = p
            r = obs[1]

            effective_prices =
                i == 1 ?
                [(1.0 + r) * p_k, (1.0 + r) * p_l] :
                [p_k, (1.0 + r) * p_l]

            x = GEMB.CES_input(
                [theta, 1.0 - theta],
                y,
                effective_prices;
                es=1.0,
                alpha=alpha[i],
            )

            h_f =
                i == 1 ?
                r * (p_k * x[1] + p_l * x[2]) / p_m :
                r * p_l * x[2] / p_m

            [x[1], x[2], h_f]
        end,
    )

    add_agent!(
        model,
        firm;
        outputs=CommodityRef(:product; period=i + 1),
        output_coefficients=1.0,
        demands=[
            CommodityRef(:product; period=i),
            CommodityRef(:labor; period=i),
            CommodityRef(:money; period=i),
        ],
        observed_variables=[rate_refs[i]],
        activity_start=activity_start[i],
        name=Symbol("firm$i"),
    )
end

# Consumer
product_refs = [
    CommodityRef(:product; period=i + 1)
    for i in 1:n_prod
]

labor_refs = [
    CommodityRef(:labor; period=i)
    for i in 1:n_prod
]

money_refs = [
    CommodityRef(:money; period=i)
    for i in 1:n_p
]

consumer = MarshallDemandConsumerSpec(
    (w, p, r) -> begin
        p_x = p[1:n_prod]
        p_m = p[(n_prod + 1):(n_prod + n_p)]

        c = [
            beta[i] * w /
            ((1.0 + r[i]) * p_x[i])
            for i in 1:n_prod
        ]

        h = [
            r[i] * beta[i] * w /
            ((1.0 + r[i]) * p_m[i])
            for i in 1:n_prod
        ]

        [
            c;
            h;
            beta_terminal_money * w / p_m[n_p];
        ]
    end,
)

add_agent!(
    model,
    consumer;
    demands=[product_refs; money_refs],
    endowments=[
        CommodityRef(:product; period=1);
        labor_refs;
        money_refs;
    ],
    endowment_quantities=[
        initial_capital;
        omega;
        ones(n_p);
    ],
    observed_variables=rate_refs,
    name=:consumer,
)

# Starting prices
#
# product 1,...,4;
# labor 1,...,3;
# money 1,...,4
p0 = [
    0.30, 1.00, 1.40, 1.40,
    0.30, 0.30, 0.20,
    70.0, 35.0, 20.0, 20.0,
]

# Solve
result = solve(
    model;
    p0=p0,
    residual_tol=1e-8,
    silent=true,
)

# Results
rates = result.agent_variable_values[1]

outputs = [
    result.agent_variable_values[i + 1][1]
    for i in 1:n_prod
]

p = result.prices

product_prices = p[1:n_p]
labor_prices = p[(n_p + 1):(n_p + n_prod)]
money_prices = p[(n_p + n_prod + 1):(2 * n_p + n_prod)]

w =
    product_prices[1] * initial_capital +
    sum(labor_prices .* omega) +
    sum(money_prices)

# Consumption of products 2,...,n_p
consumption = [
    beta[i] * w /
    ((1.0 + rates[i]) * product_prices[i + 1])
    for i in 1:n_prod
]

# Capital stocks k_1,...,k_{n_p-1}
capital = [
    initial_capital;
    [
        outputs[i] - consumption[i]
        for i in 1:(n_prod - 1)
    ];
]

println("Solved:          ", result.solved)
println("Periods:         ", n_p)
println("Production periods: ", n_prod)
println("Prices:          ", p)
println("Interest rates:  ", rates)
println("Firm outputs:    ", outputs)
println("Capital path:    ", capital)
println("Consumption:     ", consumption)
println("Consumer income: ", w)
println("Max residual:    ", result.max_natural_residual)

# Analytical benchmark
#
# Prices are ordered as:
# product 1,...,4; labor 1,...,3; money 1,...,4.
expected_prices = [
    1 / 4, 1.0, 3 / 2, 3 / 2,
    1 / 4, 9 / 32, 3 / 16,
    75.0, 75 / 2, 75 / 4, 75 / 4,
]

@assert result.solved
@assert isapprox(rates, ones(n_prod); atol=1e-6, rtol=1e-6)
@assert isapprox(p, expected_prices; atol=1e-6, rtol=1e-6)
@assert isapprox(outputs, [62.0, 24.0, 10.0]; atol=1e-6, rtol=1e-6)
@assert isapprox(capital, [62.0, 18.0, 5.0]; atol=1e-6, rtol=1e-6)
@assert isapprox(consumption, [44.0, 19.0, 10.0]; atol=1e-6, rtol=1e-6)
@assert isapprox(w, 775 / 4; atol=1e-6, rtol=1e-6)
@assert result.max_natural_residual <= 1e-8
