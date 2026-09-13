using GeneralEquilibriumModeling.GEM
using GeneralEquilibriumModeling.GEMB

# Parameters
n_p = 4
n_prod = n_p - 1

# Common-rate parameterization with r = 1
# beta_i = 2^(n_p-i)/(2^n_p-1), i = 1,...,n_p-2
# beta_(n_p-1) = 3/(2^n_p-1)
denominator = 2.0^n_p - 1.0
beta = [
    i < n_prod ?
    2.0^(n_p - i) / denominator :
    3.0 / denominator
    for i in 1:n_prod
]

alpha = ones(n_prod)
labor_endowment = fill(100.0, n_prod)
money_endowment = ones(n_prod)

terminal_product = Symbol("product_$n_p")
terminal_labor = Symbol("labor_$n_prod")

# Dated products are product 2,...,product n_p.
# Nonterminal product prices are strictly positive; the terminal product
# price is allowed to approach zero under the terminal condition.
product = CommoditySpec(
    :product;
    axes=(period=2:(n_p - 1),),
    price_lower_bound=1.0e-10,
)

# Labor and money are dated by production period 1,...,n_p-1.
labor = CommoditySpec(
    :labor;
    axes=(period=1:(n_prod - 1),),
    price_lower_bound=1.0e-10,
)

money = CommoditySpec(
    :money;
    axes=(period=1:n_prod,),
    price_lower_bound=1.0e-10,
)

model = GEMBModel(
    [product, terminal_product, labor, terminal_labor, money];
    numeraire=CommodityRef(:product; period=2),
    numeraire_value=1.0,
)

# Endogenous nonterminal interest rates and terminal interest-inclusive price
rate_count = n_prod - 1
rate_names = [Symbol("r$i") for i in 1:rate_count]
variable_names = [rate_names; :terminal_interest_inclusive_price]

interest_rate_spec = ConditionAgentSpec(
    (variables, observed_values) -> begin
        rates = variables[1:rate_count]
        terminal_price = variables[n_prod]
        p_money = observed_values

        conditions = [
            rates[i] -
            p_money[i] / sum(p_money[(i + 1):n_prod])
            for i in 1:rate_count
        ]

        push!(
            conditions,
            terminal_price -
            p_money[n_prod] /
            (
                alpha[n_prod] *
                labor_endowment[n_prod]
            ),
        )

        return conditions
    end,
)

add_agent!(
    model,
    interest_rate_spec;
    variable_names=variable_names,
    variable_start=fill(0.1, n_prod),
    variable_lower_bounds=[
        zeros(rate_count);
        1.0e-10;
    ],
    variable_upper_bounds=Inf,
    observed_variables=[
        PriceVariableRef(Symbol("money_$i"))
        for i in 1:n_prod
    ],
    name=:interestRateDeterminer,
)

rate_refs = [
    agent_variable_ref(:interestRateDeterminer, rate_names[i])
    for i in 1:rate_count
]

terminal_price_ref = agent_variable_ref(
    :interestRateDeterminer,
    :terminal_interest_inclusive_price,
)

# Firms
# Activity is measured by output y_i. Producing y_i units of product i+1
# requires y_i/alpha_i units of labor i.
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

for i in 1:rate_count
    add_agent!(
        model,
        make_firm_spec(alpha[i]);
        outputs=CommodityRef(:product; period=i + 1),
        output_coefficients=1.0,
        demands=[
            CommodityRef(:labor; period=i),
            CommodityRef(:money; period=i),
        ],
        observed_variables=[rate_refs[i]],
        activity_start=50.0,
        name=Symbol("firm$i"),
    )
end

# Terminal production period n_p-1: r_(n_p-1) -> +Inf.
# The terminal firm's money demand therefore approaches zero.
terminal_alpha = alpha[n_prod]
terminal_firm_spec = ActivityDemandSpec(
    (activity, prices, observed_values) -> [
        activity / terminal_alpha,
        zero(activity),
    ],
)

add_agent!(
    model,
    terminal_firm_spec;
    outputs=terminal_product,
    output_coefficients=1.0,
    demands=[
        terminal_labor,
        CommodityRef(:money; period=n_prod),
    ],
    activity_start=50.0,
    name=Symbol("firm$n_prod"),
)

# Consumer
product_refs = [
    [CommodityRef(:product; period=i + 1) for i in 1:rate_count];
    terminal_product;
]

labor_refs = [
    [CommodityRef(:labor; period=i) for i in 1:rate_count];
    terminal_labor;
]

money_refs = [
    CommodityRef(:money; period=i)
    for i in 1:n_prod
]

consumer_spec = MarshallDemandConsumerSpec(
    function (income, prices, observed_values)
        rates = observed_values[1:rate_count]
        terminal_price = observed_values[n_prod]

        p_product = prices[1:n_prod]
        p_money = prices[(n_prod + 1):(2 * n_prod)]

        product_demand = [
            beta[i] * income /
            ((1.0 + rates[i]) * p_product[i])
            for i in 1:rate_count
        ]

        push!(
            product_demand,
            beta[n_prod] * income / terminal_price,
        )

        money_demand = [
            rates[i] * beta[i] * income /
            ((1.0 + rates[i]) * p_money[i])
            for i in 1:rate_count
        ]

        push!(
            money_demand,
            beta[n_prod] * income / p_money[n_prod],
        )

        return [product_demand; money_demand]
    end,
)

add_agent!(
    model,
    consumer_spec;
    demands=[product_refs; money_refs],
    endowments=[labor_refs; money_refs],
    endowment_quantities=[labor_endowment; money_endowment],
    observed_variables=[rate_refs; terminal_price_ref],
    name=:consumer,
)

# Starting values
product_start = [
    [0.6^(i - 1) for i in 1:rate_count];
    0.01;
]

labor_start = [
    [0.4 * 0.6^(i - 1) for i in 1:rate_count];
    0.01;
]

money_start = [
    100.0 * 0.6^(i - 1)
    for i in 1:n_prod
]

p0 = [
    product_start;
    labor_start;
    money_start;
]

p0[1] = 1.0

# Solve
result = solve(
    model;
    p0=p0,
    residual_tol=1.0e-8,
    silent=true,
)

# Results
interest_variables = result.agent_variable_values[1]
rates = interest_variables[1:rate_count]
terminal_interest_inclusive_price = interest_variables[n_prod]

firm_activities = [
    result.agent_variable_values[i][1]
    for i in 2:(n_prod + 1)
]

p = result.prices

labor_range = (n_prod + 1):(2 * n_prod)
money_range = (2 * n_prod + 1):(3 * n_prod)

consumer_income =
    sum(p[labor_range] .* labor_endowment) +
    sum(p[money_range] .* money_endowment)

product_consumption = [
    [
        beta[i] * consumer_income /
        ((1.0 + rates[i]) * p[i])
        for i in 1:rate_count
    ];
    beta[n_prod] * consumer_income /
    terminal_interest_inclusive_price;
]

println()
println("========== GEMB parameterized endogenous-interest equilibrium ==========")
println("Total periods:                      ", n_p)
println("Production periods:                 ", n_prod)
println("Solved:                             ", result.solved)
println("Beta:                               ", beta)
println("Interest rates:                     ", rates)
println("Terminal interest-inclusive price:  ", terminal_interest_inclusive_price)
println("Equilibrium prices:                 ", p)
println("Firm activities (outputs):          ", firm_activities)
println("Product consumption:                ", product_consumption)
println("Max natural residual:               ", result.max_natural_residual)
println("========================================================================")
