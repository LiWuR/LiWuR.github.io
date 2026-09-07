using GeneralEquilibriumModeling

alpha1 = 2.5
alpha2 = 50.0
theta = 1.0
omega = 10.0
beta = 0.5

# Degree-one homogenization of firm 2:
# g(q, l, cs) = alpha2 * cs * l / (cs + theta * q)
production_function = inputs -> begin
    q, l, cs = inputs
    alpha2 * cs * l / (cs + theta * q)
end

marginal_product_function = inputs -> begin
    q, l, cs = inputs
    denominator = cs + theta * q

    [
        -theta * alpha2 * cs * l / denominator^2,
        alpha2 * cs / denominator,
        theta * alpha2 * q * l / denominator^2,
    ]
end

model = GEMB.GEMBModel(
    [
        GEMB.CommoditySpec(
            :product1;
            price_lower_bound=1.0e-10,
        ),
        :product2,
        GEMB.CommoditySpec(
            :pollution;
            price_lower_bound=-Inf,
            price_upper_bound=Inf,
        ),
        GEMB.CommoditySpec(
            :labor;
            price_lower_bound=1.0e-10,
        ),
        GEMB.CommoditySpec(
            :scale_claim;
            price_lower_bound=1.0e-10,
        ),
    ];
    numeraire=:product2,
    numeraire_value=1.0,
)

# Firm 1:
# one unit of labor produces alpha1 units of product 1
# and alpha1 units of pollution.
firm1_spec = GEMB.ActivityDemandSpec(
    (z, prices) -> [z],
)

GEMB.add_agent!(
    model,
    firm1_spec;
    outputs=[:product1, :pollution],
    output_coefficients=[alpha1, alpha1],
    demands=:labor,
    activity_start=3.0,
    name=:firm1,
)

# Firm 2:
# pollution, labor, and scale claim are the three inputs.
GEMB.add_agent!(
    model,
    GEMB.ProductionFunctionSpec(
        production_function,
        marginal_product_function;
        behavior=:stationary,
    );
    outputs=:product2,
    demands=[:pollution, :labor, :scale_claim],
    activity_start=60.0,
    demand_start=[3.0, 6.0, 0.8],
    demand_lower_bounds=[1.0e-10, 1.0e-10, 1.0e-10],
    production_multiplier_start=0.8,
    name=:firm2,
)

# Consumer:
# Cobb-Douglas preferences over products 1 and 2,
# with 10 units of labor and one unit of scale claim.
GEMB.add_agent!(
    model,
    GEMB.CESSpec(
        [beta, 1.0 - beta];
        es=1.0,
        alpha=1.0,
    );
    demands=[:product1, :product2],
    endowments=[:labor, :scale_claim],
    endowment_quantities=[omega, 1.0],
    activity_start=15.0,
    name=:consumer,
)

result = GEMB.solve(
    model;
    p0=[10.0, 1.0, -5.0, 5.0, 20.0],
    residual_tol=1.0e-9,
    silent=true,
)

GEMB.print_equilibrium_statistics(
    model,
    result;
    display_tol=1.0e-9,
    sigdigits=8,
)

l1 = result.agent_variable_values[1][1]

_, pollution_input, l2, scale_claim_input, _ =
    result.agent_variable_values[2]

w = omega * result.prices[4] +
    result.prices[5]

println("\nLabor inputs:")
println("l1 = ", l1)
println("l2 = ", l2)

println("\nFirm 2 inputs:")
println("pollution = ", pollution_input)
println("scale claim = ", scale_claim_input)

println("\nPrices:")
println(result.prices)

println("\nConsumer income:")
println(w)
