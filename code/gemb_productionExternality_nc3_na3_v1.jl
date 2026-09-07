using GeneralEquilibriumModeling

alpha1 = 2.5
alpha2 = 50.0
theta = 1.0
omega = 10.0
beta = 0.5

model = GEMB.GEMBModel(
    [
        GEMB.CommoditySpec(
            :product1;
            price_lower_bound=1.0e-10,
        ),
        :product2,
        GEMB.CommoditySpec(
            :labor;
            price_lower_bound=1.0e-10,
        ),
    ];
    numeraire=:product2,
    numeraire_value=1.0,
)

# Firm 1:
# one unit of labor produces alpha1 units of product 1.
firm1_spec = GEMB.ActivityDemandSpec(
    (z, prices) -> [z],
)

GEMB.add_agent!(
    model,
    firm1_spec;
    outputs=:product1,
    output_coefficients=alpha1,
    demands=:labor,
    activity_start=3.0,
    name=:firm1,
)

# Firm 2:
# its labor requirement depends on firm 1's activity through
# the production externality.
firm2_spec = GEMB.ActivityDemandSpec(
    (y, prices, observed_values) -> begin
        l1 = observed_values[1]

        [
            y * (1.0 + theta * alpha1 * l1) / alpha2
        ]
    end,
)

GEMB.add_agent!(
    model,
    firm2_spec;
    outputs=:product2,
    demands=:labor,
    observed_variables=[
        GEMB.agent_variable_ref(:firm1, :activity),
    ],
    activity_start=20.0,
    name=:firm2,
)

# Consumer:
# Cobb-Douglas preferences over products 1 and 2.
GEMB.add_agent!(
    model,
    GEMB.CESSpec(
        [beta, 1.0 - beta];
        es=1.0,
        alpha=1.0,
    );
    demands=[:product1, :product2],
    endowments=:labor,
    endowment_quantities=omega,
    activity_start=10.0,
    name=:consumer,
)

result = GEMB.solve(
    model;
    p0=[1.0, 1.0, 2.0],
    residual_tol=1.0e-9,
    silent=true,
)

GEMB.print_equilibrium_statistics(
    model,
    result;
    display_tol=1.0e-9,
    sigdigits=8,
)

stats = GEMB.equilibrium_statistics(model, result)

l1 = stats.activity_levels[1]
y2 = stats.activity_levels[2]
l2 = y2 * (1.0 + theta * alpha1 * l1) / alpha2
w = omega * result.prices[3]

p_full = [
    result.prices[1],
    result.prices[2],
    0.0,
    result.prices[3],
]

println("\nLabor allocation:")
println("l1 = ", l1)
println("l2 = ", l2)

println("\nFull price vector (including pollution price p3 = 0):")
println(p_full)

println("\nConsumer income:")
println(w)
