using GeneralEquilibriumModeling.GEMB
import GeneralEquilibriumModeling.GEM

np = 20

rho = 0.7
alpha1 = 1.0
alpha_bar = 0.005
omega = 100.0

np >= 3 || throw(ArgumentError("np must be at least 3."))

product(t) = CommodityRef(:product; period=t)
labor(t) = CommodityRef(:labor; period=t)
technology_stock(t) = CommodityRef(:technology_stock; period=t)
technology_service(t) = CommodityRef(:technology_service; period=t)
product_scale_claim(t) = CommodityRef(:product_scale_claim; period=t)
research_scale_claim(t) = CommodityRef(:research_scale_claim; period=t)

quantity_start = 100.0
research_start = 10.0

product_p0 =
    rho .^ (0:(np - 2))

labor_p0 =
    alpha1 .* product_p0

technology_service_p0 =
    quantity_start .* product_p0

technology_stock_p0 =
    copy(technology_service_p0)

for t in (np - 2):-1:1
    technology_stock_p0[t] =
        technology_service_p0[t] +
        (1 + alpha_bar * research_start) *
        technology_stock_p0[t + 1]
end

product_scale_claim_p0 =
    -quantity_start .* product_p0

research_scale_claim_p0 =
    -research_start .* alpha_bar .* technology_stock_p0[2:end]

p0 = vcat(
    product_p0,
    labor_p0,
    technology_stock_p0,
    technology_service_p0,
    product_scale_claim_p0,
    research_scale_claim_p0,
)

production_function =
    v -> v[1] * v[2] / v[3]

marginal_product_function = v -> begin
    a, x, cs = v
    [x * cs, a * cs, -a * x]
end

production_spec = ProductionFunctionSpec(
    production_function,
    marginal_product_function;
    behavior=:stationary,
)

research_net_supply = function (
    variables,
    prices,
    observed_values=Any[],
)
    ell, a, cs = variables
    [
        -a,
        -ell,
        -cs,
        a,
        a * (1 + alpha_bar * ell / cs),
    ]
end

research_conditions = function (
    variables,
    prices,
    net_supply,
)
    ell, a, cs = variables
    p_stock,
    wage,
    p_claim,
    p_service,
    p_next_stock = prices

    [
        wage -
        p_next_stock * alpha_bar * a / cs,

        p_stock -
        p_service -
        p_next_stock *
        (1 + alpha_bar * ell / cs),

        p_claim +
        p_next_stock *
        alpha_bar * a * ell / cs^2,
    ]
end

model = GEMBModel(
    [
        CommoditySpec(
            :product;
            axes=(period=2:np,),
            price_lower_bound=0.0,
        ),
        CommoditySpec(
            :labor;
            axes=(period=1:(np - 1),),
            price_lower_bound=0.0,
        ),
        CommoditySpec(
            :technology_stock;
            axes=(period=1:(np - 1),),
            price_lower_bound=0.0,
        ),
        CommoditySpec(
            :technology_service;
            axes=(period=1:(np - 1),),
            price_lower_bound=0.0,
        ),
        CommoditySpec(
            :product_scale_claim;
            axes=(period=1:(np - 1),),
            price_lower_bound=-Inf,
        ),
        CommoditySpec(
            :research_scale_claim;
            axes=(period=1:(np - 2),),
            price_lower_bound=-Inf,
        ),
    ];
    numeraire=product(2),
)

for t in 1:(np - 1)
    add_agent!(
        model,
        production_spec;
        outputs=product(t + 1),
        demands=[
            technology_service(t),
            labor(t),
            product_scale_claim(t),
        ],
        activity_start=quantity_start,
        demand_start=[
            alpha1,
            quantity_start,
            1.0,
        ],
        demand_lower_bounds=fill(1.0e-12, 3),
        production_multiplier_start=product_p0[t],
        name=Symbol("production_$t"),
    )
end

for t in 1:(np - 2)
    add_net_supply_agent!(
        model,
        research_net_supply;
        commodities=[
            technology_stock(t),
            labor(t),
            research_scale_claim(t),
            technology_service(t),
            technology_stock(t + 1),
        ],
        variable_names=[
            :activity,
            :technology_input,
            :scale_claim,
        ],
        variable_lower_bounds=[
            0.0,
            1.0e-12,
            1.0e-12,
        ],
        variable_upper_bounds=[
            omega,
            Inf,
            Inf,
        ],
        variable_start=[
            research_start,
            alpha1,
            1.0,
        ],
        condition_rule=
            GEM.ExplicitAgentConditions(
                research_conditions,
            ),
        name=Symbol("research_$t"),
    )
end

add_agent!(
    model,
    ActivityDemandSpec(
        (activity, prices) -> [activity],
    );
    outputs=technology_service(np - 1),
    demands=technology_stock(np - 1),
    activity_start=alpha1,
    activity_lower_bound=0.0,
    name=:terminal_technology,
)

add_agent!(
    model,
    CESMarginalUtilitySpec(
        rho .^ (2:np);
        es=1.0,
    );
    demands=[CommodityRef(:product)],
    endowments=[
        CommodityRef(:labor),
        technology_stock(1),
        CommodityRef(:product_scale_claim),
        CommodityRef(:research_scale_claim),
    ],
    endowment_quantities=vcat(
        fill(omega, np - 1),
        [alpha1],
        fill(1.0, np - 1),
        fill(1.0, np - 2),
    ),
    demand_start=fill(quantity_start, np - 1),
    demand_lower_bounds=fill(1.0e-12, np - 1),
    multiplier_start=
        rho^2 / quantity_start,
    multiplier_lower_bound=0.0,
    name=:consumer,
)

result = solve(
    model;
    p0=p0,
    residual_tol=1.0e-9,
    silent=true,
)

result.solved || error("Equilibrium solve failed.")

stats = equilibrium_statistics(model, result)

print_equilibrium_statistics(
    model,
    result;
    show_matrices=false,
)

research_labor =
    stats.activity_levels[np:(2 * np - 3)]

alpha_path =
    alpha1 .* cumprod(
        1 .+ alpha_bar .* research_labor,
    )
