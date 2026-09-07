using GeneralEquilibriumModeling.GEMB
using Plots

np = 9

rho = 0.8
alpha1 = 0.01
alpha_bar = 0.0001
omega = 100.0

np >= 3 || throw(ArgumentError("np must be at least 3."))

product(t) = CommodityRef(:product; period=t)
labor(t) = CommodityRef(:labor; period=t)
parameter(t) = CommodityRef(:parameter; period=t)
scale_claim(t) = CommodityRef(:scale_claim; period=t)

alpha_star =
    rho * alpha_bar * omega / (1 - rho)

alpha_target_start = if alpha1 < alpha_star
    alpha_star .-
    (alpha_star - alpha1) .* rho .^ (0:(np - 2))
else
    fill(alpha1, np - 1)
end

research_labor_start = clamp.(
    diff(alpha_target_start) ./ alpha_bar,
    0.0,
    0.9 * omega,
)

alpha_start =
    alpha1 .+
    alpha_bar .* vcat(
        0.0,
        cumsum(research_labor_start),
    )

production_labor_start =
    vcat(
        omega .- research_labor_start,
        omega,
    )

product_start =
    alpha_start .* production_labor_start

product_p0 = [
    rho^(t - 2) *
    product_start[1] /
    product_start[t - 1]
    for t in 2:np
]

labor_p0 =
    product_p0 .* alpha_start

parameter_p0 =
    product_p0 .* production_labor_start

scale_claim_p0 =
    -product_p0 .* product_start

p0 = vcat(
    product_p0,
    labor_p0,
    parameter_p0,
    scale_claim_p0,
)

production_function = inputs -> begin
    a, x, cs = inputs
    a * x / cs
end

marginal_product_function = inputs -> begin
    a, x, cs = inputs
    [x * cs, a * cs, -a * x]
end

production_spec = ProductionFunctionSpec(
    production_function,
    marginal_product_function;
    behavior=:stationary,
)

research_spec = ActivityDemandSpec(
    (activity, prices) -> [
        activity,
    ],
)

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
            :parameter;
            axes=(period=1:(np - 1),),
            price_lower_bound=0.0,
        ),
        CommoditySpec(
            :scale_claim;
            axes=(period=1:(np - 1),),
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
            parameter(t),
            labor(t),
            scale_claim(t),
        ],
        activity_start=product_start[t],
        demand_start=[
            alpha_start[t],
            production_labor_start[t],
            1.0,
        ],
        demand_lower_bounds=[
            1.0e-12,
            1.0e-12,
            1.0e-12,
        ],
        production_multiplier_start=product_p0[t],
        name=Symbol("production_$t"),
    )
end

for t in 1:(np - 2)
    add_agent!(
        model,
        research_spec;
        outputs=[
            parameter(s)
            for s in (t + 1):(np - 1)
        ],
        output_coefficients=fill(
            alpha_bar,
            np - t - 1,
        ),
        demands=labor(t),
        activity_start=research_labor_start[t],
        activity_lower_bound=0.0,
        activity_upper_bound=omega,
        name=Symbol("research_$t"),
    )
end

utility_weights =
    rho .^ (2:np)

add_agent!(
    model,
    CESMarginalUtilitySpec(
        utility_weights;
        es=1.0,
    );
    demands=CommodityRef(:product),
    endowments=[
        CommodityRef(:labor),
        CommodityRef(:parameter),
        CommodityRef(:scale_claim),
    ],
    endowment_quantities=vcat(
        fill(omega, np - 1),
        fill(alpha1, np - 1),
        fill(1.0, np - 1),
    ),
    demand_start=product_start,
    demand_lower_bounds=fill(
        1.0e-12,
        np - 1,
    ),
    multiplier_start=
        rho^2 / product_start[1],
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
print_equilibrium_statistics(model, result)

research_labor =
    stats.activity_levels[np:(2 * np - 3)]

alpha_path =
    alpha1 .+
    alpha_bar .* cumsum(research_labor)

display(
    plot(
        2:(np - 1),
        alpha_path;
        xlabel="Period",
        ylabel="alpha^(t)",
        label="alpha^(t)",
        legend=:bottomright,
        marker=:circle,
        grid=true,
        gridalpha=1.0,
    ),
)
