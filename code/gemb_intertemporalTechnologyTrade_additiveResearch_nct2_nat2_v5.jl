using GeneralEquilibriumModeling.GEMB
using Plots

np = 9

rho = 0.8
alpha1 = 0.01
alpha_bar = 0.0001
omega = 100.0

np >= 3 || throw(ArgumentError("np must be at least 3."))

nt = np - 2

technology(t) = CommodityRef(:technology; period=t)
labor(t) = CommodityRef(:labor; period=t)

utility_weights = vcat(
    rho .^ (3:np),
    rho .^ (2:(np - 1)),
)

# ------------------------------------------------------------
# Starting values
# ------------------------------------------------------------

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
    omega .-
    research_labor_start

technology_p0 = [
    rho^(t - 1) *
    production_labor_start[1] /
    alpha_start[t]
    for t in 2:(np - 1)
]

labor_p0 = [
    rho^(t - 1) *
    production_labor_start[1] /
    production_labor_start[t]
    for t in 1:(np - 2)
]

p0 = vcat(
    technology_p0,
    labor_p0,
)

multiplier_start =
    rho^2 / production_labor_start[1]

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

model = GEMBModel(
    [
        CommoditySpec(
            :technology;
            axes=(period=2:(np - 1),),
            price_lower_bound=0.0,
        ),
        CommoditySpec(
            :labor;
            axes=(period=1:(np - 2),),
            price_lower_bound=0.0,
        ),
    ];
    numeraire=labor(1),
)

research_spec = ActivityDemandSpec(
    (activity, prices) -> [
        activity,
    ],
)

for t in 1:(np - 2)
    add_agent!(
        model,
        research_spec;
        outputs=[
            technology(s)
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

add_agent!(
    model,
    CESMarginalUtilitySpec(
        utility_weights;
        es=1.0,
    );
    demands=[
        CommodityRef(:technology),
        CommodityRef(:labor),
    ],
    endowments=[
        CommodityRef(:technology),
        CommodityRef(:labor),
    ],
    endowment_quantities=vcat(
        fill(alpha1, nt),
        fill(omega, nt),
    ),
    demand_start=vcat(
        alpha_start[2:end],
        production_labor_start,
    ),
    demand_lower_bounds=fill(
        1.0e-12,
        2 * nt,
    ),
    multiplier_start=multiplier_start,
    multiplier_lower_bound=0.0,
    name=:consumer,
)

# ------------------------------------------------------------
# Solve
# ------------------------------------------------------------

result = solve(
    model;
    p0=p0,
    residual_tol=1.0e-9,
    silent=true,
)

result.solved || error("Equilibrium solve failed.")

stats = equilibrium_statistics(model, result)

print_equilibrium_statistics(model, result)

# ------------------------------------------------------------
# Technology path
# ------------------------------------------------------------

research_labor = stats.activity_levels

alpha_path =
    alpha1 .+
    alpha_bar .* cumsum(research_labor)

display(
    plot(
        2:(np - 1),
        alpha_path;
        mark=:circle,
        xlabel="Period",
        ylabel="alpha^(t)",
        label="alpha^(t)",
        legend=:bottomright,
        grid=true,
        gridalpha=1.0,
    ),
)