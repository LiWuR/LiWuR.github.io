using GeneralEquilibriumModeling

input_coefficients = [
    8.0  6.0  1.0
    4.0  2.0  1.5
    2.0  1.5  0.5
]

output_coefficients = [60.0, 30.0, 20.0]
factor_endowments = [48.0, 20.0, 8.0]
factors = [:factor_1, :factor_2, :factor_3]

model = GEMB.GEMBModel(
    [:product, factors...];
    numeraire=:product,
)

for j in 1:3
    coefficients = input_coefficients[:, j]

    production = GEMB.ActivityDemandSpec(
        (z, p) -> z .* coefficients,
    )

    GEMB.add_agent!(
        model,
        production;
        outputs=:product,
        output_coefficients=output_coefficients[j],
        demands=factors,
        activity_start=100.0,
        name=Symbol("firm_", j),
    )
end

consumer = GEMB.ActivityDemandSpec(
    (u, p) -> [u],
)

GEMB.add_agent!(
    model,
    consumer;
    demands=:product,
    endowments=factors,
    endowment_quantities=factor_endowments,
    activity_start=100.0,
    name=:consumer,
)

result = GEMB.solve(
    model;
    silent=true,
)

GEMB.print_equilibrium_statistics(model, result)