using GeneralEquilibriumModeling

beta1 = 0.05
beta2 = 0.05
alpha1 = 120.0
alpha2 = 100.0

omega1 = 1000.0
omega2 = 100.0

tau = 1.0

consumer1_demand = (income, p) -> begin
    p1, p3 = p
    x3 = alpha1 - p3 / (beta1 * p1)
    x1 = (income - p3 * x3) / p1
    [x1, x3]
end

consumer2_demand = (income, p) -> begin
    p1, p2 = p
    x2 = alpha2 - p2 / (beta2 * p1)
    x1 = (income - p2 * x2) / p1
    [x1, x2]
end

model = GEMB.GEMBModel(
    [
        :corn,
        GEMB.CommoditySpec(
            :iron;
            price_lower_bound=1.0e-8,
        ),
        GEMB.CommoditySpec(
            :taxed_iron;
            price_lower_bound=1.0e-8,
        ),
        GEMB.CommoditySpec(
            :tax_certificate;
            price_lower_bound=1.0e-8,
        ),
    ];
    numeraire=:corn,
    numeraire_value=1.0,
)

GEMB.add_agent!(
    model,
    GEMB.MarshallDemandConsumerSpec(consumer1_demand);
    demands=[:corn, :taxed_iron],
    endowments=:corn,
    endowment_quantities=omega1,
    name=:consumer1,
)

GEMB.add_agent!(
    model,
    GEMB.MarshallDemandConsumerSpec(consumer2_demand);
    demands=[:corn, :iron],
    endowments=:iron,
    endowment_quantities=omega2,
    name=:consumer2,
)

GEMB.add_agent!(
    model,
    GEMB.CESSpec(
        [1.0];
        es=0.0,
        alpha=1.0,
    );
    outputs=:taxed_iron,
    demands=:iron,
    claim_rate=tau,
    claim=GEMB.CommodityRef(:tax_certificate),
    activity_start=100.0,
    name=:tax_agency,
)

GEMB.add_agent!(
    model,
    GEMB.CESSpec(
        [1.0];
        es=0.0,
        alpha=1.0,
    );
    demands=:corn,
    endowments=:tax_certificate,
    endowment_quantities=1.0,
    activity_start=100.0,
    name=:government,
)

result = GEMB.solve(
    model;
    p0=[1.0, 1.5, 3.0, 60.0],
    residual_tol=1.0e-10,
    silent=true,
)

stats = GEMB.equilibrium_statistics(model, result)

GEMB.print_equilibrium_statistics(model, result)