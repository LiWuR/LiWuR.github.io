using GeneralEquilibriumModeling.GEM
using GeneralEquilibriumModeling.GEMB
import GeneralEquilibriumModeling.GEMB: activity_supply

alpha = 2.0
beta = 0.5
land_supply = 100.0
fixed_wage = 2.0

struct LaborLandSupplySpec <: AbstractActivitySupplySpec
    land_supply::Float64
end

function activity_supply(
    spec::LaborLandSupplySpec,
    activity,
    prices,
    observed_values,
)
    labor_supply = observed_values[1]
    [labor_supply, spec.land_supply]
end

model = GEMBModel(
    [
        :product,
        CommoditySpec(:labor; price_lower_bound=1.0e-6),
        CommoditySpec(:land; price_lower_bound=1.0e-6),
    ];
    numeraire=:product,
    numeraire_value=1.0,
)

add_agent!(
    model,
    ConditionAgentSpec(
        (variables, observed_values) -> [
            observed_values[1] - fixed_wage,
        ],
    );
    variable_names=:labor_supply,
    variable_start=100.0,
    variable_lower_bounds=0.0,
    variable_upper_bounds=Inf,
    observed_variables=[
        PriceVariableRef(:labor),
    ],
    name=:labor_condition,
)

add_agent!(
    model,
    CESSpec(
        [beta, 1.0 - beta];
        es=1.0,
        alpha=alpha,
    );
    outputs=:product,
    demands=[:labor, :land],
    activity_start=100.0,
    name=:firm,
)

add_agent!(
    model,
    ActivityDemandSpec(
        (activity, prices) -> [activity],
    );
    outputs=[:labor, :land],
    output_spec=LaborLandSupplySpec(land_supply),
    demands=:product,
    activity_start=100.0,
    observed_variables=[
        agent_variable_ref(:labor_condition, :labor_supply),
    ],
    condition_rule=TotalRevenueExpenditureBalanceConditions(),
    name=:consumer,
)

result = solve(
    model;
    p0=[1.0, 1.0, 1.0],
    residual_tol=1.0e-10,
    silent=true,
)

labor_supply = result.agent_variable_values[1][1]
firm_output = result.agent_variable_values[2][1]
consumer_consumption = result.agent_variable_values[3][1]

println("Solved: ", result.solved)
println("Prices [product, labor, land]: ", result.prices)
println("Labor supply: ", labor_supply)
println("Firm output: ", firm_output)
println("Consumer consumption: ", consumer_consumption)
println("Total net supply: ", result.total_net_supply)
