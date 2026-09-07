using GeneralEquilibriumModeling.GEM
using GeneralEquilibriumModeling.GEMB
import GeneralEquilibriumModeling.GEMB: activity_supply

s1 = 100.0
fixed_input_price = 2.0

struct EndogenousOutputSpec <: AbstractActivitySupplySpec end

function activity_supply(
    ::EndogenousOutputSpec,
    activity,
    prices,
    observed_values,
)
    [observed_values[1]]
end

model = GEMBModel(
    [:input, :output];
    numeraire=:output,
    numeraire_value=1.0,
)

add_agent!(
    model,
    ConditionAgentSpec(
        (variables, observed_values) -> [
            observed_values[1] - fixed_input_price,
        ],
    );
    variable_names=:output_quantity,
    variable_start=100.0,
    variable_lower_bounds=0.0,
    variable_upper_bounds=Inf,
    observed_variables=[
        PriceVariableRef(:input),
    ],
    name=:output_condition,
)

add_agent!(
    model,
    ActivityDemandSpec(
        (activity, prices) -> [activity],
    );
    outputs=:output,
    output_spec=EndogenousOutputSpec(),
    demands=:input,
    activity_start=100.0,
    observed_variables=[
        agent_variable_ref(:output_condition, :output_quantity),
    ],
    condition_rule=TotalRevenueExpenditureBalanceConditions(),
    name=:producer,
)

add_agent!(
    model,
    MarshallDemandConsumerSpec(
        (income, prices) -> [income / prices[1]],
    );
    demands=:output,
    endowments=:input,
    endowment_quantities=[s1],
    name=:consumer,
)

result = solve(
    model;
    p0=[1.0, 1.0],
    residual_tol=1.0e-10,
    silent=true,
)

output_quantity = result.agent_variable_values[1][1]
producer_activity = result.agent_variable_values[2][1]

println("Solved: ", result.solved)
println("Prices [input, output]: ", result.prices)
println("Output quantity: ", output_quantity)
println("Producer activity: ", producer_activity)
println("Total net supply: ", result.total_net_supply)
