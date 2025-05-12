from ax import Data, OrderConstraint, SumConstraint
from ax.core.experiment import Experiment
from ax.core.search_space import SearchSpace
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.metrics.noisy_function import GenericNoisyFunctionMetric
from ax.runners.synthetic import SyntheticRunner
from ax.modelbridge.registry import Models
from ax.service.utils.report_utils import exp_to_df
import numpy as np

import packer

# --- Define search space ---
K_SLOTS = 4

GPU_TYPES = ["DGX-B300", "H20"]
INVENTORY = {"DGX-B300": 32, "H20": 64}

pack = packer.Packer(
    gpu_types=GPU_TYPES,
)

parameters = []
parameter_constraints = []

# Island slots
for gpu_type in GPU_TYPES:
    # GPU Sum
    gpu_sum = []

    # For each GPU type, create K_SLOTS
    for k in range(K_SLOTS):
        # GPU Count
        gpu_count = RangeParameter(
            name=f"count_{gpu_type}_{k}", parameter_type=ParameterType.INT,
            lower=0, upper=INVENTORY[gpu_type] / 4,
        )
        # GPU TP
        gpu_tp = RangeParameter(
            name=f"tp_{gpu_type}_{k}", parameter_type=ParameterType.INT,
            lower=1, upper=4,
        )

        # Limit TP to count
        parameter_constraints += [
            OrderConstraint(
                lower_parameter=gpu_tp,
                upper_parameter=gpu_count,
            )
        ]
        
        # Add parameters to search space
        parameters.append(gpu_count)
        parameters.append(gpu_tp)

        # Add to sum
        gpu_sum.append(gpu_count)

    # Constrain GPU count per type to inventory
    parameter_constraints += [
        SumConstraint(
            parameters=gpu_sum,
            is_upper_bound=True,
            bound=INVENTORY[gpu_type],
        ),
    ]

search_space = SearchSpace(parameters=parameters, parameter_constraints=parameter_constraints)

# --- Inner-loop evaluator ---
def eval_config_outer(params):

    print("\n\n***** Evaluating configuration *****")

    # Inventory constraint: count GPUs used per type
    for gpu_type in GPU_TYPES:
        total_count = 0
        for k in range(K_SLOTS):
            total_count += params[f"count_{gpu_type}_{k}"]
        percent_used = 100 * total_count / INVENTORY[gpu_type]
        print(f"GPU usage: {gpu_type} {percent_used:.2f}%")
        if total_count > INVENTORY[gpu_type]:
            print(f"ERROR - Configuration rejected: {gpu_type} used {total_count} > {INVENTORY[gpu_type]} available")
            return 1e9

    slots = []
    for gpu_type in GPU_TYPES:
        for k in range(K_SLOTS):
            # Extract island parameters
            count = params[f"count_{gpu_type}_{k}"]
            tp = params[f"tp_{gpu_type}_{k}"]

            # Calculate dp (count / tp, whole number)
            dp = count // tp if tp > 0 else 0

            # Skip unused slots
            if count <= 0 or tp <= 0 or dp <= 0:
                print(f"Slot {k} not used: gpu_type={gpu_type}, count={count}, tp={tp}, dp={dp}")
                continue
            print(f"Slot {k} used: gpu_type={gpu_type}, count={count}, tp={tp}, dp={dp}")

            # Create island
            island = packer.Island(
                id=f"{gpu_type}_{k}",
                gpu_type=gpu_type,
                role="prefill",
                dp=dp,
                tp=tp,
            )

            # Add island to the list
            slots.append(island)

    # Evaluate configuration
    prefill_speed = pack.solve_prefill(slots, 1000)
    rho_max = prefill_speed if prefill_speed != np.inf else 1e9

    print(f"***** rho_max = {rho_max:.2f} *****")

    return rho_max

# --- Optimization config ---
optimization_config = OptimizationConfig(
    objective=Objective(
        metric=GenericNoisyFunctionMetric(
                name="rho_max",
                f=lambda param_dict: eval_config_outer(param_dict),
                noise_sd=1e-4,
                lower_is_better=True,
        ),
        minimize=True,
    )
)

# --- Build experiment ---
experiment = Experiment(
    name="gpu_island_scheduler",
    search_space=search_space,
    optimization_config=optimization_config,
    runner=SyntheticRunner(),
)

# --- Initialization ---
N_INIT = 10

def initialize_experiment(exp, n_init):
    sobol = Models.SOBOL(search_space=exp.search_space)
    batch = sobol.gen(n_init)
    trial = exp.new_batch_trial(generator_run=batch)
    trial.run()
    return exp.fetch_data()

data = initialize_experiment(experiment, N_INIT)

# --- SAASBO loop ---
BATCH_SIZE = 3
N_BATCH = 4

for i in range(N_BATCH):
    # Build SAAS surrogate with NUTS
    model = Models.SAASBO(experiment=experiment, data=data)

    # Generate next candidates with SAASBO
    generator_run = model.gen(BATCH_SIZE)
    trial = experiment.new_batch_trial(generator_run=generator_run)
    trial.run()

    # Update data
    new_data = trial.fetch_data()
    for _, row in new_data.df.iterrows():
        print(f"Iteration {i+1}, arm {row['arm_name']}: rho_max = {row['mean']}")

    data = Data.from_multiple_data([data, new_data])

    # Print intermediate results
    print("\nIntermediate results:")
    print(data.df[["arm_name", "mean"]].sort_values(by="mean"))
    print("\n")

    # Print best roh
    best = data.df.loc[data.df["mean"].idxmin()]
    print(f"Best rho_max: {best['mean']} for arm {best['arm_name']}")

# --- Final results ---
df = exp_to_df(experiment).sort_values(by=["trial_index"])
best = df.loc[df["rho_max"].idxmin()]
print("Best parameters:")
print(best.drop(["trial_index"]))