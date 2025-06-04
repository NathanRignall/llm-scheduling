import math
import numpy as np

from ax import Arm, Data, Experiment # SumConstraint likely not needed anymore
from ax.core.search_space import SearchSpace
from ax.core.parameter import ParameterType, ChoiceParameter, RangeParameter
from ax.core.metric import Metric
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.metrics.noisy_function import GenericNoisyFunctionMetric
from ax.runners.synthetic import SyntheticRunner # Using SyntheticRunner as in your example
from ax.modelbridge.registry import Models
# from ax.modelbridge.cross_validation import cross_validate, compute_diagnostics # For model diagnostics if needed
from ax.service.utils.report_utils import exp_to_df # For nice output
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from ax.models.torch.botorch_modular.surrogate import SurrogateSpec, ModelConfig # Corrected import path
from ax.models.torch.botorch_modular.surrogate import Surrogate

import packer
import evaluator

# --- Island Divider Function ---
def island_divider_v2(
    total_gpus_for_type: int,
    num_islands_target: int,
    size_distribution_skew_exponent: float,
    min_island_size: int = 1
) -> list[int]:
    if min_island_size <= 0:
        raise ValueError("Minimum island size must be positive.")
    if total_gpus_for_type < 0:
        raise ValueError("Total GPUs for type cannot be negative.")
    if num_islands_target < 0:
        raise ValueError("Target number of islands cannot be negative.")

    if num_islands_target == 0:
        return []
    gpus_to_allocate = total_gpus_for_type
    if gpus_to_allocate == 0:
        return []
    if gpus_to_allocate < min_island_size:
        return []
        
    max_possible_islands_with_min_size = math.floor(gpus_to_allocate / min_island_size)
    actual_num_islands = min(num_islands_target, max_possible_islands_with_min_size)
    if actual_num_islands == 0 : # Handles cases where num_islands_target was > 0 but max_possible is 0
        return []


    base_allocation_per_island = min_island_size
    gpus_used_for_base = actual_num_islands * base_allocation_per_island
    remaining_gpus_to_distribute = gpus_to_allocate - gpus_used_for_base
    final_island_sizes = [base_allocation_per_island] * actual_num_islands

    if remaining_gpus_to_distribute < 0: # Should be prevented by actual_num_islands logic
        if actual_num_islands == 1 and gpus_to_allocate >= min_island_size:
             return [gpus_to_allocate]
        return [] 
    if remaining_gpus_to_distribute == 0:
        return final_island_sizes
    if actual_num_islands == 1:
        final_island_sizes[0] += remaining_gpus_to_distribute
        return final_island_sizes

    weights = []
    for i in range(actual_num_islands):
        base_val = float(i + 1) 
        if math.isclose(size_distribution_skew_exponent, 0.0):
            weights.append(1.0)
        else:
            weights.append(math.pow(base_val, size_distribution_skew_exponent))
    
    sum_weights = sum(weights)
    if math.isclose(sum_weights, 0.0) or sum_weights <= 0:
        proportions = [1.0 / actual_num_islands] * actual_num_islands
    else:
        proportions = [w / sum_weights for w in weights]

    additional_allocations_float = [p * remaining_gpus_to_distribute for p in proportions]
    additional_allocations_int = [math.floor(val) for val in additional_allocations_float]
    current_sum_additional_int = sum(additional_allocations_int)
    discrepancy_to_distribute = int(round(remaining_gpus_to_distribute - current_sum_additional_int))

    for i in range(actual_num_islands):
        final_island_sizes[i] += additional_allocations_int[i]

    if discrepancy_to_distribute > 0:
        fractional_parts = [(val - math.floor(val), idx) for idx, val in enumerate(additional_allocations_float)]
        fractional_parts.sort(key=lambda x: x[0], reverse=True)
        for k in range(discrepancy_to_distribute):
            island_idx_to_increment = fractional_parts[k % actual_num_islands][1]
            final_island_sizes[island_idx_to_increment] += 1
    elif discrepancy_to_distribute < 0:
        pass # Not attempting to decrement for simplicity here

    return final_island_sizes

# --- Define constants ---
GPU_TYPES = ["DGX-B300", "H200"]
INVENTORY = {"DGX-B300": 128, "H200": 256} # Example inventory, adjust as needed
MIN_ISLAND_SIZE_GLOBAL = 4 # Or a more sensible minimum like 4, 8, etc.

pack = packer.Packer(
    gpu_types=GPU_TYPES,
)
trace_pdf = evaluator.load_trace_pdf("traces/generated_trace_pdf.csv")

# --- Define New Search Space ---
parameters = []
parameter_constraints = [] # Likely empty with the divider function approach

for gpu_type in GPU_TYPES:
    if INVENTORY.get(gpu_type, 0) <= 0:
        print(f"Skipping BO parameters for {gpu_type} due to zero or negative inventory.")
        continue

    max_islands_for_this_type = INVENTORY[gpu_type] // MIN_ISLAND_SIZE_GLOBAL
    # Ensure upper bound for num_islands is at least 1 if any islands are possible
    upper_bound_num_islands = max(1, max_islands_for_this_type)
    # If max_islands_for_this_type is 0, upper_bound_num_islands becomes 1.
    # The divider will return [] if 1 island of MIN_ISLAND_SIZE_GLOBAL is not possible.

    parameters.append(
        RangeParameter(
            name=f"num_islands_{gpu_type}",
            parameter_type=ParameterType.INT,
            lower=1, # Try to make at least one island
            upper=upper_bound_num_islands
        )
    )
    parameters.append(
        RangeParameter(
            name=f"skew_exponent_{gpu_type}",
            parameter_type=ParameterType.FLOAT,
            lower=-3.0, # Expanded range for skew, adjust as needed
            upper=3.0,
        )
    )

if not parameters:
    raise ValueError("No parameters defined for BO. Check INVENTORY and MIN_ISLAND_SIZE_GLOBAL.")

search_space = SearchSpace(
    parameters=parameters,
    parameter_constraints=parameter_constraints
)
print(f"Defined Ax Search Space with parameters: {[p.name for p in parameters]}")

# --- Inner-loop evaluator ---
def eval_config_outer(params_dict):
    print(f"\n\n***** Evaluating configuration (divider v2) with params: {params_dict} *****")
    all_generated_islands = []

    for gpu_type in GPU_TYPES:
        num_islands_param_name = f"num_islands_{gpu_type}"
        skew_exponent_param_name = f"skew_exponent_{gpu_type}"

        if num_islands_param_name not in params_dict or skew_exponent_param_name not in params_dict:
            if INVENTORY.get(gpu_type, 0) > 0:
                 print(f"Warning: Parameters for {gpu_type} not in params_dict. Skipping island generation for it.")
            continue

        num_islands_param = params_dict[num_islands_param_name]
        skew_exponent_param = params_dict[skew_exponent_param_name]
        current_inventory_for_type = INVENTORY[gpu_type]

        # print(f"GPU type: {gpu_type}, Total Inventory: {current_inventory_for_type}")
        # print(f"  Divider params: num_islands_target={num_islands_param}, skew_exponent={skew_exponent_param}, min_size={MIN_ISLAND_SIZE_GLOBAL}")

        island_sizes_for_type = island_divider_v2(
            total_gpus_for_type=current_inventory_for_type,
            num_islands_target=num_islands_param,
            size_distribution_skew_exponent=skew_exponent_param,
            min_island_size=MIN_ISLAND_SIZE_GLOBAL
        )
        print(f"  Generated island sizes for {gpu_type}: {island_sizes_for_type} (Sum: {sum(island_sizes_for_type)})")

        for size_val in island_sizes_for_type:
            if size_val > 0:
                # Using the MockEvaluator.Island which handles its own ID generation
                island = evaluator.Island(gpu_type=gpu_type, size=size_val)
                all_generated_islands.append(island)

    if not all_generated_islands:
        print("ERROR - No islands were generated. Configuration rejected.")
        return {"rho_max": 1e9} # Ax expects a dictionary of metric names to values

    islands_dict = {island.id: island for island in all_generated_islands}
    
    # print(f"Calling packer.solve_linear with {len(islands_dict)} islands...")
    model_solution, prefill_throughput, decode_throughput, delta, objective_val = pack.solve_linear(
        islands_dict, trace_pdf, resolution=100, print_debug=False # Set print_debug as needed
    )

    if prefill_throughput is None or decode_throughput is None or prefill_throughput <= 1e-6 or decode_throughput <= 1e-6:
        print(f"ERROR - Configuration rejected: prefill ({prefill_throughput}) or decode ({decode_throughput}) throughput is None or non-positive.")
        return 1e9 # Ax expects a dictionary of metric names to values

    throughput = min(prefill_throughput, decode_throughput)
    rho_max = 1.0 / throughput
    print(f"***** rho_max = {rho_max:.4f} (Throughput: {throughput:.4f}) *****")
    return rho_max

# --- Optimization config ---
optimization_config = OptimizationConfig(
    objective=Objective(
        metric=GenericNoisyFunctionMetric(
                name="rho_max",
                f=eval_config_outer, # Use the new eval_config_outer
                noise_sd=None, # Assuming deterministic for now, or estimate (e.g., 1e-4)
                lower_is_better=True,
        ),
        minimize=True,
    )
)

# --- Build experiment ---
experiment = Experiment(
    name="gpu_island_scheduler_v2_divider",
    search_space=search_space,
    optimization_config=optimization_config,
    runner=SyntheticRunner(), # eval_config_outer will be called directly
)

# --- Initialization ---
def generate_okay_configs_v2(n_configs: int = 1, search_space_obj=None):
    if search_space_obj is None:
        raise ValueError("search_space_obj must be provided to generate_okay_configs_v2")

    configs = []
    for i in range(n_configs):
        cfg = {}
        config_possible_for_at_least_one_gpu = False
        for gpu_type in GPU_TYPES:
            num_islands_param_name = f"num_islands_{gpu_type}"
            skew_exponent_param_name = f"skew_exponent_{gpu_type}"

            if num_islands_param_name in search_space_obj.parameters and \
               skew_exponent_param_name in search_space_obj.parameters:
                
                num_islands_param_def = search_space_obj.parameters[num_islands_param_name]
                skew_exponent_param_def = search_space_obj.parameters[skew_exponent_param_name]
                
                cfg[num_islands_param_name] = np.random.randint(
                    num_islands_param_def.lower,
                    num_islands_param_def.upper + 1
                )
                cfg[skew_exponent_param_name] = np.random.uniform(
                    skew_exponent_param_def.lower,
                    skew_exponent_param_def.upper
                )
                config_possible_for_at_least_one_gpu = True
        
        if config_possible_for_at_least_one_gpu and cfg:
            print(f"Generated initial configuration (v2) {i+1}: {cfg}")
            configs.append(cfg)
        elif not config_possible_for_at_least_one_gpu:
            print(f"Warning: Could not generate initial config {i+1} as no valid BO parameters seem to be defined for any GPU type in the search space.")
    return configs

def add_initial_arms_to_experiment(exp_obj, initial_params_list):
    for idx, params_dict in enumerate(initial_params_list):
        if not params_dict:
            print(f"Skipping empty initial parameter set {idx}.")
            continue
        arm = Arm(name=f"warm_start_{idx}", parameters=params_dict)
        # For SyntheticRunner, new_trial().add_arm().run() is typical
        trial = exp_obj.new_trial()
        trial.add_arm(arm)
        trial.run() # This will call eval_config_outer

# Generate initial configurations
initial_parameters_list = generate_okay_configs_v2(n_configs=5, search_space_obj=experiment.search_space)

if initial_parameters_list:
    add_initial_arms_to_experiment(experiment, initial_parameters_list)
    # Fetch data from the initial trials. This is important for the model.
    data = experiment.fetch_data()
    print("\n--- Initial Warm-Start Data ---")
    print(data.df)
else:
    print("No initial parameters generated. Starting BO without warm-start.")
    data = Data() # Start with empty data if no initial trials

# --- SAASBO loop ---
BATCH_SIZE = 5
NUM_SAMPLES = 256
WARMUP_STEPS = 512
N_ITERATIONS = 20

print(f"\n--- Starting SAASBO Optimization ({N_ITERATIONS} iterations, {BATCH_SIZE} batch size) ---")

# Configure the SAASBO model
# Using SurrogateSpec for more explicit configuration
saas_model_config = ModelConfig(
    botorch_model_class=SaasFullyBayesianSingleTaskGP,
    mll_options={ # Options passed to fit_gpytorch_mll or similar for MCMC
        "num_samples": NUM_SAMPLES,
        "warmup_steps": WARMUP_STEPS,
        # "thinning": 16, # Optional: for MCMC
        # "disable_progbar": True, # Optional
    },
    # model_options can be used to pass kwargs directly to SaasFullyBayesianSingleTaskGP constructor if needed
)
surrogate = Surrogate(
    botorch_model_class=saas_model_config.botorch_model_class,
    model_options=saas_model_config.model_options, # Pass constructor options if any
    mll_options=saas_model_config.mll_options    # Pass MLL fitting options
)


for i in range(N_ITERATIONS):
    print(f"\n--- Iteration {i + 1}/{N_ITERATIONS} ---")
    # Re-initialize model at each step with all available data
    # This is standard practice for Ax service-like API and some modular examples
    model = Models.BOTORCH_MODULAR(
        experiment=experiment,
        data=data, # Use all data collected so far
        surrogate=surrogate, # Pass the configured surrogate
        # search_space=experiment.search_space, # Usually inferred from experiment
    )

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

# save intermediate results to CSV
df.to_csv("data/scratch/gpu_island_scheduler_results.csv", index=False)