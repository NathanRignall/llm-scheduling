import numpy as np
import os

from ax import Arm, Data, Experiment
from ax.core.search_space import SearchSpace
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.metrics.noisy_function import GenericNoisyFunctionMetric
from ax.runners.synthetic import SyntheticRunner
from ax.modelbridge.registry import Models
from ax.service.utils.report_utils import exp_to_df
from ax.models.torch.botorch_modular.surrogate import Surrogate
from botorch.models import SingleTaskGP

import packer
import evaluator
import divider

# --- Define constants ---
GPU_TYPES = ["DGX-B300", "H200"]
INVENTORY = {"DGX-B300": 128, "H200": 256}
MIN_ISLAND_SIZE_GLOBAL = 2

# --- Initialization ---
sys_pack = packer.Packer(gpu_types=GPU_TYPES)
trace_pdf = evaluator.load_trace_pdf("traces/conv_context_tokens_hist.csv")

# --- Define the search space ---
def create_search_space(gpu_types: list[str], inventory: dict[str, int], min_island_size: int) -> SearchSpace:
    parameters: list[RangeParameter] = []

    for gpu_type in gpu_types:
        total = inventory.get(gpu_type, 0)
        if total <= 0:
            print(f"Skipping BO parameters for {gpu_type} due to zero or negative inventory.")
            continue

        max_islands = total // min_island_size
        upper = max(1, max_islands)
        # num_islands parameter
        parameters.append(
            RangeParameter(
                name=f"num_islands_{gpu_type}",
                parameter_type=ParameterType.INT,
                lower=1,
                upper=upper,
            )
        )
        # skew_exponent parameter
        parameters.append(
            RangeParameter(
                name=f"skew_exponent_{gpu_type}",
                parameter_type=ParameterType.FLOAT,
                lower=-5.0,
                upper=5.0,
            )
        )

    if not parameters:
        raise ValueError("No parameters defined for BO. Check inventory and min_island_size.")

    return SearchSpace(parameters=parameters)

# --- Inner-loop evaluator ---
def eval_config_outer(params_dict):
    print(f"\n\n***** Evaluating configuration with params: {params_dict} *****")
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

        island_sizes_for_type = divider.island_divider(
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
        10.0 # Reject this configuration

    islands_dict = {island.id: island for island in all_generated_islands}
    
    # print(f"Calling packer.solve_linear with {len(islands_dict)} islands...")
    model_solution, prefill_throughput, decode_throughput, delta, objective_val = sys_pack.solve_linear(
        islands_dict, trace_pdf, resolution=100, print_debug=False # Set print_debug as needed
    )

    if prefill_throughput is None or decode_throughput is None or prefill_throughput <= 1e-6 or decode_throughput <= 1e-6:
        print(f"ERROR - Configuration rejected: prefill ({prefill_throughput}) or decode ({decode_throughput}) throughput is None or non-positive.")
        return 10.0 # Reject this configuration

    throughput = min(prefill_throughput, decode_throughput)
    rho_max = 1.0 / throughput
    print(f"***** rho_max = {rho_max:.4f} (Throughput: {throughput:.4f}) *****")
    return rho_max

# --- Create the experiment ---
def create_experiment(search_space) -> Experiment:
    # --- Optimization config ---
    optimization_config = OptimizationConfig(
        objective=Objective(
            metric=GenericNoisyFunctionMetric(
                name="rho_max",
                f=eval_config_outer,        # Use the existing eval function
                noise_sd=1e-4,              # Assuming near‐deterministic
                lower_is_better=True,
            ),
            minimize=True,
        ),
    )

    # --- Build experiment ---
    experiment = Experiment(
        name="gpu_island_scheduler_v2_divider",
        search_space=search_space,
        optimization_config=optimization_config,
        runner=SyntheticRunner(),  # Will call eval_config_outer directly
    )
    return experiment

# --- Initialization ---
def generate_okay_configs(n_configs: int = 1, search_space_obj=None):
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
            print(f"Generated initial configuration {i+1}: {cfg}")
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

# --- Main function ---
def main(batch_size: int = 5, n_iterations: int = 20, path: str = "./data/scratch/gpu_island_scheduler_results.csv"):
    # Create search space and experiment
    search_space = create_search_space(GPU_TYPES, INVENTORY, MIN_ISLAND_SIZE_GLOBAL)
    experiment = create_experiment(search_space)

    # Generate initial configurations
    initial_parameters_list = generate_okay_configs(
        n_configs=5, search_space_obj=experiment.search_space
    )

    if initial_parameters_list:
        add_initial_arms_to_experiment(experiment, initial_parameters_list)
        # Fetch data from the initial trials. This is important for the model.
        data = experiment.fetch_data()
        print("\n--- Initial Warm-Start Data ---")
        print(data.df)
    else:
        print("No initial parameters generated. Starting BO without warm-start.")
        data = Data()  # Start with empty data if no initial trials

    # --- BO loop ---

    # Configure the Standard GP model
    surrogate = Surrogate(
        botorch_model_class=SingleTaskGP,
        model_options={},
        mll_options={}
    )

    print(f"\n--- Starting Standard GP Optimization "
          f"({n_iterations} iterations, {batch_size} batch size) ---")
    for i in range(n_iterations):
        print(f"\n--- Iteration {i + 1}/{n_iterations} ---")
        model = Models.BOTORCH_MODULAR(
            experiment=experiment,
            data=data,
            surrogate=surrogate,
        )

        # Generate next candidates
        generator_run = model.gen(n=batch_size)
        trial = experiment.new_batch_trial(generator_run=generator_run)
        trial.run()

        # Update data
        new_data = trial.fetch_data()
        for _, row in new_data.df.iterrows():
            print(f"Iteration {i+1}, arm {row['arm_name']}: rho_max = {row['mean']}")

        data = Data.from_multiple_data([data, new_data])

        print("\nIntermediate results:")
        print(data.df[["arm_name", "mean"]].sort_values(by="mean"))
        print("\n")

        best = data.df.loc[data.df["mean"].idxmin()]
        print(f"Best rho_max: {best['mean']} for arm {best['arm_name']}")

    # --- Final results ---
    df = exp_to_df(experiment).sort_values(by=["trial_index"])
    best = df.loc[df["rho_max"].idxmin()]
    print("Best parameters:")
    print(best.drop(["trial_index"]))

    # Save intermediate results to CSV
    df.to_csv(path, index=False)

# simple test (batch size 5, 20 iterations, 10 trials)
# if __name__ == "__main__":
#     # make folder if it doesn't exist
#     os.makedirs(os.path.dirname("./data/scratch/5_20_10/"), exist_ok=True)

#     for i in range(10):
#         print(f"\n\n--- Running main iteration {i + 1} ---")
#         main(batch_size=5, n_iterations=20, path=f"./data/scratch/5_20_10/gpu_island_scheduler_results_{i + 1}.csv")

# simple test (batch size 10, 20 iterations, 10 trials)
# if __name__ == "__main__":
#     # make folder if it doesn't exist
#     os.makedirs(os.path.dirname("./data/scratch/10_20_10/"), exist_ok=True)

#     for i in range(10):
#         print(f"\n\n--- Running main iteration {i + 1} ---")
#         main(batch_size=10, n_iterations=20, path=f"./data/scratch/10_20_10/gpu_island_scheduler_results_{i + 1}.csv")