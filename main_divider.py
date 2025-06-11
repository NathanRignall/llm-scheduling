# Scheduling of Distributed LLM Serving on Heterogeneous GPUs
# Nathan Rignall
# Main Divider - Used to generate islands and evaluate configurations using Bayesian Optimization (outer loop)

import numpy as np
import os
import pandas as pd

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
GPU_TYPES = ["DGX-B300", "GB300-NVL72", "H200", "H800", "H20"]

# --- Create packer and evaluator instances ---
sys_pack = packer.Packer(gpu_types=GPU_TYPES)

# converation AzureLLM
trace_pdf = evaluator.load_trace_pdf("traces/conv_context_tokens_hist.csv")
decode_len_avg = 106
decode_len_max = 1500

# # code AzureLLM
# trace_pdf = evaluator.load_trace_pdf("traces/code_context_tokens_hist.csv")
# decode_len_avg = 22
# decode_len_max = 5000

# --- Define the search space ---
def create_search_space(gpu_types: list[str], inventory: dict[str, int], min_island_size: int, skew_exponent_range=(-5.0, 5.0)) -> SearchSpace:
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
                lower=skew_exponent_range[0],
                upper=skew_exponent_range[1],
            )
        )

    return SearchSpace(parameters=parameters)

# --- Inner-loop evaluator ---
def eval_config_outer(params_dict, inventory, min_island_size):
    print(f"\n\n***** Evaluating configuration with params: {params_dict}, min_island_size={min_island_size} *****")
    all_generated_islands = []

    for gpu_type in GPU_TYPES:
        num_islands_param_name = f"num_islands_{gpu_type}"
        skew_exponent_param_name = f"skew_exponent_{gpu_type}"

        if num_islands_param_name not in params_dict or skew_exponent_param_name not in params_dict:
            if inventory.get(gpu_type, 0) > 0:
                 print(f"Warning: Parameters for {gpu_type} not in params_dict. Skipping island generation for it.")
            continue

        num_islands_param = params_dict[num_islands_param_name]
        skew_exponent_param = params_dict[skew_exponent_param_name]
        current_inventory_for_type = inventory[gpu_type]

        island_sizes_for_type = divider.island_divider(
            total_gpus_for_type=current_inventory_for_type,
            num_islands_target=num_islands_param,
            size_distribution_skew_exponent=skew_exponent_param,
            min_island_size=min_island_size,
        )
        print(f"  Generated island sizes for {gpu_type}: {island_sizes_for_type} (Sum: {sum(island_sizes_for_type)})")

        for size_val in island_sizes_for_type:
            if size_val > 0:
                island = evaluator.Island(gpu_type=gpu_type, size=size_val)
                all_generated_islands.append(island)

    if not all_generated_islands:
        print("ERROR - No islands were generated. Configuration rejected.")
        return 10.0 # Reject this configuration

    islands_dict = {island.id: island for island in all_generated_islands}
    
    _, prefill_throughput, decode_throughput, _, _ = sys_pack.solve_linear(
        islands_dict, trace_pdf, decode_len_avg, decode_len_max, resolution=100, print_debug=False # Set print_debug as needed
    )

    if prefill_throughput is None or decode_throughput is None or prefill_throughput <= 1e-6 or decode_throughput <= 1e-6:
        print(f"ERROR - Configuration rejected: prefill ({prefill_throughput}) or decode ({decode_throughput}) throughput is None or non-positive.")
        return 10.0 # Reject this configuration

    throughput = min(prefill_throughput, decode_throughput)
    rho_max = 1.0 / throughput
    print(f"***** rho_max = {rho_max:.4f} (Throughput: {throughput:.4f}) *****")
    return rho_max

# --- Create the experiment ---
def create_experiment(search_space, inventory, min_island_size) -> Experiment:
    optimization_config = OptimizationConfig(
        objective=Objective(
            metric=GenericNoisyFunctionMetric(
                name="rho_max",
                f=lambda param_dict: eval_config_outer(param_dict, inventory, min_island_size),
                noise_sd=1e-4,
                lower_is_better=True,
            ),
            minimize=True,
        ),
    )

    experiment = Experiment(
        name="gpu_island_scheduler_divider",
        search_space=search_space,
        optimization_config=optimization_config,
        runner=SyntheticRunner(),
    )
    return experiment

# --- Initialization ---
def generate_okay_configs(n_configs: int = 1, search_space_obj=None):
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
        arm = Arm(name=f"warm_start_{idx}", parameters=params_dict)
        trial = exp_obj.new_trial()
        trial.add_arm(arm)
        trial.run()

# --- Main function ---
def main(inventory: dict[str, int], batch_size: int = 5, n_iterations: int = 20, min_island_size: int = 2, path: str = "./data/scratch/gpu_island_scheduler_results.csv"):
    # new df for storing timestamps
    time_df = pd.DataFrame(columns=['trial index', 'timestamp'])

    # add first timestamp
    time_df.loc[len(time_df)] = [0, pd.Timestamp.now()]

    # create search space and experiment
    search_space = create_search_space(GPU_TYPES, inventory, min_island_size)
    experiment = create_experiment(search_space, inventory, min_island_size)

    # generate initial configurations
    initial_parameters_list = generate_okay_configs(
        n_configs=5, search_space_obj=experiment.search_space
    )

    # add initial arms to the experiment
    if initial_parameters_list:
        add_initial_arms_to_experiment(experiment, initial_parameters_list)
        data = experiment.fetch_data()
        print("\n--- Initial Warm-Start Data ---")
        print(data.df)
    else:
        print("No initial parameters generated. Starting BO without warm-start.")
        data = Data()

    # Configure the Standard GP model
    surrogate = Surrogate(
        botorch_model_class=SingleTaskGP,
        model_options={},
        mll_options={}
    )

    print(f"\n--- Starting Standard GP Optimization ({n_iterations} iterations, {batch_size} batch size) ---")

    for i in range(n_iterations):
        print(f"\n--- Iteration {i + 1}/{n_iterations} ---")
        model = Models.BOTORCH_MODULAR(
            experiment=experiment,
            data=data,
            surrogate=surrogate,
        )

        # generate new arms
        generator_run = model.gen(n=batch_size)
        trial = experiment.new_batch_trial(generator_run=generator_run)
        trial.run()

        # fetch new data
        new_data = trial.fetch_data()
        for _, row in new_data.df.iterrows():
            print(f"Iteration {i+1}, arm {row['arm_name']}: rho_max = {row['mean']}")
        data = Data.from_multiple_data([data, new_data])

        # print intermediate results
        print("\nIntermediate results:")
        print(data.df[["arm_name", "mean"]].sort_values(by="mean"))
        print("\n")

        # print best arm so far
        best = data.df.loc[data.df["mean"].idxmin()]
        print(f"Best rho_max: {best['mean']} for arm {best['arm_name']}")

        # add timestamp for this trial
        time_df.loc[len(time_df)] = [trial.index, pd.Timestamp.now()]

    # --- Final results ---
    df = exp_to_df(experiment).sort_values(by=["trial_index"])
    best = df.loc[df["rho_max"].idxmin()]
    print("Best parameters:")
    print(best.drop(["trial_index"]))

    # Save intermediate results to CSV
    df.to_csv(path, index=False)
    time_df.to_csv(path.replace(".csv", "_timestamps.csv"), index=False)

# main experiments
if __name__ == "__main__":
    #test various batch sizes
    batch_sizes = [2,4,8,16]
    for batch in batch_sizes:
        dir_path = f"./data/scratch/{batch}_20_10"
        os.makedirs(dir_path, exist_ok=True)
        for i in range(10):
            print(f"\n\n--- Running batch size {batch}, trial {i + 1}/10 ---")
            main(
                inventory={"DGX-B300": 128, "H200": 256},
                batch_size=batch,
                n_iterations=20,
                path=f"{dir_path}/trial_{i + 1}.csv",
            )

    # test various skew exponent ranges
    skew_ranges = [(-5.0, 5.0), (-3.0, 3.0), (-1.0, 1.0)]
    for skew_range in skew_ranges:
        dir_path = f"./data/scratch/skew_{skew_range[0]}_{skew_range[1]}"
        os.makedirs(dir_path, exist_ok=True)
        for i in range(10):
            print(f"\n\n--- Running skew range {skew_range}, trial {i + 1}/10 ---")
            main(
                inventory={"DGX-B300": 128, "H200": 256},
                batch_size=8,
                n_iterations=20,
                path=f"{dir_path}/trial_{i + 1}.csv",
            )

    # test various min island sizes
    min_island_sizes = [8, 4, 2]
    for min_island_size in min_island_sizes:
        dir_path = f"./data/scratch/min_island_size_{min_island_size}"
        os.makedirs(dir_path, exist_ok=True)
        for i in range(10):
            print(f"\n\n--- Running min island size {min_island_size}, trial {i + 1}/10 ---")
            main(
                inventory={"DGX-B300": 128, "H200": 256},
                batch_size=8,
                n_iterations=20,
                min_island_size=min_island_size,
                path=f"{dir_path}/trial_{i + 1}.csv",
            )

    inventory_configs = [
        {"H200": 64, "H800": 64, "H20": 128},
        {"H200": 32, "H800": 128, "H20": 128},
        {"H200": 128},
        {"H800": 256}
    ]

    # converation AzureLLM
    trace_pdf = evaluator.load_trace_pdf("traces/conv_context_tokens_hist.csv")
    decode_len_avg = 106
    decode_len_max = 1500

    # azure conversation 
    for inventory in inventory_configs:
        dir_path = f"./data/scratch/het_azure_conv_{'_'.join(f'{k}-{v}' for k, v in inventory.items())}"
        os.makedirs(dir_path, exist_ok=True)
        for i in range(5):
            print(f"\n\n--- Running inventory {inventory}, trial {i + 1}/10 ---")
            main(
                inventory=inventory,
                batch_size=16,
                n_iterations=15,
                path=f"{dir_path}/trial_{i + 1}.csv",
            )

    # # code AzureLLM
    trace_pdf = evaluator.load_trace_pdf("traces/code_context_tokens_hist.csv")
    decode_len_avg = 22
    decode_len_max = 5000

    # azure code 
    for inventory in inventory_configs:
        dir_path = f"./data/scratch/het_azure_code_{'_'.join(f'{k}-{v}' for k, v in inventory.items())}"
        os.makedirs(dir_path, exist_ok=True)
        for i in range(5):
            print(f"\n\n--- Running inventory {inventory}, trial {i + 1}/10 ---")
            main(
                inventory=inventory,
                batch_size=16,
                n_iterations=15,
                path=f"{dir_path}/trial_{i + 1}.csv",
            )

    print("All experiments completed.")