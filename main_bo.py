import numpy as np
import os
import pandas as pd

from ax import Arm, Data, SumConstraint
from ax.core.experiment import Experiment
from ax.core.search_space import SearchSpace
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.metrics.noisy_function import GenericNoisyFunctionMetric
from ax.runners.synthetic import SyntheticRunner
from ax.modelbridge.registry import Models
from ax.service.utils.report_utils import exp_to_df
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from ax.models.torch.botorch_modular.surrogate import SurrogateSpec, ModelConfig
from botorch.models import SingleTaskGP
from ax.models.torch.botorch_modular.surrogate import Surrogate

import packer
import evaluator

# --- Define search space ---
GPU_TYPES = ["DGX-B300", "H200"]
INVENTORY = {"DGX-B300": 128, "H200": 256}

# --- Create packer and evaluator instances ---
sys_packer = packer.Packer(GPU_TYPES)

# converation AzureLLM
trace_pdf = evaluator.load_trace_pdf("traces/conv_context_tokens_hist.csv")
decode_len_avg = 106
decode_len_max = 1500

# code AzureLLM
trace_pdf = evaluator.load_trace_pdf("traces/code_context_tokens_hist.csv")
decode_len_avg = 22
decode_len_max = 5000

# --- Define the search space ---
def create_search_space(gpu_types: list[str], inventory: dict[str, int], k_slots: int) -> SearchSpace:
    parameters = []
    parameter_constraints = []

    for gpu_type in gpu_types:
        # GPU Sum
        gpu_sum = []

        # For each GPU type, create k slots of parameters
        for k in range(k_slots):
            # GPU Count
            gpu_count = RangeParameter(
                name=f"count_{gpu_type}_{k}", parameter_type=ParameterType.INT,
                lower=0, upper=inventory[gpu_type],
            )
            
            # Add parameters to search space
            parameters.append(gpu_count)

            # Add to sum
            gpu_sum.append(gpu_count)

        # Constrain GPU count per type to inventory
        parameter_constraints += [
            SumConstraint(
                parameters=gpu_sum,
                is_upper_bound=True,
                bound=inventory[gpu_type],
            ),
        ]

    return SearchSpace(parameters=parameters, parameter_constraints=parameter_constraints)

# --- Inner-loop evaluator ---
def eval_config_outer(params, k_slots):

    print("\n\n***** Evaluating configuration *****")

    # Inventory constraint: count GPUs used per type
    for gpu_type in GPU_TYPES:
        total_count = 0
        for k in range(k_slots):
            total_count += params[f"count_{gpu_type}_{k}"]
        percent_used = 100 * total_count / INVENTORY[gpu_type]
        print(f"GPU usage: {gpu_type} {percent_used:.2f}%")
        if total_count > INVENTORY[gpu_type]:
            print(f"ERROR - Configuration rejected: {gpu_type} used {total_count} > {INVENTORY[gpu_type]} available")
            return 10.0

    islands = []
    for gpu_type in GPU_TYPES:
        for k in range(k_slots):
            # Extract island parameters
            count = params[f"count_{gpu_type}_{k}"]

            # Skip unused slots
            if count <= 0:
                continue

            # Create island
            island = evaluator.Island(
                gpu_type=gpu_type,
                size=count,
            )

            # Add island to the list
            islands.append(island)

    # turn into dict
    islands = {island.id: island for island in islands}

    # Evaluate configuration
    _, prefill_throughput, decode_throughput, _, _ =  sys_packer.solve_linear(islands, trace_pdf, decode_len_avg, decode_len_max, resolution=100, print_debug=False)

    # pick lowest throughput if neither is None
    if prefill_throughput is None or decode_throughput is None:
        print("ERROR - Configuration rejected: prefill or decode throughput is None")
        return 10.0

    # calculate rho_max
    throughput = min(prefill_throughput, decode_throughput)
    rho_max = 1 / throughput

    print(f"***** rho_max = {rho_max:.2f} *****")

    return rho_max

# --- Create the experiment ---
def create_experiment(search_space, k_slots) -> Experiment:
    # --- Optimization config ---
    optimization_config = OptimizationConfig(
        objective=Objective(
            metric=GenericNoisyFunctionMetric(
                    name="rho_max",
                    f=lambda param_dict: eval_config_outer(param_dict, k_slots),
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
    return experiment

# --- Initialization ---
def generate_okay_configs(n_configs: int, k_slots: int) -> list[dict]:
    configs = []
    for _ in range(n_configs):
        cfg = {}
        # For each GPU type, track how many we've allocated so we don't exceed inventory
        for gpu_type in GPU_TYPES:
            total_used = 0
            for k in range(k_slots):
                # Remaining budget for this gpu_type
                remaining = INVENTORY[gpu_type] - total_used

                count = np.random.randint(0, remaining + 1)
                cfg[f"count_{gpu_type}_{k}"] = count

                total_used += count

        print(f"Initial configuration: {cfg}")
        configs.append(cfg)
    return configs

def add_initial_arms(experiment, initial_parameters):
    for idx, params in enumerate(initial_parameters):
        arm = Arm(name=f"warm_start_{idx}", parameters=params)
        trial = experiment.new_trial()
        trial.add_arm(arm)
        trial.run()

# --- Main function ---
def main(batch_size: int = 5, n_iterations: int = 20, path: str = "./data/scratch/gpu_island_scheduler_results.csv", k_slots: int = 16):
    # new df for storing timestamps
    time_df = pd.DataFrame(columns=['trial index', 'timestamp'])

    # add first timestamp
    time_df.loc[len(time_df)] = [0, pd.Timestamp.now()]

    # Create search space and experiment
    search_space = create_search_space(GPU_TYPES, INVENTORY, k_slots)
    experiment = create_experiment(search_space, k_slots)

    # Generate initial configurations
    initial_parameters_list = generate_okay_configs(
        n_configs=5,
        k_slots=k_slots
    )

    # Add initial arms to the experiment
    if initial_parameters_list:
        add_initial_arms(experiment, initial_parameters_list)
        data = experiment.fetch_data()
        print("\n--- Initial Warm-Start Data ---")
        print(data.df)
    else:
        print("No initial parameters generated. Starting BO without warm-start.")
        data = Data()

    # # --- SAASBO loop ---
    # NUM_SAMPLES = 128
    # WARMUP_STEPS = 256

    # model_config = ModelConfig(
    #     botorch_model_class=SaasFullyBayesianSingleTaskGP,
    #     model_options={},  
    #     mll_options={
    #         "num_samples": NUM_SAMPLES,
    #         "warmup_steps": WARMUP_STEPS,
    #     },
    # )

    # surrogate = SurrogateSpec(
    #     model_configs=[model_config],
    # )

    # --- BO loop ---

    # Configure the Standard GP model
    surrogate = Surrogate(
        botorch_model_class=SingleTaskGP,
        model_options={},
        mll_options={}
    )

    print(f"\n--- Starting Optimization ({n_iterations} iterations, {batch_size} batch size) ---")

    for i in range(n_iterations):
        print(f"\n--- Iteration {i + 1}/{n_iterations} ---")
        model = Models.BOTORCH_MODULAR(
            experiment=experiment,
            data=data,
            surrogate=surrogate,
        )

        # Generate next candidates with SAASBO
        generator_run = model.gen(n=batch_size)
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

        # Add timestamp for this trial
        time_df.loc[len(time_df)] = [trial.index, pd.Timestamp.now()]

    # --- Final results ---
    df = exp_to_df(experiment).sort_values(by=["trial_index"])
    best = df.loc[df["rho_max"].idxmin()]
    print("Best parameters:")
    print(best.drop(["trial_index"]))

    # Save intermediate results to CSV
    df.to_csv(path, index=False)
    time_df.to_csv(path.replace(".csv", "_timestamps.csv"), index=False)

# uncomment to execute experiment
if __name__ == "__main__":
    # # Test various batch sizes
    # batch_sizes = [2,4,8,16]
    # for batch in batch_sizes:
    #     dir_path = f"./data/scratch/xxxbo_{batch}_10_5"
    #     os.makedirs(dir_path, exist_ok=True)
    #     for i in range(5):
    #         print(f"\n\n--- Running batch size {batch}, trial {i + 1}/5 ---")
    #         main(
    #             batch_size=batch,
    #             n_iterations=10,
    #             path=f"{dir_path}/trial_{i + 1}.csv",
    #             k_slots=16,
    #         )

    # # test various k_slots
    # k_slots_list = [16]
    # for k_slots in k_slots_list:
    #     dir_path = f"./data/scratch/saasbo_k_slots_{k_slots}"
    #     os.makedirs(dir_path, exist_ok=True)
    #     for i in range(4,5):
    #         print(f"\n\n--- Running k_slots {k_slots}, trial {i + 1}/5 ---")
    #         main(
    #             batch_size=4,
    #             n_iterations=10,
    #             path=f"{dir_path}/trial_{i + 1}.csv",
    #             k_slots=k_slots,
    #         )

    print("All experiments completed.")