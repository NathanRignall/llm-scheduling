from ax import Data
from ax.core.experiment import Experiment
from ax.core.search_space import SearchSpace
from ax.core.parameter import ParameterType, ChoiceParameter, RangeParameter
from ax.core.metric import Metric
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.metrics.noisy_function import GenericNoisyFunctionMetric
from ax.runners.synthetic import SyntheticRunner
from ax.modelbridge.registry import Models
from ax.modelbridge.cross_validation import cross_validate, compute_diagnostics
from ax.service.utils.report_utils import exp_to_df
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from ax.models.torch.botorch_modular.surrogate import Surrogate

import evaluator

# --- Define search space ---
N_BINS = 2
K_SLOTS = 3

GPU_TYPES = ["H200", "H20"]
INVENTORY = {"H200": 32, "H20": 16}

evaluatorSim = evaluator.Evaluator(
    inventory=INVENTORY,
    gpu_types=GPU_TYPES,
)

parameters = []

# Bin boundaries
for b in range(N_BINS):
    parameters.append(
        RangeParameter(
            name=f"prompt_max_{b}", parameter_type=ParameterType.INT,
            lower=1, upper=2048,
        )
    )
    parameters.append(
        RangeParameter(
            name=f"decode_max_{b}", parameter_type=ParameterType.INT,
            lower=1, upper=2048,
        )
    )

# Island slots
for k in range(K_SLOTS):
    parameters += [
        ChoiceParameter(
            name=f"gpu_type_{k}", parameter_type=ParameterType.STRING,
            values=GPU_TYPES,
        ),
        RangeParameter(
            name=f"size_{k}", parameter_type=ParameterType.INT,
            lower=0, upper=16,
        ),
        RangeParameter(
            name=f"dp_{k}", parameter_type=ParameterType.INT,
            lower=1, upper=16,
        ),
        RangeParameter(
            name=f"tp_{k}", parameter_type=ParameterType.INT,
            lower=1, upper=16,
        ),
        ChoiceParameter(
            name=f"role_{k}", parameter_type=ParameterType.STRING,
            values=["prefill", "decode"],
        ),
    ]

search_space = SearchSpace(parameters=parameters)

# --- Inner-loop evaluator ---
def eval_config_outer(params):

    # Inventory constraint: count GPUs used per type
    usage = {t: 0 for t in GPU_TYPES}
    for k in range(K_SLOTS):
        t = params[f"gpu_type_{k}"]
        s = params[f"size_{k}"]
        # skip unused slots
        if s <= 0:
            continue
        usage[t] += s

    # Reject configurations exceeding stock
    for t, used in usage.items():
        if used > INVENTORY[t]:
            return 1e6
        
    # extract parameters
    bins = []
    for b in range(N_BINS):
        # extract bin parameters
        prompt_max = params[f"prompt_max_{b}"]
        decode_max = params[f"decode_max_{b}"]
        
        # create bin
        bin = evaluator.Bin(
            prompt_max=prompt_max,
            decode_max=decode_max,
        )

        # add bin to the list
        bins.append(bin)

    slots = []
    for k in range(K_SLOTS):
        # extract island parameters
        gpu_type = params[f"gpu_type_{k}"]
        size = params[f"size_{k}"]
        dp = params[f"dp_{k}"]
        tp = params[f"tp_{k}"]
        role = params[f"role_{k}"]

        # skip unused slots
        if size <= 0:
            continue

        # create island
        island = evaluator.Island(
            gpu_type=gpu_type,
            size=size,
            dp=dp,
            tp=tp,
            role=role,
        )

        # add island to the list
        slots.append(island)

    # evaluate configuration
    rho_max = evaluatorSim.evaluate(bins, slots)

    # if evaluation fails, return a large value
    if rho_max is None:
        return 1e6

    return rho_max

# --- Define metric ---
metric = GenericNoisyFunctionMetric(
    name="rho_max",
    f=lambda param_dict: eval_config_outer(param_dict),
    noise_sd=0.0,
    lower_is_better=True,
)

# --- Optimization config ---
optimization_config = OptimizationConfig(
    objective=Objective(metric=metric, minimize=True)
)

# --- Build experiment ---
experiment = Experiment(
    name="gpu_island_scheduler",
    search_space=search_space,
    optimization_config=optimization_config,
    runner=SyntheticRunner(),  # runs metric.f directly
)

# --- Initialization ---
N_INIT = 5 * len(parameters)

def initialize_experiment(exp, n_init):
    sobol = Models.SOBOL(search_space=exp.search_space)
    batch = sobol.gen(n_init)
    trial = exp.new_batch_trial(generator_run=batch)
    trial.run()
    return exp.fetch_data()

data = initialize_experiment(experiment, N_INIT)

# --- SAASBO loop ---
BATCH_SIZE = 4
NUM_SAMPLES = 128
WARMUP_STEPS = 256
N_BATCH = 10

for i in range(N_BATCH):
    # Build SAAS surrogate with NUTS
    model = Models.BOTORCH_MODULAR(
        experiment=experiment,
        data=data,
        surrogate=Surrogate(
            botorch_model_class=SaasFullyBayesianSingleTaskGP,
            mll_options={
                "num_samples": NUM_SAMPLES,
                "warmup_steps": WARMUP_STEPS,
            },
        ),
    )

    # Generate next candidates with SAASBO
    generator_run = model.gen(BATCH_SIZE)
    trial = experiment.new_batch_trial(generator_run=generator_run)
    trial.run()

    # Update data
    new_data = trial.fetch_data()
    data = Data.from_multiple_data([data, new_data])

    # Diagnostics
    exp_df = exp_to_df(experiment)
    print(f"Iteration {i+1}/{N_BATCH} completed; total trials: {len(exp_df['trial_index'].unique())}")

# --- Final results ---
df = exp_to_df(experiment).sort_values(by=["trial_index"])
best = df.loc[df["rho_max"].idxmin()]
print("Best parameters:")
print(best.drop(["trial_index"]))