import simulator
import evaluator
import numpy as np
import random
import string
import math
import pulp

import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad

class Packer:
    def __init__(self, gpu_types):
        self.gpu_types = gpu_types
        self.args = simulator.ModelArgs()
        self.prefill_gpu_list = simulator.get_gpu_info('./device/gpu_info.csv', decoding_mode=False, device_list=gpu_types, discount_rate=0.85)
        self.decode_gpu_list = simulator.get_gpu_info('./device/gpu_info.csv', decoding_mode=True, device_list=gpu_types, discount_rate=0.85)

    def solve_prefill(self, islands, trace_pdf, resolution=10, print_debug=True):
        # store probabilities
        ranges = {}
        range_idx = 0
        total_probability = 0

        # check every sequence length, sample at the midpoint of each resolution interval
        for idx, sequence_length in enumerate(trace_pdf[0]):
            midpoint = resolution // 2
            if idx % resolution == midpoint:
                # compute the probability of the bin (sum of all bins in range)
                bin_start = idx - midpoint
                bin_end = bin_start + resolution
                probability = np.sum(trace_pdf[1][bin_start:bin_end])

                # store the probability
                ranges[range_idx] = {
                    'sequence_length': sequence_length,
                    'probability': probability,
                    'min': bin_start,
                    'max': bin_end,
                }

                # increment total probability
                total_probability += probability

                # increment the range index
                range_idx += 1

        # check if the total probability is equal to 1
        if print_debug:
            print(f"Total probability: {total_probability:.4f}")

        # store the results
        prefill_benchmark = {}

        # for each prefill island, compute the time
        for index, (island_id, island) in enumerate(islands.items()):
            if island.role == 'prefill':
                
                # for each range, compute the time
                for range_idx, range in ranges.items():
                    # compute the throughput
                    time_ms = simulator.prefill_time(
                        self.args,
                        self.prefill_gpu_list[island.gpu_type],
                        range['sequence_length'],
                        kv_cache_rate=1,
                        tp_num=island.tp,
                        dp_num=island.dp
                    )

                    time_s = time_ms / 1000
                    request_per_second = 1 / time_s

                    # store the result
                    prefill_benchmark[(island_id, range_idx)] = request_per_second

        # extract Island IDs and sequence lengths
        island_ids = [island_id for island_id, _ in islands.items() if island.role == 'prefill']
        range_ids = [range_idx for range_idx in ranges.keys()]

        # print the results
        if print_debug:
            print("\n=== Ranges ===")
            for range_idx, range in ranges.items():
                print(f"Range {range_idx}: {range['sequence_length']} tokens, Probability: {range['probability']:.4f}, Range: {range['min']}-{range['max']} tokens")

            print("\n=== Prefill Benchmark ===")
            for island_id, range_idx in prefill_benchmark.keys():
                print(f"Island {island_id}, Range {range_idx}: {prefill_benchmark[(island_id, range_idx)]:.4f} requests/s")

        # create the problem
        problem = pulp.LpProblem("Linearized_LoadBalance", pulp.LpMaximize)

        # variables
        x = pulp.LpVariable.dicts("x", (island_ids, range_ids), lowBound=0)
        Rj = pulp.LpVariable.dicts("Rj", range_ids, lowBound=0)
        R = pulp.LpVariable("R", lowBound=0)
        delta = pulp.LpVariable.dicts("delta", range_ids, lowBound=0)
        D = pulp.LpVariable("D", lowBound=0)

        # constraints
        tau = 0.1 # hard‐cap on each deviation
        M = 1000 # base penalty scale
        lambda_dev = {j: M/(1 + ranges[j]['probability']) for j in range_ids}  # soft‐penalty scale

        # each island distributes exactly 1 unit of load
        for island_id in island_ids:
            problem += pulp.lpSum(x[island_id][range_idx] for range_idx in range_ids) == 1, f"LoadBalance_{island_id}"

        # define Rj[j] = sum_i b[i,j] * x[i,j]
        for range_idx in range_ids:
            problem += (
                pulp.lpSum(prefill_benchmark[(island_id, range_idx)] * x[island_id][range_idx] for island_id in island_ids) == Rj[range_idx]
            ), f"Define_Rj_{range_idx}"

        # define R = sum_j Rj[j]
        problem += pulp.lpSum(Rj[range_idx] for range_idx in range_ids) == R, "Define_R"

        # deviation linearization: |Rj - p_j*R| <= delta_j
        for range_idx in range_ids:
            problem += Rj[range_idx] - ranges[range_idx]['probability'] * R <= delta[range_idx], f"DevPos_{range_idx}"
            problem += ranges[range_idx]['probability'] * R - Rj[range_idx] <= delta[range_idx], f"DevNeg_{range_idx}"

        # total deviation D = sum delta_j
        problem += pulp.lpSum(delta[range_idx] for range_idx in range_ids) == D, "Define_D"
        
        # Hard‐cap on each deviation: δ_j ≤ τ·p_j·R
        for range_idx in range_ids:
            problem += delta[range_idx] <= tau * ranges[range_idx]['probability'] * R, f"HardCap_{range_idx}"

        # Objective: maximize R minus weighted sum of deviations
        problem += R - pulp.lpSum(lambda_dev[range_idx] * delta[range_idx] for range_idx in range_ids), "Objective"

        # solve
        problem.solve(pulp.PULP_CBC_CMD(
            msg=True,
            threads=8,
        ))

        if print_debug:
            print("\n=== Allocations ===")
            for island_id in island_ids:
                for range_idx in range_ids:
                    print(f"x[{island_id},{range_idx}] = {x[island_id][range_idx].varValue:.4f}")

            print("\n=== Range perf Rj[j] and deviation |Rj - p_j·R|:")
            for range_idx in range_ids:
                rj = Rj[range_idx].varValue
                dj = delta[range_idx].varValue
                print(f"  j={range_idx}: Rj={rj:.4f}, δ={dj:.4f}  (target {ranges[range_idx]['probability'] * pulp.value(R):.4f})")

        # check if the problem is infeasible
        if pulp.LpStatus[problem.status] == "Infeasible":
            print("Problem is infeasible. Please check the constraints.")
            return None, None, None, None

        # extract the assignment
        assignment = {}
        for island_id in island_ids:
            for range_idx in range_ids:
                if x[island_id][range_idx].varValue > 0:
                    assignment[(island_id, range_idx)] = x[island_id][range_idx].varValue

        # extract the throughput and deviation
        throughput = {}
        deviation = {}
        for range_idx in range_ids:
            throughput[range_idx] = Rj[range_idx].varValue
            deviation[range_idx] = delta[range_idx].varValue

        # construct model
        model = {
            'assignment': assignment,
            'throughput': throughput,
            'deviation': deviation,
            'ranges': ranges,
        }

        # return the assignment
        return model, pulp.value(R), pulp.value(D), pulp.value(problem.objective)
            
def generate_trace_pdf(max_x, mu=100, sigma=75):
    # Define the unnormalized truncated PDF
    def truncated_gaussian(x):
        return norm.pdf(x, loc=mu, scale=sigma) if x >= 0 else 0

    # Compute normalization constant over [0, inf)
    normalization_constant, _ = quad(truncated_gaussian, 0, np.inf)

    # Define the normalized PDF
    def normalized_pdf(x):
        return truncated_gaussian(x) / normalization_constant

    # Generate x values at integer intervals
    x_vals = np.arange(0, max_x + 1, 1)
    # Compute PDF values
    y_vals = np.array([normalized_pdf(xi) for xi in x_vals])

    return x_vals, y_vals

# Example usage
if __name__ == "__main__":
    gpu_types = ["RubinU-NVL576", "H200", "H800"]

    # Create a packer instance
    packer = Packer(gpu_types)

    # Define bins and slots
    islands = [
        evaluator.Island(role="prefill", gpu_type="RubinU-NVL576", dp=1, tp=1, size=1, id="prefill_island_1"),
        evaluator.Island(role="prefill", gpu_type="RubinU-NVL576", dp=2, tp=4, size=1, id="prefill_island_2"),
        evaluator.Island(role="prefill", gpu_type="RubinU-NVL576", dp=4, tp=8, size=1, id="prefill_island_3"),
        evaluator.Island(role="prefill", gpu_type="H200", dp=1, tp=1, size=1, id="prefill_island_4"),
        evaluator.Island(role="prefill", gpu_type="H200", dp=2, tp=4, size=1, id="prefill_island_5"),
        evaluator.Island(role="prefill", gpu_type="H200", dp=4, tp=8, size=1, id="prefill_island_6"),
        evaluator.Island(role="prefill", gpu_type="H800", dp=1, tp=1, size=1, id="prefill_island_7"),
        evaluator.Island(role="prefill", gpu_type="H800", dp=2, tp=4, size=1, id="prefill_island_8"),
        evaluator.Island(role="prefill", gpu_type="H800", dp=4, tp=8, size=1, id="prefill_island_9"),
        evaluator.Island(role="prefill", gpu_type="H800", dp=8, tp=16, size=1, id="prefill_island_10"),
        evaluator.Island(role="prefill", gpu_type="H800", dp=16, tp=32, size=1, id="prefill_island_11"),
        evaluator.Island(role="prefill", gpu_type="H800", dp=32, tp=64, size=1, id="prefill_island_12"),
    ]
    islands = {island.id: island for island in islands}

    # Generate trace PDF
    pdf_x, pdf_y = generate_trace_pdf(max_x=8000, mu=500, sigma=2000)
    trace_pdf = (pdf_x, pdf_y / np.sum(pdf_y))
    print("Trace PDF sum:", np.sum(trace_pdf[1]))

    # Pack the prefill bins
    model, throughput, delta, objective = packer.solve(islands, trace_pdf, resolution=50, print_debug=True)

    if model is not None:
        print("\n=== Results ===")
        print(f"Throughput: {throughput:.4f} requests/s")
        print(f"Deviation: {delta:.4f}")
        print(f"Objective: {objective:.4f}")