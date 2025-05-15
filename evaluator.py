import simulator
import numpy as np
import random
import string
import math

import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad

import simulator

def generate_random_string(length):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

class Island:
    def __init__(self, role, gpu_type, dp, tp, size, id=None):
        self.id = id if id is not None else generate_random_string(8)
        self.role = role
        self.gpu_type = gpu_type
        self.dp = dp
        self.tp = tp
        self.size = size
class Bin:
    def __init__(self, role, min, max, id=None):
        self.id = id if id is not None else generate_random_string(8)
        self.role = role
        self.min = min
        self.max = max
class Evaluator:
    def __init__(self, gpu_types):
        self.gpu_types = gpu_types
        self.args = simulator.ModelArgs()
        self.gpu_list = simulator.get_gpu_info('./device/gpu_info.csv', decoding_mode=False, device_list=gpu_types, discount_rate=0.85)

    def evaluate(self, config, islands, bins, trace_pdf, ratio=1.0, resolution=10):
        # for each element in trace_pdf, find the corresponding bin
        # for each bin, find the corresponding island and calculate the throughput
        # use resolution to determine the interval to sample the trace_pdf

        results = []

        for idx, (sequence_length, probability) in enumerate(zip(trace_pdf[0], trace_pdf[1])):
            # only sample the trace_pdf at the resolution intervals
            if (idx + 1) % resolution == 0:
                # find the prefill bin
                prefill_bin_id = None
                for index, (id, bin) in enumerate(bins.items()):
                    if bin.role == "prefill" and bin.min <= sequence_length < bin.max:
                        prefill_bin_id = id
                        break

                # find the decode bin
                decode_bin_id = None
                for index, (id, bin) in enumerate(bins.items()):
                    if bin.role == "decode" and bin.min <= sequence_length < bin.max:
                        decode_bin_id = id
                        break
                        
                # check if bins were found
                if prefill_bin_id is None and decode_bin_id is  None:
                    raise ValueError(f"No bin found for sequence length {sequence_length}")
                
                # find the island id
                prefill_island_id = config[prefill_bin_id]
                decode_island_id = config[decode_bin_id]

                # check if islands were found
                if prefill_island_id is None and decode_island_id is  None:
                    raise ValueError(f"No island found for bin {prefill_bin_id} or {decode_bin_id}")
                
                # calculate the throughput
                prefill_island = islands[prefill_island_id]
                decode_island = islands[decode_island_id]

                prefill_throughput = self.prefill_throughput(prefill_island, sequence_length)
                decode_throughput = self.decode_throughput(decode_island, sequence_length, sequence_length)

                overall_throughput = self.calculate_throughput(prefill_throughput, decode_throughput, ratio)
                adjusted_throughput = overall_throughput * probability * resolution

                # store the results
                results.append({
                    "sequence_length": sequence_length,
                    "prefill_throughput": prefill_throughput,
                    "decode_throughput": decode_throughput,
                    "overall_throughput": overall_throughput,
                    "adjusted_throughput": adjusted_throughput,
                    "probability": probability,
                })

        # calculate average adjusted throughput
        total_adjusted_throughput = sum([result["adjusted_throughput"] for result in results])
        average_adjusted_throughput = total_adjusted_throughput / len(results) if results else 0

        return results, average_adjusted_throughput

    def calculate_throughput(self, prefill_throughput, decode_throughput, ratio):
        throughput = (ratio + 1) / (ratio / decode_throughput + 1 / prefill_throughput)
        return throughput

    def prefill_throughput(self, island, sequence_length):
        time = simulator.prefill_time(
            self.args,
            self.gpu_list[island.gpu_type],
            sequence_length,
            kv_cache_rate=0.5,
            tp_num=island.tp,
            dp_num=island.dp
        )

        return sequence_length / time
    
    def decode_throughput(self, island, sequence_length, decode_length):
        throughput = simulator.decode_time(
            self.args,
            self.gpu_list[island.gpu_type],
            dp_num=island.dp,
            tp_num=island.tp,
            bs_num=16,
            seq_len=sequence_length,
            decode_len=decode_length,
            gemm_group_per_device=math.ceil(self.args.n_routed_experts / island.size),
            device_num=island.size,
        )

        return throughput
            
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

    # create evaluator
    gpu_types=["DGX-B300", "H20"]
    evaluator = Evaluator(gpu_types)
    
    # Create islands
    islands = [
        Island(role="prefill", gpu_type="DGX-B300", dp=1, tp=1, size=1, id="prefill_island_1"),
        Island(role="decode", gpu_type="DGX-B300", dp=4, tp=1, size=32, id="decode_island_1")
    ]
    islands = {island.id: island for island in islands}

    # Create bins
    bins = [
        Bin(role="prefill", min=0, max=65537, id="prefill_bin_1"),
        Bin(role="decode", min=0, max=65537, id="decode_bin_1")
    ]
    bins = {bin.id: bin for bin in bins}

    # create config
    config = {
        "prefill_bin_1": "prefill_island_1",
        "decode_bin_1": "decode_island_1"
    }
    
    # Generate trace PDF
    pdf_x, pdf_y = generate_trace_pdf(max_x=65536, mu=8192, sigma=8192)
    trace_pdf = (pdf_x, pdf_y)

    # plot pdf
    plt.plot(pdf_x, pdf_y, label="Truncated Gaussian PDF")
    plt.title("Truncated Gaussian PDF")
    plt.xlabel("Sequence Length")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid()
    plt.show()

    # Evaluate
    ratio = 1.0
    resolution = 1
    results, throughput = evaluator.evaluate(config, islands, bins, trace_pdf, ratio=ratio, resolution=1000)

    # Plot all throughputs on one graph
    sequence_lengths = [r["sequence_length"] for r in results]
    prefill_throughputs = [r["prefill_throughput"] for r in results]
    decode_throughputs = [r["decode_throughput"] for r in results]
    overall_throughputs = [r["overall_throughput"] for r in results]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sequence_lengths, prefill_throughputs, label="Prefill Throughput", color="tab:blue")
    ax.plot(sequence_lengths, decode_throughputs, label="Decode Throughput", color="tab:orange")
    ax.plot(sequence_lengths, overall_throughputs, label="Overall Throughput", color="tab:green")
    ax.set_title("Throughput vs Sequence Length")
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Tokens/s")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Plot adjusted throughput
    adjusted_throughputs = [r["adjusted_throughput"] for r in results]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sequence_lengths, adjusted_throughputs, label="Adjusted Throughput", color="tab:red")
    ax.axhline(y=throughput, color='r', linestyle='--', label="Average Adjusted Throughput")
    ax.set_title("Adjusted Throughput vs Sequence Length")
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Adjusted Tokens/s")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()