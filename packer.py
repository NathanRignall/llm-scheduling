import simulator
import itertools
import numpy as np

def generate_trace(num_samples):
    np.random.seed(42)

    # Generate prompt and decode lengths
    prompt_lengths = np.random.normal(loc=2048, scale=1024, size=num_samples)
    decode_lengths = np.random.normal(loc=512, scale=128, size=num_samples)

    # Convert to whole numbers and drop negative values
    prompt_lengths = np.clip(prompt_lengths, 0, None).astype(int)
    decode_lengths = np.clip(decode_lengths, 0, None).astype(int)

    # Create tuples
    test_trace = list(zip(prompt_lengths, decode_lengths))

    return test_trace

class Island:
    def __init__(self, role, gpu_type, dp = 1, tp = 1, size = 0):
        self.gpu_type = gpu_type
        self.role = role
        self.dp = dp
        self.tp = tp
        self.size = size

        if self.role == 'prefill':
            self.size = self.dp * self.tp
            
class Bin:
    def __init__(self, role, prompt_max, decode_max):
        self.role = role
        self.prompt_max = prompt_max
        self.decode_max = decode_max

class Packer:
    def __init__(self, inventory, gpu_types):
        self.inventory = inventory
        self.gpu_types = gpu_types
        self.args = simulator.ModelArgs()
        self.gpu_list = simulator.get_gpu_info('./device/gpu_info.csv', decoding_mode=True, device_list=gpu_types, discount_rate=0.85)

    def pack_prefill(self, bins, islands, num_samples, print_debug=False):
        # Generate a test trace
        test_trace = generate_trace(num_samples)

        # extract the prefill bins and islands
        prefill_bins = [bin for bin in bins if bin.role == 'prefill']
        prefill_islands = [island for island in islands if island.role == 'prefill']

        # Debug print start
        if print_debug:
            print(f"Packing {len(prefill_bins)} prefill bins onto {len(prefill_islands)} prefill islands...")

        # sort bins by their prompt_max so we can use previous/current as bounds
        sorted_bins = sorted(prefill_bins, key=lambda b: b.prompt_max)
        n = len(sorted_bins)

        # filtered traces
        filtered_traces = []

        for idx, bin in enumerate(sorted_bins):
            # calculate the bounds for the current bin
            prev_prefill = sorted_bins[idx - 1].prompt_max if idx > 0 else 0
            upper_prefill = bin.prompt_max if idx < n - 1 else float('inf')

            # filter the test trace to only include samples within the bounds
            filtered_trace = [
                (prompt_length, decode_length)
                for prompt_length, decode_length in test_trace
                if prev_prefill < prompt_length <= upper_prefill
            ]
            filtered_traces.append(filtered_trace)

        # Store the results of every test
        results = []

        for filtered_trace, bin in zip(filtered_traces, sorted_bins):
            if len(filtered_trace) == 0:
                # No samples in this bin range, skip evaluation
                print(f"Bin {bin.prompt_max} has no samples in the trace.")

            # evaluate on the filtered trace
            for island in prefill_islands:
                avg_time = self.evaluate_prefill(island, filtered_trace)
                results.append((bin, island, avg_time, filtered_trace))

        # debug‐print the results
        if print_debug:
            for bin, island, t, trace in results:
                print(
                    f"Bin(pmax={bin.prompt_max}, dmax={bin.decode_max}) "
                    f"on Isl(gpu={island.gpu_type}, size={island.size}): {t:.2f} "
                    f"for {len(trace)} samples"
                )

        # # scale results to the number of samples (divide by the number of samples)
        # for i in range(len(results)):
        #     bin, island, t, trace = results[i]
        #     avg_time = t / len(trace) if len(trace) > 0 else 0
        #     results[i] = (bin, island, avg_time, trace)

        # generate all possible configurations to pack the bins
        configs = []

        for perm in itertools.permutations(prefill_islands, r=len(prefill_bins)):
            mapping = dict(zip(bins, perm))
            configs.append(mapping)

        if len(configs) == 0:
            print("ERROR - No valid configurations found.")
            return None, None, filtered_traces, results

        # Print the configurations for debugging
        if print_debug:
            for config in configs:
                print("Configuration:")
                for bin, island in config.items():
                    print(f"  Island (gpu_type={island.gpu_type}, size={island.size}) "
                        f"-> Bin (prompt_max={bin.prompt_max}, decode_max={bin.decode_max})")
   
            print(f"\nEvaluating {len(configs)} configurations...")
                
        # Evaluate each configuration
        best_time = float('inf')
        best_config = None

        for config in configs:
            time = 0

            # use the result 
            for bin, island in config.items():
                # look up the result
                for b, i, result, t in results:
                    if b == bin and i == island:
                        # if bigger than time
                        if result > time:
                            time = result
                        break
                else:
                    print(f"Configuration not found for bin {bin} and island {island}")
                    time = float('inf') 
                    # continue
                    # throw an error

            if time < best_time and time > 0:
                best_time = time
                best_config = config

            if print_debug:
                print(f"Configuration time: {time:.2f}")
                print("Configuration:")
                for bin, island in config.items():
                    print(f"  Island (gpu_type={island.gpu_type}, size={island.size}) "
                        f"-> Bin (prompt_max={bin.prompt_max}, decode_max={bin.decode_max})")

        if print_debug:
            print(f"\nBest configuration speed: {best_time:.2f}")
            print("Best configuration:")
            for bin, island in best_config.items():
                print(f"  Island (gpu_type={island.gpu_type}, size={island.size}) "
                    f"-> Bin (prompt_max={bin.prompt_max}, decode_max={bin.decode_max})")
                
        if best_time == float('inf'):
            print("No valid configuration found.")
        
        return best_time, best_config, filtered_traces, results

    def evaluate_prefill(self, island, test_trace):
        # count the frequency of each prompt length
        prompt_lengths = [prompt_length for prompt_length, _ in test_trace]
        prompt_length_counts = {length: prompt_lengths.count(length) for length in set(prompt_lengths)}

        # Simulate the prefill time for each trace and calculate the average
        total_time = 0
        for prompt_length, _ in test_trace:
            time = simulator.prefill_time(
                self.args,
                self.gpu_list[island.gpu_type],
                prompt_length,
                kv_cache_rate=1,
                tp_num=island.tp,
                dp_num=island.dp
            )

            # scale the time by the frequency of the prompt length
            frequency = prompt_length_counts[prompt_length]
            total_time += time * frequency

        # Return the average prefill time
        average_time = total_time 
        return average_time

# Example usage
if __name__ == "__main__":
    # Define the inventory and GPU types
    inventory = {
        'H200': 32,
        'DGX-B300': 16,
    }
    gpu_types = ['H200', 'DGX-B300']

    # Create a packer instance
    packer = Packer(inventory, gpu_types)

    # Define bins and slots
    bins = [Bin('prefill', 6096, 512)]
    slots = [Island('prefill', 'DGX-B300', dp=1, tp=4), Island('prefill', 'DGX-B300', tp=2, dp=2)]

    # Pack the prefill bins
    prefill_speed, prefill_config, filtered_traces, results = packer.pack_prefill(bins, slots, 10000, print_debug=True)

    print(f"Prefill speed: {prefill_speed:.2f}")
    print("Prefill configuration:")
    for bin, island in prefill_config.items():
        print(f"  Island (gpu_type={island.gpu_type}, size={island.size}) "
            f"-> Bin (prompt_max={bin.prompt_max}, decode_max={bin.decode_max})")
        

    # plot the filtered traces as a histogram
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    for i, filtered_trace in enumerate(filtered_traces):
        if not filtered_trace:
            continue
        # Extract prompt lengths
        prompt_lengths = [p for p, _ in filtered_trace]
        # Compute bin edges with width 32
        min_len, max_len = min(prompt_lengths), max(prompt_lengths)
        bins = np.arange(min_len, max_len + 32, 32)
        plt.hist(prompt_lengths, bins=bins, alpha=0.5, label=f'Bin {i+1}')
    plt.xlabel('Prompt Length')
    plt.ylabel('Frequency')
    plt.title('Filtered Traces Histogram (bin width = 10)')
    plt.legend()
    plt.show()