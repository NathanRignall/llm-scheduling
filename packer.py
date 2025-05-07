import simulator
import itertools
import numpy as np

def generate_trace(num_samples):
    np.random.seed(42)

    # Generate prompt and decode lengths
    prompt_lengths = np.random.normal(loc=2048, scale=1024, size=num_samples)
    decode_lengths = np.random.normal(loc=512, scale=128, size=num_samples)

    # Convert to whole numbers and clip to minimum of 1
    prompt_lengths = np.clip(prompt_lengths.astype(int), 1, None)
    decode_lengths = np.clip(decode_lengths.astype(int), 1, None)

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

    def pack_prefill(self, bins, islands, print_debug=False):
        # Generate a test trace
        test_trace = generate_trace(num_samples=1000)

        # extract the prefill bins and islands
        prefill_bins = [bin for bin in bins if bin.role == 'prefill']
        prefill_islands = [island for island in islands if island.role == 'prefill']

        if print_debug:
            print(f"Packing {len(prefill_bins)} prefill bins onto {len(prefill_islands)} prefill islands...")

        # sort bins by their prompt_max so we can use previous/current as bounds
        sorted_bins = sorted(prefill_bins, key=lambda b: b.prompt_max)
        n = len(sorted_bins)

        # filtered traces
        filtered_traces = []

        for idx, bin in enumerate(sorted_bins):
            # lower bounds
            prev_prefill = sorted_bins[idx - 1].prompt_max if idx > 0 else 0

            # upper bounds
            upper_prefill = bin.prompt_max if idx < n - 1 else float('inf')

            # filter the test trace to only include samples within the bounds
            filtered_trace = [
                (prompt_length, decode_length)
                for prompt_length, decode_length in test_trace
                if prev_prefill < prompt_length <= upper_prefill
            ]
            print(f"Bin {idx + 1}: {bin.prompt_max} ({prev_prefill}, {upper_prefill})")
            for item in filtered_trace:
                sequence_length = item[0]
                print(f"Filtered trace: {sequence_length}")

            filtered_traces.append(filtered_trace)

        #plot the filtered traces as a histogram, different colors each element
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(10, 6))
        # for i, filtered_trace in enumerate(filtered_traces):
        #     if len(filtered_trace) == 0:
        #         # No samples in this bin range, skip plotting
        #         continue
        #     # Extract prompt lengths for the histogram
        #     prompt_lengths = [prompt_length for prompt_length, _ in filtered_trace]
        #     # Plot the histogram
        #     plt.hist(prompt_lengths, bins=20, alpha=0.5, label=f'Bin {i+1}')
        # plt.xlabel('Prompt Length')
        # plt.ylabel('Frequency')
        # plt.title('Filtered Traces Histogram')
        # plt.legend()
        # plt.show()

        # Store the results of every test
        results = []

        for filtered_trace, bin in zip(filtered_traces, sorted_bins):
            if len(filtered_trace) == 0:
                # No samples in this bin range, skip evaluation
                continue

            # evaluate on the filtered trace
            for island in prefill_islands:
                avg_time = self.evaluate_prefill(bin, island, filtered_trace)
                results.append((bin, island, avg_time, filtered_trace))

        # debug‐print the results
        if print_debug:
            for bin, island, t, trace in results:
                print(
                    f"Bin(pmax={bin.prompt_max}, dmax={bin.decode_max}) "
                    f"on Isl(gpu={island.gpu_type}, size={island.size}): {t:.2f} "
                    f"for {len(trace)} samples"
                )
            
        # Create a mapping of bins to islands
        configs = []
        for perm in itertools.permutations(prefill_bins, len(prefill_islands)):
            mapping = dict(zip(prefill_islands, perm))
            configs.append(mapping) 

        # Print the configurations for debugging
        if print_debug:
            for config in configs:
                print("Configuration:")
                for island, bin in config.items():
                    print(f"  Island (gpu_type={island.gpu_type}, size={island.size}) "
                        f"-> Bin (prompt_max={bin.prompt_max}, decode_max={bin.decode_max})")
   
            print(f"\nEvaluating {len(configs)} configurations...")
                
        # Evaluate each configuration
        best_time = float('inf')
        best_config = None

        for config in configs:
            time = 0

            # use the result 
            for island, bin in config.items():
                # look up the result
                for b, i, result, t in results:
                    if b == bin and i == island:
                        # if bigger than time
                        if result > time:
                            time = result
                        break
                else:
                    print(f"Configuration not found for bin {bin} and island {island}")
                    continue

            if time < best_time and time > 0:
                best_time = time
                best_config = config

            if print_debug:
                print(f"Configuration time: {time:.2f}")
                print("Configuration:")
                for island, bin in config.items():
                    print(f"  Island (gpu_type={island.gpu_type}, size={island.size}) "
                        f"-> Bin (prompt_max={bin.prompt_max}, decode_max={bin.decode_max})")

        if print_debug:
            print(f"\nBest configuration speed: {best_time:.2f}")
            print("Best configuration:")
            for island, bin in best_config.items():
                print(f"  Island (gpu_type={island.gpu_type}, size={island.size}) "
                    f"-> Bin (prompt_max={bin.prompt_max}, decode_max={bin.decode_max})")
        
        return best_time, best_config

    def evaluate_prefill(self, bin, island, test_trace):
        # Simulate the prefill time for each trace and calculate the average
        total_time = 0
        for prompt_length, _ in test_trace:
            total_time += simulator.prefill_time(
                self.args,
                self.gpu_list[island.gpu_type],
                prompt_length,
                kv_cache_rate=1,
                tp_num=island.tp,
                dp_num=island.dp
            )

        # Return the average prefill time
        average_time = total_time 
        return average_time

# Example usage
if __name__ == "__main__":
    # Define the inventory and GPU types
    inventory = {
        'H200': 32,
        'H800': 16,
    }
    gpu_types = ['H200', 'H800']

    # Create a packer instance
    packer = Packer(inventory, gpu_types)

    # Define bins and slots
    bins = [Bin('prefill', 1024, 0), Bin('prefill', 2048, 0)]
    slots = [Island('prefill', 'H200', 8, 1), Island('prefill', 'H200', 8, 1)]

    # Pack the prefill bins
    prefill_speed, prefill_config = packer.pack_prefill(bins, slots, print_debug=True)

    print(f"Prefill speed: {prefill_speed:.2f}")
    print("Prefill configuration:")
    for island, bin in prefill_config.items():
        print(f"  Island (gpu_type={island.gpu_type}, size={island.size}) "
            f"-> Bin (prompt_max={bin.prompt_max}, decode_max={bin.decode_max})")