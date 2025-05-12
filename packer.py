import simulator
import itertools
import numpy as np

from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, PULP_CBC_CMD

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
    def __init__(self, id, role, gpu_type, dp = 1, tp = 1, size = 0):
        self.id = id
        self.gpu_type = gpu_type
        self.role = role
        self.dp = dp
        self.tp = tp
        self.size = size

        if self.role == 'prefill':
            self.size = self.dp * self.tp
            
class Bin:
    def __init__(self, role, prompt_range):
        self.role = role
        self.prompt_range = prompt_range

class Packer:
    def __init__(self, inventory, gpu_types):
        self.inventory = inventory
        self.gpu_types = gpu_types
        self.args = simulator.ModelArgs()
        self.gpu_list = simulator.get_gpu_info('./device/gpu_info.csv', decoding_mode=True, device_list=gpu_types, discount_rate=0.85)

    # test each island with range of prompt lengths to find best bins
    def solve_prefill(self, islands, print_debug=True):
        # create a list of prompt lengths evenly spaced
        min_prompt_lengths = np.arange(0, 1024, 64).astype(int)
        max_prompt_lengths = min_prompt_lengths + 64
        midpoint_prompt_lengths = (min_prompt_lengths + max_prompt_lengths) // 2

        # # print the prompt lengths
        # if print_debug:
        #     for i in range(len(min_prompt_lengths)):
        #         print(f"Prompt length: {min_prompt_lengths[i]} - {midpoint_prompt_lengths[i]} - {max_prompt_lengths[i]}")

        # extract the prefill islands
        prefill_islands = [island for island in islands if island.role == 'prefill']

        # measure the time for each prompt length
        results = []

        for prompt_length in midpoint_prompt_lengths:
            local_results = {}

            for island in prefill_islands:
                # measure the time for each prompt length
                time = simulator.prefill_time(
                    self.args,
                    self.gpu_list[island.gpu_type],
                    prompt_length,
                    kv_cache_rate=1,
                    tp_num=island.tp,
                    dp_num=island.dp
                )

                # store the result for each island
                local_results[island.id] = time

            # store the result for each prompt length
            results.append(local_results)

        # print the results
        if print_debug:
            print("\n=== Prefill Time Results ===")
            for i in range(len(min_prompt_lengths)):
                print(f"Prompt length: {min_prompt_lengths[i]} - {midpoint_prompt_lengths[i]} - {max_prompt_lengths[i]}")
                for island in prefill_islands:
                    print(f"  Island (id={island.id}, gpu_type={island.gpu_type}, size={island.size}): {results[i][island.id]:.2f}")

        # create the problem
        problem = LpProblem("Prefill_Packing_Problem", LpMinimize)

        # decision vars
        x = {
            (isl.id, p): LpVariable(f"x_island{isl.id}_mid{p}", cat=LpBinary)
            for isl in prefill_islands
            for p in range(len(midpoint_prompt_lengths))
        }
        # makespan variable
        M = LpVariable("Makespan", lowBound=0)

        # objective: minimize the makespan
        problem += M

        # each prompt range p goes to exactly one island
        for p in range(len(midpoint_prompt_lengths)):
            problem += lpSum(x[(isl.id, p)] for isl in prefill_islands) == 1

        # for each island, its total assigned time ≤ M
        for isl in prefill_islands:
            problem += lpSum(
                results[p][isl.id] * x[(isl.id, p)]
                for p in range(len(midpoint_prompt_lengths))
            ) <= M

        # solve
        problem.solve(PULP_CBC_CMD(msg=False))

        # extract
        assignment = {}
        for p in range(len(midpoint_prompt_lengths)):
            for isl in prefill_islands:
                if x[(isl.id, p)].value() == 1:
                    assignment[p] = isl
                    break

        # compute the makespan explicitly
        island_loads = {
            isl.id: sum(results[p][isl.id] for p, a in assignment.items() if a.id == isl.id)
            for isl in prefill_islands
        }
        makespan = max(island_loads.values())

        # debug output
        if print_debug:
            print("\n=== Prefill Assignment ===")
            for p in range(len(midpoint_prompt_lengths)):
                isl = assignment[p]
                print(f"Range {p} ({min_prompt_lengths[p]}-{max_prompt_lengths[p]} tokens) "
                      f"→ Island {isl.id}: {results[p][isl.id]:.2f}s")
            print("\n=== Island Totals ===")
            for isl in prefill_islands:
                print(f"Island {isl.id} total prefill time: {island_loads[isl.id]:.2f}s")
            print(f"\nMakespan: {makespan:.2f}s")

        return makespan

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
    slots = [Island('a1', 'prefill', 'DGX-B300', dp=1, tp=4), Island('a2', 'prefill', 'DGX-B300', tp=2, dp=2), Island('a3', 'prefill', 'H200', dp=8, tp=4)]

    # Pack the prefill bins
    packer.solve_prefill(slots, 10000)

    # print(f"Prefill speed: {prefill_speed:.2f}")
    # print("Prefill configuration:")
    # for bin, island in prefill_config.items():
    #     print(f"  Island (gpu_type={island.gpu_type}, size={island.size}) "
    #         f"-> Bin (prompt_max={bin.prompt_max}, decode_max={bin.decode_max})")
        

    # # plot the filtered traces as a histogram
    # import matplotlib.pyplot as plt

    # plt.figure(figsize=(10, 6))
    # for i, filtered_trace in enumerate(filtered_traces):
    #     if not filtered_trace:
    #         continue
    #     # Extract prompt lengths
    #     prompt_lengths = [p for p, _ in filtered_trace]
    #     # Compute bin edges with width 32
    #     min_len, max_len = min(prompt_lengths), max(prompt_lengths)
    #     bins = np.arange(min_len, max_len + 32, 32)
    #     plt.hist(prompt_lengths, bins=bins, alpha=0.5, label=f'Bin {i+1}')
    # plt.xlabel('Prompt Length')
    # plt.ylabel('Frequency')
    # plt.title('Filtered Traces Histogram (bin width = 10)')
    # plt.legend()
    # plt.show()