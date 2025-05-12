import simulator
import numpy as np
import random
import string
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, PULP_CBC_CMD

def generate_random_string(length):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
class Island:
    def __init__(self, role, gpu_type, dp = 1, tp = 1, size = 0):
        self.id = generate_random_string(8)
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
    def __init__(self, gpu_types):
        self.gpu_types = gpu_types
        self.args = simulator.ModelArgs()
        self.gpu_list = simulator.get_gpu_info('./device/gpu_info.csv', decoding_mode=False, device_list=gpu_types, discount_rate=0.85)

    # test each island with range of prompt lengths to find best bins
    def solve_prefill(self, islands, print_debug=True):
        # create a list of prompt lengths evenly spaced
        prompt_lengths_min = np.arange(0, 1024, 64).astype(int)
        prompt_lengths_max = prompt_lengths_min + 64
        prompt_lengths_mid = (prompt_lengths_min + prompt_lengths_max) // 2

        # extract the prefill islands
        prefill_islands = [island for island in islands if island.role == 'prefill']

        print("\n=== Prefill Islands ===")
        for island in prefill_islands:
            print(f"Island (id={island.id}, gpu_type={island.gpu_type}, size={island.size}, dp={island.dp}, tp={island.tp}, role={island.role})")

        # measure the time for each prompt length
        results = []

        for prompt_length in prompt_lengths_mid:
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

        # create the problem
        problem = LpProblem("Prefill_Packing_Problem", LpMinimize)

        # decision vars
        x = {
            (isl.id, p): LpVariable(f"x_island{isl.id}_mid{p}", cat=LpBinary)
            for isl in prefill_islands
            for p in range(len(prompt_lengths_mid))
        }
        # makespan variable
        M = LpVariable("Makespan", lowBound=0)

        # objective: minimize the makespan
        problem += M

        # each prompt range p goes to exactly one island
        for p in range(len(prompt_lengths_mid)):
            problem += lpSum(x[(isl.id, p)] for isl in prefill_islands) == 1

        # for each island, its total assigned time ≤ M
        for isl in prefill_islands:
            problem += lpSum(
                results[p][isl.id] * x[(isl.id, p)]
                for p in range(len(prompt_lengths_mid))
            ) <= M

        # solve
        problem.solve(PULP_CBC_CMD(
            msg=False,
            threads=8,
            timeLimit=60,
        ))

        # extract
        assignment = {}
        for p in range(len(prompt_lengths_mid)):
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
            for p in range(len(prompt_lengths_mid)):
                isl = assignment[p]
                print(f"Range {p} ({prompt_lengths_min[p]}-{prompt_lengths_max[p]} tokens) "
                      f"→ Island {isl.id}: {results[p][isl.id]:.2f}s")
            print("\n=== Island Totals ===")
            for isl in prefill_islands:
                print(f"Island {isl.id} total prefill time: {island_loads[isl.id]:.2f}s")
            print(f"Makespan: {makespan:.2f}s")

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
    gpu_types = ["DGX-B300", "H20"]

    # Create a packer instance
    packer = Packer(gpu_types)

    # Define bins and slots
    slots = [
        Island('prefill', 'DGX-B300', tp=4, dp=2),
        Island('prefill', 'DGX-B300', tp=4, dp=2),
        Island('prefill', 'DGX-B300', tp=4, dp=2),
        Island('prefill', 'DGX-B300', tp=4, dp=2),
        Island('prefill', 'DGX-B300', tp=4, dp=2),
        Island('prefill', 'DGX-B300', tp=4, dp=2),
        Island('prefill', 'DGX-B300', tp=4, dp=2),
        Island('prefill', 'DGX-B300', tp=4, dp=2)
    ]

    # Pack the prefill bins
    time = packer.solve_prefill(slots, print_debug=True)

    print(f"\nTotal prefill time: {time:.2f}s")