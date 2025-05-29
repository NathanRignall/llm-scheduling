import random

import packer
import evaluator

# --- Define search space ---
GPU_TYPES = ["DGX-B300", "H200"]
INVENTORY = {"DGX-B300": 128, "H200": 256}

# --- Create packer and evaluator instances ---
sys_packer = packer.Packer(GPU_TYPES)

# --- Define trace PDF ---
trace_pdf = evaluator.load_trace_pdf("traces/generated_trace_pdf.csv")

# -- Test configuration --
def test_config(islands, trace_pdf):
    print("\n\n***** Testing configuration *****")

    # print the number of islands for each GPU type
    gpu_counts = {gpu_type: 0 for gpu_type in GPU_TYPES}
    for island in islands:
        gpu_counts[island.gpu_type] += 1
    print("Current configuration:")
    for gpu_type, count in gpu_counts.items():
        print(f"{gpu_type}: {count} islands")

    # count the total number of GPUs in the configuration
    total_gpus = sum(island.size for island in islands)
    print(f"Total GPUs in configuration: {total_gpus}")

    # Convert islands to a format suitable for the packer
    islands = {island.id: island for island in islands}
    
    # Pack the islands
    _, prefill_throughput, decode_throughput, _, _ = sys_packer.solve_linear(islands, trace_pdf, resolution=500, print_debug=False)

    # Pick lowest throughput if neither is None
    if prefill_throughput is None or decode_throughput is None:
        return None
    else:
        return min(prefill_throughput, decode_throughput)

# -- Create initial configuration --
def create_initial_config():
    islands = []

    # for each GPU type in inventory, create an island of that size
    for gpu_type in GPU_TYPES:
        island = evaluator.Island(
            gpu_type=gpu_type,
            size=INVENTORY[gpu_type],
        )
        islands.append(island)

    return islands

# -- Iterate configuration --
def iterate_config(islands):
    #print("\n\n***** Iterating configuration *****")
    
    # take a random island and halve its size
    if not islands:
        print("No islands to iterate.")
        return islands
    
    island = random.choice(islands)
    if island.size <= 1:
        print(f"Island {island.id} is too small to halve.")
        return islands
    
    # take the island and halve its size, then add a new island of the same type with the halved size
    new_size = island.size // 2
    new_island = evaluator.Island(
        gpu_type=island.gpu_type,
        size=new_size,
    )
    islands.remove(island)
    islands.append(new_island)
    islands.append(new_island)

    return islands

# -- Main loop --
def main_loop():
    # Create initial configuration
    islands = create_initial_config()

    # Test initial configuration
    throughput = test_config(islands, trace_pdf)
    print(f"Initial throughput: {throughput}")

    # Iterate configuration until no more halving is possible
    while True:
        new_islands = iterate_config(islands)
        new_throughput = test_config(new_islands, trace_pdf)

        print(f"New throughput: {new_throughput}")
        islands = new_islands
        throughput = new_throughput

if __name__ == "__main__":
    main_loop()
    print("Main loop completed.")