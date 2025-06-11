# Scheduling of Distributed LLM Serving on Heterogeneous GPUs
# Nathan Rignall
# Divider - Takes a total number of GPUs for a specific type and divides them into islands

import math

# divide the total number of GPUs for a specific type into islands
def island_divider(total_gpus_for_type: int, num_islands_target: int, size_distribution_skew_exponent: float, min_island_size: int = 1) -> list[int]:
    # check constraints
    if min_island_size <= 0:
        raise ValueError("Minimum island size must be positive.")
    if total_gpus_for_type < 0:
        raise ValueError("Total GPUs for type cannot be negative.")
    if num_islands_target < 0:
        raise ValueError("Target number of islands cannot be negative.")

    if num_islands_target == 0:
        return []

    gpus_to_allocate = total_gpus_for_type

    if gpus_to_allocate == 0:
        return []

    if gpus_to_allocate < min_island_size:
        return []
        
    max_possible_islands_with_min_size = math.floor(gpus_to_allocate / min_island_size)
    actual_num_islands = min(num_islands_target, max_possible_islands_with_min_size)
    
    if actual_num_islands == 0:
        return []

    # compute the base allocation per island
    base_allocation_per_island = min_island_size
    gpus_used_for_base = actual_num_islands * base_allocation_per_island
    remaining_gpus_to_distribute = gpus_to_allocate - gpus_used_for_base

    final_island_sizes = [base_allocation_per_island] * actual_num_islands

    if remaining_gpus_to_distribute == 0:
        return final_island_sizes

    if actual_num_islands == 1:
        final_island_sizes[0] += remaining_gpus_to_distribute
        return final_island_sizes

    # distribute the remaining GPUs according to the skew exponent
    weights = []
    for i in range(actual_num_islands):
        base_val = float(i + 1) 
        if math.isclose(size_distribution_skew_exponent, 0.0):
            weights.append(1.0)
        else:
            weights.append(math.pow(base_val, size_distribution_skew_exponent))
    
    sum_weights = sum(weights)
    if math.isclose(sum_weights, 0.0) or sum_weights <= 0: 
        proportions = [1.0 / actual_num_islands] * actual_num_islands
    else:
        proportions = [w / sum_weights for w in weights]

    additional_allocations_float = [p * remaining_gpus_to_distribute for p in proportions]
    additional_allocations_int = [math.floor(val) for val in additional_allocations_float]
    current_sum_additional_int = sum(additional_allocations_int)
    discrepancy_to_distribute = int(round(remaining_gpus_to_distribute - current_sum_additional_int))

    for i in range(actual_num_islands):
        final_island_sizes[i] += additional_allocations_int[i]

    # if there is a discrepancy, distribute it based on the fractional parts
    if discrepancy_to_distribute > 0: 
        fractional_parts = []
        for i in range(actual_num_islands):
            fractional_parts.append(
                (additional_allocations_float[i] - additional_allocations_int[i], i)
            )
        fractional_parts.sort(key=lambda x: x[0], reverse=True)
        
        for k in range(discrepancy_to_distribute):
            island_idx_to_increment = fractional_parts[k % actual_num_islands][1]
            final_island_sizes[island_idx_to_increment] += 1
    return final_island_sizes
