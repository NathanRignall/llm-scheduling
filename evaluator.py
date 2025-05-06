import itertools
import math

import simulator

class Island:
    def __init__(self, gpu_type, size, dp, tp, role):
        self.gpu_type = gpu_type
        self.size = size
        self.dp = dp
        self.tp = tp
        self.role = role

class Bin:
    def __init__(self, prompt_max, decode_max):
        self.prompt_max = prompt_max
        self.decode_max = decode_max

class Evaluator:
    def __init__(self, inventory, gpu_types):
        self.inventory = inventory
        self.gpu_types = gpu_types
        self.args = simulator.ModelArgs()
        self.gpu_list = simulator.get_gpu_info('./device/gpu_info.csv', decoding_mode=True, device_list=gpu_types, discount_rate=0.85)

        # check if GPUs in inventory are in gpu_list
        for gpu_type in gpu_types:
            if gpu_type not in self.gpu_list:
                raise ValueError(f"GPU type {gpu_type} not found in GPU list.")

    def evaluate(self, bins, slots):
        
        configs = []
        
        # For each permutation of bins of length len(slots)
        for perm in itertools.permutations(bins, len(slots)):
            mapping = dict(zip(slots, perm))
            configs.append(mapping)

        # Evaluate each configuration
        best_rho = float('inf')
        for config in configs:
            rho = self._evaluate_config(config)
            if rho is not None and rho < best_rho:
                best_rho = rho

        return best_rho if best_rho != float('inf') else None
    
    def _evaluate_config(self, config):
        # Initialize rho
        rho = 0

        # Check if the configuration is valid
        for slot, bin in config.items():
            if slot.size <= 0:
                continue
            
            # Calculate rho for this slot-bin pair
            rho_slot = self._calculate_rho(slot, bin)

            if rho_slot is None:
                return None
            
            rho += rho_slot

        return rho
    
    def _calculate_rho(self, slot, bin):
        gpu_type = slot.gpu_type
        size = slot.size
        dp_num = slot.dp
        tp_num = slot.tp

        bs_num = 8

        seq_len = bin.decode_max
        decode_len = bin.prompt_max
        fp8_combine = False

        # Check if the GPU type is valid
        if gpu_type not in self.gpu_types:
            return None
        
        # get device info
        device = self.gpu_list[gpu_type]
        
        if slot.role == "decoder":
            # get the number of devices
            gemm_group_per_device = math.ceil(self.args.n_routed_experts / size)

            # Simulate the decoding time
            total = simulator.decode_time(
                self.args,
                device,
                dp_num,
                tp_num,
                bs_num,
                seq_len,
                decode_len,
                gemm_group_per_device,
                size
            )

            return None if total is None else 1 / total
        else:
            # Simulate the prefill time
            total = simulator.prefill_time(
                self.args,
                device,
                seq_len,
                kv_cache_rate=1,
                tp_num=tp_num,
                dp_num=size
            )
            return None if total is None else 1 / total
        
# Example usage
if __name__ == "__main__":
    # Define the inventory and GPU types
    inventory = {
        'H200': 32,
        'H800': 16,
    }
    gpu_types = ['H200', 'H800']

    # Create an evaluator instance
    evaluator = Evaluator(inventory, gpu_types)

    # Define bins and slots
    bins = [Bin(1024, 2048), Bin(512, 1024)]
    slots = [Island('H200', 32, 1, 1, 'decoder'), Island('H800', 2, 1, 1, 'prefill')]

    # Evaluate the configuration
    best_rho = evaluator.evaluate(bins, slots)
    print(f"Best rho: {best_rho}")