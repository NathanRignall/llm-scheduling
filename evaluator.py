# Scheduling of Distributed LLM Serving on Heterogeneous GPUs
# Nathan Rignall
# Evaluator - Takes sample configuration, evaluates using the simulator, and returns throughput values.

import os
import pandas as pd
import random
import string
import math
import numpy as np

import simulator

def generate_random_string(length):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

class Island:
    def __init__(self, gpu_type, size, id=None):
        self.id = id if id is not None else generate_random_string(8)
        self.gpu_type = gpu_type
        self.size = size
        self.prefill_assignments = []
        self.decode_assignments = []
    def assign_prefill(self, assignment):
        self.prefill_assignments.append(assignment)
    def assign_decode(self, assignment):
        self.decode_assignments.append(assignment)
class Bin:
    def __init__(self, min, max, dp, tp, bs, id=None):
        self.id = id if id is not None else generate_random_string(8)
        self.min = min
        self.max = max
        self.dp = dp
        self.tp = tp
        self.bs = bs
class Evaluator:
    def __init__(self, gpu_types, gpu_info_file='./device/gpu_info.csv'):
        self.gpu_types = gpu_types
        self.args = simulator.ModelArgs()
        self.prefill_gpu_list = simulator.get_gpu_info(gpu_info_file, decoding_mode=False, device_list=gpu_types, discount_rate=0.85)
        self.decode_gpu_list = simulator.get_gpu_info(gpu_info_file, decoding_mode=True, device_list=gpu_types, discount_rate=0.85)
        self.decode_len_avg = 100
        self.decode_len_max = 1000

    def evaluate(self, islands, trace_pdf, id, print_debug=True, path="{path}/"):
        # store the throughput for each assignment
        results = {}

        # fill results with trace_pdf
        for _, (sequence_length, probability) in enumerate(zip(trace_pdf[0], trace_pdf[1])):
            results[str(sequence_length)] = {
                "probability": probability,
                "prefill_throughputs": {},
                "decode_throughputs": {},
            }

        # calculate the throughput for each island
        for island_id, island in islands.items():
            # for each prefill assignment in the island
            for (bin, value) in island.prefill_assignments:
                # sum the probabilities for each sequence length in the bin
                probability_sum = 0

                # for any sequence length in within the assignment
                for idx, sequence_length in enumerate(trace_pdf[0]):
                    if sequence_length >= bin.min and sequence_length < bin.max:
                        # calculate the throughput for each assignment
                        time_ms = simulator.prefill_time(
                            self.args,
                            self.prefill_gpu_list[island.gpu_type],
                            seq_len=sequence_length,
                            kv_cache_rate=1,
                            tp_num=bin.tp,
                            dp_num=bin.dp
                        )

                        time_s = time_ms / 1000
                        request_per_second = 1 / time_s

                        # store the throughput in the results dictionary
                        results[str(sequence_length)]["prefill_throughputs"][f"{island_id}"] = {
                            "request_per_second": request_per_second,
                            "value": value,
                        }

                        # add the probability to the sum
                        probability_sum += results[str(sequence_length)]["probability"]

                # go back to the results dictionary and add the probability sum
                for idx, sequence_length in enumerate(trace_pdf[0]):
                    if sequence_length >= bin.min and sequence_length < bin.max:
                        for island_id, throughput in results[str(sequence_length)]["prefill_throughputs"].items():
                            # calculate the weighted probability
                            throughput["weighted_probability"] = results[str(sequence_length)]["probability"] / probability_sum
                            # calculate the weighted throughput
                            throughput["weighted_request_per_second"] = throughput["request_per_second"] * throughput["weighted_probability"] * throughput["value"]
                            
            # for each decode assignment in the island
            for  (bin, value) in island.decode_assignments:
                # sum the probabilities for each sequence length in the bin
                probability_sum = 0

                # calculate the number of devices
                loop_num_gpus = island.size // bin.dp

                # for any sequence length in within the assignment
                for idx, sequence_length in enumerate(trace_pdf[0]):
                    if sequence_length >= bin.min and sequence_length < bin.max:
                        # compute the throughput for each token
                        total_time_ms = 0
                        for current_decode_length in range(1, self.decode_len_avg, 1):
                            time_ms, _ = simulator.decode_time(
                                self.args,
                                self.decode_gpu_list[island.gpu_type],
                                tp_num=bin.tp,
                                bs_num=bin.bs,
                                seq_len=sequence_length,
                                decode_len=current_decode_length,
                                gemm_group_per_device=math.ceil(self.args.n_routed_experts / loop_num_gpus),
                                device_num=loop_num_gpus,
                            )

                            # don't continue if the time is infinite
                            if time_ms == np.inf:
                                total_time_ms = np.inf
                                #print(f"Error: Infinite time for sequence length {sequence_length} and decode length {current_decode_length} on island {island_id}.")
                                break
                            else:
                                total_time_ms += time_ms

                        time_s = total_time_ms / 1000
                        request_per_second = (1 / time_s * bin.bs) * bin.dp if total_time_ms != np.inf else 0

                        # store the throughput in the results dictionary
                        results[str(sequence_length)]["decode_throughputs"][f"{island_id}"] = {
                            "request_per_second": request_per_second,
                            "value": value,
                        }

                        # add the probability to the sum
                        probability_sum += results[str(sequence_length)]["probability"]

                # go back to the results dictionary and add the probability sum
                for idx, sequence_length in enumerate(trace_pdf[0]):
                    if sequence_length >= bin.min and sequence_length < bin.max:
                        for island_id, throughput in results[str(sequence_length)]["decode_throughputs"].items():
                            # calculate the weighted probability
                            throughput["weighted_probability"] = results[str(sequence_length)]["probability"] / probability_sum
                            # calculate the weighted throughput
                            throughput["weighted_request_per_second"] = throughput["request_per_second"] * throughput["weighted_probability"] * throughput["value"]

        # print the throughput dictionary
        if print_debug:
            print("\nResults Dictionary:")
            for key, value in results.items():
                print(f"{key}: {value}")

        # sum the throughput for each sequence length
        for sequence_length, result in results.items():
            # sum the prefill throughput for each island
            total_prefill_throughput = 0
            for island_id, throughput in result["prefill_throughputs"].items():
                total_prefill_throughput += throughput["weighted_request_per_second"]

            # sum the decode throughput for each island
            total_decode_throughput = 0
            for island_id, throughput in result["decode_throughputs"].items():
                total_decode_throughput += throughput["weighted_request_per_second"]

            # store the total throughput in the results dictionary
            results[sequence_length]["total_prefill_throughput"] = total_prefill_throughput
            results[sequence_length]["total_decode_throughput"] = total_decode_throughput

        # make folder for id
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(f"{path}/{id}"):
            os.makedirs(f"{path}/{id}")

        # save throughput values to a csv file
        with open(f"{path}/{id}/eval_prefill_throughput.csv", "w") as f:
            f.write("Sequence,Throughput\n")
            for sequence_length, result in results.items():
                f.write(f"{sequence_length},{result['total_prefill_throughput']}\n")
        with open(f"{path}/{id}/eval_decode_throughput.csv", "w") as f:
            f.write("Sequence,Throughput\n")
            for sequence_length, result in results.items():
                f.write(f"{sequence_length},{result['total_decode_throughput']}\n")    

        # sum total_prefill_throughput
        total_prefill_throughput = 0
        for sequence_length, result in results.items():
            total_prefill_throughput += result["total_prefill_throughput"]

        # sum total_decode_throughput
        total_decode_throughput = 0
        for sequence_length, result in results.items():
            total_decode_throughput += result["total_decode_throughput"]

        return total_prefill_throughput, total_decode_throughput
    
def load_trace_pdf(trace_pdf_path):
    # open csv file
    trace_df = pd.read_csv(trace_pdf_path)

    # get the sequence length and probability
    sequence_length = trace_df["Length"].values
    probability = trace_df["Probability"].values

    # return as a tuple
    return (sequence_length, probability)
