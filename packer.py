import math
import numpy as np
import pulp
import simulator
import evaluator

class Packer:
    def __init__(self, gpu_types):
        self.gpu_types = gpu_types
        self.args = simulator.ModelArgs()
        self.prefill_gpu_list = simulator.get_gpu_info('./device/gpu_info.csv', decoding_mode=False, device_list=gpu_types, discount_rate=0.85)
        self.decode_gpu_list = simulator.get_gpu_info('./device/gpu_info.csv', decoding_mode=True, device_list=gpu_types, discount_rate=0.85)
        self.decode_len_avg = 100
        self.decode_len_max = 1000
        self.batch_size = 32

    def solve_linear(self, islands, trace_pdf, resolution=10, print_debug=True):
        print("\n=== Packer ===")

        print("*** Forming Ranges ***")
        # store probabilities
        ranges = {}
        range_id = 0
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
                ranges[range_id] = {
                    'sequence_length': sequence_length,
                    'probability': probability,
                    'min': bin_start,
                    'max': bin_end,
                }

                # increment total probability
                total_probability += probability

                # increment the range index
                range_id += 1

        # check if the total probability is equal to 1
        if print_debug:
            print(f"Total probability: {total_probability:.4f}")

        # for each island, compute the prefill time
        print("*** Prefill Benchmark ***")
        prefill_benchmark = {}
        for index, (island_id, island) in enumerate(islands.items()):
            # for each range, compute the time
            for range_id, range in ranges.items():
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
                prefill_benchmark[(island_id, range_id)] = request_per_second

        # for each island, compute the decode time
        print("*** Decode Benchmark ***")
        decode_benchmark = {}
        for index, (island_id, island) in enumerate(islands.items()):
            failed = False
            # for each range, compute the time
            for range_id, range in ranges.items():
                # if failed, skip the range
                if failed:
                    decode_benchmark[(island_id, range_id)] = 0
                    continue

                # check if the range can support the largest decode length
                max_bs = simulator.check_max_bs(
                    self.args,
                    self.decode_gpu_list[island.gpu_type],
                    tp_num=island.tp,
                    seq_len=range['max'],
                    decode_len=self.decode_len_max,
                    gemm_group_per_device=math.ceil(self.args.n_routed_experts / island.size),
                    device_num=island.size,
                )

                # don't continue if the time is infinite
                if max_bs < self.batch_size:
                    if print_debug:
                        print(f"Island {island_id} cannot support the decode length for range {range_id}. Batch size: {self.batch_size}, Max batch size: {max_bs}")
                    failed = True
                    decode_benchmark[(island_id, range_id)] = 0
                    continue

                # compute the throughput for each token
                total_time_ms = 0
                for current_decode_length in __builtins__.range(1, self.decode_len_avg, 1):
                    time_ms, _ = simulator.decode_time(
                        self.args,
                        self.decode_gpu_list[island.gpu_type],
                        tp_num=island.tp,
                        bs_num=self.batch_size,
                        seq_len=range['sequence_length'] + current_decode_length,
                        decode_len=self.decode_len_avg - current_decode_length,
                        gemm_group_per_device=math.ceil(self.args.n_routed_experts / island.size),
                        device_num=island.size,
                    )

                    # don't continue if the time is infinite
                    if time_ms == np.inf:
                        total_time_ms = np.inf
                        if print_debug:
                            print(f"Island {island_id} cannot support the decode length for range {range_id}.")
                        break
                    else:
                        total_time_ms += time_ms

                time_s = total_time_ms / 1000
                request_per_second = (1 / time_s * self.batch_size ) if total_time_ms != np.inf else 0

                # store the result
                decode_benchmark[(island_id, range_id)] = request_per_second 

        # print the results
        if print_debug:
            print("\n=== Ranges ===")
            for range_id, range in ranges.items():
                print(f"Range {range_id}: {range['sequence_length']} tokens, Probability: {range['probability']:.4f}, Range: {range['min']}-{range['max']} tokens")

            print("\n=== Prefill Benchmark ===")
            for island_id, range_id in prefill_benchmark.keys():
                print(f"Island {island_id}, Range {range_id}: {prefill_benchmark[(island_id, range_id)]:.4f} requests/s")

            print("\n=== Decode Benchmark ===")
            for island_id, range_id in decode_benchmark.keys():
                print(f"Island {island_id}, Range {range_id}: {decode_benchmark[(island_id, range_id)]:.4f} requests/s")

        # extract Island IDs and sequence lengths
        island_ids = [island_id for island_id, _ in islands.items()]
        range_ids = [range_id for range_id in ranges.keys()]
        
        # create the problem
        print("*** Creating Problem ***")
        problem = pulp.LpProblem("Linearized_LoadBalance", pulp.LpMaximize)

        # variables
        x_prefill = pulp.LpVariable.dicts("x_prefill", (island_ids, range_ids), lowBound=0)
        x_decode = pulp.LpVariable.dicts("x_decode", (island_ids, range_ids), lowBound=0)

        Rj_prefill = pulp.LpVariable.dicts("Rj_prefill", range_ids, lowBound=0)
        Rj_decode = pulp.LpVariable.dicts("Rj_decode", range_ids, lowBound=0)

        R_prefill = pulp.LpVariable("R_prefill", lowBound=0)
        R_decode = pulp.LpVariable("R_decode", lowBound=0)

        delta_prefill = pulp.LpVariable.dicts("delta_prefill", range_ids, lowBound=0)
        delta_decode = pulp.LpVariable.dicts("delta_decode", range_ids, lowBound=0)
        delta_match = pulp.LpVariable.dicts("delta_match", range_ids, lowBound=0)

        D_total = pulp.LpVariable("D_total", lowBound=0)

        z = pulp.LpVariable.dicts("z", island_ids, cat='Binary')

        # constraints
        tau = 0.1
        M = 100
        lambda_dev = {j: M/(1 + ranges[j]['probability']) for j in range_ids}

        # prevent zero assignments if the benchmark is zero
        for i in island_ids:
            for j in range_ids:
                if prefill_benchmark[(i, j)] == 0:
                    x_prefill[i][j].upBound = 0
                if decode_benchmark[(i, j)] == 0:
                    x_decode[i][j].upBound = 0

        ## each island does either decode or prefill and
        #for island_id in island_ids:
        #    problem += pulp.lpSum(x_decode[island_id][j] for j in range_ids) <= z[island_id], f"DecodeRole_{island_id}"
        #    problem += pulp.lpSum(x_prefill[island_id][j] for j in range_ids) <= 1 - z[island_id], f"PrefillRole_{island_id}"
        #    problem += pulp.lpSum(x_decode[island_id][j] + x_prefill[island_id][j] for j in range_ids) == 1, f"TotalLoad_{island_id}"
            
        # each island picks exactly one (mode, range), and is either decode or prefill
        for i in island_ids:
            # 1) each island must assign exactly one unit of load
            problem += (
                pulp.lpSum(x_decode[i][j] + x_prefill[i][j]
                        for j in range_ids) == 1,
                f"TotalLoad_{i}"
            )
            # 2) tight Big-M per (i,j):
            for j in range_ids:
                # if z[i] = 1 (decode island) → x_prefill[i,j] = 0
                problem += (
                    x_prefill[i][j] <= (1 - z[i]),
                    f"link_prefill_{i}_{j}"
                )
                # if z[i] = 0 (prefill island) → x_decode[i,j] = 0
                problem += (
                    x_decode[i][j] <= z[i],
                    f"link_decode_{i}_{j}"
                )
                
        # define Rj and R totals
        for j in range_ids:
            problem += pulp.lpSum(decode_benchmark[(i, j)] * x_decode[i][j] for i in island_ids) == Rj_decode[j], f"Define_RjDecode_{j}"
            problem += pulp.lpSum(prefill_benchmark[(i, j)] * x_prefill[i][j] for i in island_ids) == Rj_prefill[j], f"Define_RjPrefill_{j}"

        problem += pulp.lpSum(Rj_decode[j] for j in range_ids) == R_decode, "Total_R_decode"
        problem += pulp.lpSum(Rj_prefill[j] for j in range_ids) == R_prefill, "Total_R_prefill"

        # performance of decode and prefill cannot be zero
        problem += R_prefill >= 0.01, "Prefill_Throughput"
        problem += R_decode >= 0.01, "Decode_Throughput"

        # deviation constraints for decode
        for j in range_ids:
            pj = ranges[j]['probability']
            problem += Rj_decode[j] - pj * R_decode <= delta_decode[j], f"DevDecodePos_{j}"
            problem += pj * R_decode - Rj_decode[j] <= delta_decode[j], f"DevDecodeNeg_{j}"
            problem += delta_decode[j] <= tau * pj * R_decode, f"HardCapDecode_{j}"

        # deviation constraints for prefill
        for j in range_ids:
            pj = ranges[j]['probability']
            problem += Rj_prefill[j] - pj * R_prefill <= delta_prefill[j], f"DevPrefillPos_{j}"
            problem += pj * R_prefill - Rj_prefill[j] <= delta_prefill[j], f"DevPrefillNeg_{j}"
            problem += delta_prefill[j] <= tau * pj * R_prefill, f"HardCapPrefill_{j}"

        # match decode and prefill throughput
        for j in range_ids:
            problem += Rj_decode[j] - Rj_prefill[j] <= delta_match[j], f"DevMatchPos_{j}"
            problem += Rj_prefill[j] - Rj_decode[j] <= delta_match[j], f"DevMatchNeg_{j}"

        # total deviation
        problem += (
            pulp.lpSum(delta_decode[j] + delta_prefill[j] + delta_match[j] for j in range_ids) == D_total,
            "Define_D_total"
        )

        # objective function
        problem += (
            R_decode + R_prefill
            - pulp.lpSum((lambda_dev[j] * delta_match[j]) + delta_decode[j] + delta_prefill[j] for j in range_ids)
        ), "Objective"

        # solve
        print("*** Solving Problem ***")
        problem.solve(pulp.PULP_CBC_CMD(
            msg=print_debug,
            threads=8,
        ))

        if print_debug:
            print("\n=== Prefill Allocations ===")
            for island_id in island_ids:
                for range_id in range_ids:
                    print(f"  Island {island_id}, Range {range_id}: {x_prefill[island_id][range_id].varValue:.4f} requests/s")

            print("\n=== Decode Allocations ===")
            for island_id in island_ids:
                for range_id in range_ids:
                    print(f"  Island {island_id}, Range {range_id}: {x_decode[island_id][range_id].varValue:.4f} requests/s")

            print("\n=== Prefill Range performance and deviation ===")
            for range_id in range_ids:
                rj = Rj_prefill[range_id].varValue
                dj = delta_prefill[range_id].varValue
                print(f"  j={range_id}: Rj={rj:.4f}, δ={dj:.4f}  (target {ranges[range_id]['probability'] * R_prefill.varValue:.4f})")

            print("\n=== Decode Range performance and deviation ===")
            for range_id in range_ids:
                rj = Rj_decode[range_id].varValue
                dj = delta_decode[range_id].varValue
                print(f"  j={range_id}: Rj={rj:.4f}, δ={dj:.4f}  (target {ranges[range_id]['probability'] * R_decode.varValue:.4f})")

        # check if the problem is infeasible
        if pulp.LpStatus[problem.status] == "Infeasible":
            print("Problem is infeasible. Please check the constraints.")
            return None, None, None, None, None

        # extract the assignment
        modes = {}
        prefill_assignment = {}
        decode_assignment = {}
        for island_id in island_ids:
            for range_id in range_ids:
                modes[(island_id, range_id)] = z[island_id].varValue
                prefill_assignment[(island_id, range_id)] = x_prefill[island_id][range_id].varValue
                decode_assignment[(island_id, range_id)] = x_decode[island_id][range_id].varValue

                # throw an error if both are assigned
                if x_prefill[island_id][range_id].varValue > 0 and x_decode[island_id][range_id].varValue > 0:
                    raise ValueError(f"Both prefill and decode are assigned for island {island_id} and range {range_id}.")

        # extract the throughput and deviation
        prefill_throughput = {}
        prefill_deviation = {}
        decode_throughput = {}
        decode_deviation = {}
        for range_id in range_ids:
            prefill_throughput[range_id] = Rj_prefill[range_id].varValue
            prefill_deviation[range_id] = delta_prefill[range_id].varValue
            decode_throughput[range_id] = Rj_decode[range_id].varValue
            decode_deviation[range_id] = delta_decode[range_id].varValue

        # construct model
        model = {
            'ranges': ranges,
            'prefill': {
                'assignment': prefill_assignment,
                'throughput': prefill_throughput,
                'deviation': prefill_deviation,
                'benchmark': prefill_benchmark,
            },
            'decode': {
                'assignment': decode_assignment,
                'throughput': decode_throughput,
                'deviation': decode_deviation,
                'benchmark': decode_benchmark,
            },
            'modes': modes,
        }

        if True:
            # print the binary variables (to see if prefill or decode is selected)
            print("\n=== Binary Variables ===")
            for island_id in island_ids:
                print(f"  Island {island_id}: {"Decode" if z[island_id].varValue == 1 else "Prefill"}")

            # print the overall stats
            print("\n=== Overall Stats ===")
            print(f"  Prefill Throughput: {R_prefill.varValue:.4f} requests/s")
            print(f"  Decode Throughput: {R_decode.varValue:.4f} requests/s")
            print(f"  Total Deviation: {D_total.varValue:.4f}")
            print(f"  Objective: {pulp.value(problem.objective):.4f}")

        # return the assignment
        return model, pulp.value(R_prefill), pulp.value(R_decode), pulp.value(D_total), pulp.value(problem.objective)

# Example usage
if __name__ == "__main__":
    gpu_types = ["RubinU-NVL576", "H200", "H800", "H20"]

    # Create a packer instance
    packer = Packer(gpu_types)

    # Define bins and slots
    islands = [
        evaluator.Island(gpu_type="RubinU-NVL576", dp=1, tp=1, size=1, id="a"),
        evaluator.Island(gpu_type="RubinU-NVL576", dp=2, tp=1, size=1, id="b"),
        evaluator.Island(gpu_type="RubinU-NVL576", dp=4, tp=1, size=1, id="c"),
        evaluator.Island(gpu_type="RubinU-NVL576", dp=1, tp=2, size=1, id="d"),
        evaluator.Island(gpu_type="RubinU-NVL576", dp=2, tp=2, size=1, id="e"),
        evaluator.Island(gpu_type="RubinU-NVL576", dp=4, tp=2, size=1, id="f"),
        evaluator.Island(gpu_type="RubinU-NVL576", dp=1, tp=4, size=1, id="g"),
        evaluator.Island(gpu_type="RubinU-NVL576", dp=2, tp=4, size=1, id="h"),
        evaluator.Island(gpu_type="RubinU-NVL576", dp=4, tp=4, size=1, id="i"),
        evaluator.Island(gpu_type="RubinU-NVL576", dp=4, tp=2, size=128, id="j"),
        evaluator.Island(gpu_type="RubinU-NVL576", dp=4, tp=4, size=128, id="k"),
        evaluator.Island(gpu_type="RubinU-NVL576", dp=4, tp=8, size=128, id="l"),
        evaluator.Island(gpu_type="H200", dp=1, tp=1, size=1, id="m"),
        evaluator.Island(gpu_type="H200", dp=2, tp=1, size=1, id="n"),
        evaluator.Island(gpu_type="H200", dp=4, tp=1, size=1, id="o"),
        evaluator.Island(gpu_type="H200", dp=1, tp=2, size=1, id="p"),
        evaluator.Island(gpu_type="H200", dp=2, tp=2, size=1, id="q"),
        evaluator.Island(gpu_type="H20", dp=4, tp=2, size=1, id="r"),
        evaluator.Island(gpu_type="H20", dp=4, tp=4, size=1, id="s"),
        evaluator.Island(gpu_type="H20", dp=4, tp=8, size=1, id="t"),
        evaluator.Island(gpu_type="H800", dp=4, tp=2, size=64, id="u"),
        evaluator.Island(gpu_type="H800", dp=4, tp=4, size=64, id="v"),
        evaluator.Island(gpu_type="H800", dp=4, tp=8, size=64, id="w"),
    ]
    islands = {island.id: island for island in islands}

    # load the trace PDF
    trace_pdf = evaluator.load_trace_pdf("traces/generated_trace_pdf.csv")

    # Pack the prefill bins
    model, prefill_throughput, decode_throughput, delta, objective = packer.solve_linear(islands, trace_pdf, resolution=100, print_debug=False)

    # check if the model is not None
    if model is not None:
        print("\n=== Packer Results ===")
        print(f"Prefill Throughput: {prefill_throughput:.4f} requests/s")
        print(f"Decode Throughput: {decode_throughput:.4f} requests/s")
        print(f"Deviation: {delta:.4f}")
        print(f"Objective: {objective:.4f}")

        # add assignments to the islands
        for (island_id, range_id), value in model['prefill']['assignment'].items():
            islands[island_id].assign_prefill(
                (
                    evaluator.Bin(
                        min=model['ranges'][range_id]['min'],
                        max=model['ranges'][range_id]['max'],
                    ),
                    value
                )
            )
        for (island_id, range_id), value in model['decode']['assignment'].items():
            islands[island_id].assign_decode(
                (
                    evaluator.Bin(
                        min=model['ranges'][range_id]['min'],
                        max=model['ranges'][range_id]['max'],
                    ),
                    value
                )
            )

        # save model throughputs values to csv
        with open("./data/scratch/model_prefill_throughput.csv", "w") as f:
            f.write("Sequence,Throughput\n")
            for range_id, throughput in model['prefill']['throughput'].items():
                f.write(f"{model['ranges'][range_id]['sequence_length']},{throughput}\n")
        with open("./data/scratch/model_decode_throughput.csv", "w") as f:
            f.write("Sequence,Throughput\n")
            for range_id, throughput in model['decode']['throughput'].items():
                f.write(f"{model['ranges'][range_id]['sequence_length']},{throughput}\n")

        # save the assignment to a csv
        with open("./data/scratch/model_assignment.csv", "w") as f:
            f.write("Island,Range,Prefill_Assignment,Decode_Assignment,Prefill_Benchmark,Decode_Benchmark,Prefill_Assignment_Benchmark,Decode_Assignment_Benchmark,Mode\n")
            for island_id in islands.keys():
                for range_id in model['ranges'].keys():
                    f.write(f"{island_id},{range_id},{model['prefill']['assignment'][(island_id, range_id)]},{model['decode']['assignment'][(island_id, range_id)]},{model['prefill']['benchmark'][(island_id, range_id)]},{model['decode']['benchmark'][(island_id, range_id)]},{model['prefill']['assignment'][(island_id, range_id)] * model['prefill']['benchmark'][(island_id, range_id)]},{model['decode']['assignment'][(island_id, range_id)] * model['decode']['benchmark'][(island_id, range_id)]},{model['modes'][(island_id, range_id)]}\n")
                
        # pass to evaluator
        evaluator = evaluator.Evaluator(gpu_types)
        prefill_throughput, decode_throughput = evaluator.evaluate(islands, trace_pdf, print_debug=False)

        print("\n=== Evaluator Results ===")
        print(f"Prefill Throughput: {prefill_throughput:.4f} requests/s")
        print(f"Decode Throughput: {decode_throughput:.4f} requests/s")

    # do not print the model
    else:
        print("No solution found.")