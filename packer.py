import math
import numpy as np
import pandas as pd
import pulp
import builtins
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

        # for each island, find the best gpu configuration during prefill
        print("*** Prefill GPU Config ***")
        prefill_configs = {}
        for index, (island_id, island) in enumerate(islands.items()):
            # create a list of configurations for the island to test
            configs = []
            for tp in builtins.range(1, 8 + 1):
                if island.size % tp == 0:
                    dp = island.size // tp
                    configs.append((tp, dp))

            # for each range, test the configurations
            results = []
            for range_id, range in ranges.items():
                # compute the throughput for each configuration
                for tp, dp in configs:
                    time_ms = simulator.prefill_time(
                        self.args,
                        self.prefill_gpu_list[island.gpu_type],
                        range['sequence_length'],
                        kv_cache_rate=1,
                        tp_num=tp,
                        dp_num=dp
                    )

                    time_s = time_ms / 1000
                    request_per_second = 1 / time_s

                    # store the result
                    results.append({
                        'tp': tp,
                        'dp': dp,
                        'range_id': range_id,
                        'throughput': request_per_second,
                    })

            # make a df of the results
            results_df = pd.DataFrame(results)

            # calculate the percentage difference between the best and each configuration for each range
            best_configs = results_df.loc[results_df.groupby('range_id')['throughput'].idxmax()]

            # calculate delta (percentage difference) between the best config and the others
            def calculate_delta(row, best_row):
                if best_row['throughput'] == 0:
                    return 0
                return (row['throughput'] - best_row['throughput']) / best_row['throughput'] * 100

            # Add delta column to the DataFrame
            results_df['delta'] = results_df.apply(
                lambda row: calculate_delta(row, best_configs[best_configs['range_id'] == row['range_id']].iloc[0]),
                axis=1
            )

            # Calculate the average delta for each configuration
            avg_deltas = results_df.groupby(['tp', 'dp'])['delta'].mean().reset_index()

            # Find the configuration with the minimum average delta
            best_config = avg_deltas.loc[avg_deltas['delta'].idxmin()]

            # extract all the throughput values for the best configuration
            best_config_ranges = results_df[(results_df['tp'] == best_config['tp']) & (results_df['dp'] == best_config['dp'])]
            best_config_ranges = best_config_ranges[['range_id', 'throughput']].set_index('range_id').to_dict()['throughput']

            prefill_configs[island_id] = {
                'tp': best_config['tp'],
                'dp': best_config['dp'],
                'ranges': best_config_ranges,
            }

        # for each island, find the best gpu configuration during decode
        print("*** Decode GPU Config ***")
        decode_configs = {}
        for index, (island_id, island) in enumerate(islands.items()):
            # calculate variants of tp and dp
            tp_list = []
            dp_list = []
            for tp in builtins.range(1, 8 + 1):
                if island.size % tp == 0:
                    tp_list.append(tp)
            for dp in builtins.range(1, 8 + 1):
                if island.size % dp == 0:
                    dp_list.append(dp)

            # for each range, test the configurations
            results = []
            for range_id, range in ranges.items():
                # compute the throughput for each configuration
                for tp in tp_list:
                    for dp in dp_list:
                        loop_num_gpus = island.size // dp

                        max_bs = simulator.check_max_bs(
                            self.args,
                            self.decode_gpu_list[island.gpu_type],
                            tp_num=tp,
                            seq_len=range['max'],
                            decode_len=self.decode_len_max,
                            gemm_group_per_device=math.ceil(self.args.n_routed_experts / loop_num_gpus),
                            device_num=loop_num_gpus,
                        )

                        # round max bs to the nearest multiple of 8, maximum 1024, minimum 8
                        max_bs = min(max_bs, 1024)
                        batch_size = max_bs - (max_bs % 8)
                        batch_size = max(batch_size, 8)

                        total_time_ms = 0
                        for current_decode_length in builtins.range(1, self.decode_len_avg, 1):
                            time_ms, _ = simulator.decode_time(
                                self.args,
                                self.decode_gpu_list[island.gpu_type],
                                tp_num=tp,
                                bs_num=batch_size,
                                seq_len=range['sequence_length'] + current_decode_length,
                                decode_len=self.decode_len_avg - current_decode_length,
                                gemm_group_per_device=math.ceil(self.args.n_routed_experts / loop_num_gpus),
                                device_num=loop_num_gpus,
                            )

                            # don't continue if the time is infinite
                            if time_ms == np.inf:
                                total_time_ms = np.inf
                                break
                            else:
                                total_time_ms += time_ms

                        time_s = total_time_ms / 1000
                        request_per_second = (1 / time_s * batch_size ) * dp if total_time_ms != np.inf else 0

                        # store the result
                        results.append({
                            'tp': tp,
                            'dp': dp,
                            'range_id': range_id,
                            'throughput': request_per_second,
                            'batch_size': batch_size,
                        })

            # make a df of the results
            results_df = pd.DataFrame(results)

            # calculate the percentage difference between the best and each configuration for each range
            best_configs = results_df.loc[results_df.groupby('range_id')['throughput'].idxmax()]

            # calculate delta (percentage difference) between the best config and the others
            def calculate_delta(row, best_row):
                if best_row['throughput'] == 0:
                    return 0
                return (row['throughput'] - best_row['throughput']) / best_row['throughput'] * 100
            
            # Add delta column to the DataFrame
            results_df['delta'] = results_df.apply(
                lambda row: calculate_delta(row, best_configs[best_configs['range_id'] == row['range_id']].iloc[0]),
                axis=1
            )

            # Calculate the average delta for each configuration
            avg_deltas = results_df.groupby(['tp', 'dp'])['delta'].mean().reset_index()

            # Find the configuration with the minimum average delta
            best_config = avg_deltas.loc[avg_deltas['delta'].idxmin()]

            # extract a list of rangeIDs and batch sizes for the best configuration
            best_config_ranges = results_df[(results_df['tp'] == best_config['tp']) & (results_df['dp'] == best_config['dp'])]

            # extract throughput and batch_size for the best tp
            best_config_ranges = (
                best_config_ranges
                .set_index('range_id')[['throughput', 'batch_size']]
                .to_dict(orient='index')
            )

            decode_configs[island_id] = {
                'tp': best_config['tp'],
                'dp': best_config['dp'],
                'ranges': best_config_ranges,
            }

        # print the results
        if print_debug:
            print("\n=== Ranges ===")
            for range_id, range in ranges.items():
                print(f"Range {range_id}: {range['sequence_length']} tokens, Probability: {range['probability']:.4f}, Range: {range['min']}-{range['max']} tokens")

            print("\n=== Prefill Benchmark ===")
            for island_id, config in prefill_configs.items():
                for range_id, throughput in config['ranges'].items():
                    print(f"Island {island_id}, Range {range_id}: {throughput:.4f} requests/s (TP: {config['tp']}, DP: {config['dp']})")

            print("\n=== Decode Benchmark ===")
            for island_id, config in decode_configs.items():
                for range_id, values in config['ranges'].items():
                    print(f"Island {island_id}, Range {range_id}: {values['throughput']:.4f} requests/s (TP: {config['tp']}, DP: {config['dp']}, Batch Size: {values['batch_size']})")

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
                if prefill_configs[i]['ranges'][j] == 0:
                    x_prefill[i][j].upBound = 0
                if decode_configs[i]['ranges'][j]['throughput'] == 0:
                    x_decode[i][j].upBound = 0
            
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
            problem += pulp.lpSum(decode_configs[i]['ranges'][j]['throughput'] * x_decode[i][j] for i in island_ids) == Rj_decode[j], f"Define_RjDecode_{j}"
            problem += pulp.lpSum(prefill_configs[i]['ranges'][j] * x_prefill[i][j] for i in island_ids) == Rj_prefill[j], f"Define_RjPrefill_{j}"

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
            - pulp.lpSum((lambda_dev[j] * (delta_decode[j] + delta_prefill[j] + delta_match[j])) for j in range_ids)
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
                'configs': prefill_configs,
            },
            'decode': {
                'assignment': decode_assignment,
                'throughput': decode_throughput,
                'deviation': decode_deviation,
                'configs': decode_configs,
            },
            'modes': modes,
        }

        if print_debug:
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
        evaluator.Island(gpu_type="RubinU-NVL576", size=16, id="a"),
        evaluator.Island(gpu_type="RubinU-NVL576", size=16, id="b"),
        evaluator.Island(gpu_type="RubinU-NVL576", size=16, id="c"),
        evaluator.Island(gpu_type="RubinU-NVL576", size=16, id="d"),
        evaluator.Island(gpu_type="RubinU-NVL576", size=16, id="e"),
        evaluator.Island(gpu_type="RubinU-NVL576", size=32, id="f"),
        evaluator.Island(gpu_type="RubinU-NVL576", size=64, id="g"),
        evaluator.Island(gpu_type="RubinU-NVL576", size=1024, id="h"),
        evaluator.Island(gpu_type="H200", size=16, id="m"),
        evaluator.Island(gpu_type="H200", size=16, id="n"),
        evaluator.Island(gpu_type="H200", size=16, id="o"),
        evaluator.Island(gpu_type="H200", size=16, id="p"),
        evaluator.Island(gpu_type="H20", size=16, id="q"),
        evaluator.Island(gpu_type="H20", size=16, id="r"),
        evaluator.Island(gpu_type="H20", size=16, id="s"),
        evaluator.Island(gpu_type="H20", size=16, id="t"),
        evaluator.Island(gpu_type="H800", size=16, id="u"),
        evaluator.Island(gpu_type="H800", size=16, id="v"),
        evaluator.Island(gpu_type="H800", size=16, id="w"),
        evaluator.Island(gpu_type="H800", size=16, id="x"),
        evaluator.Island(gpu_type="H800", size=16, id="y"),
        evaluator.Island(gpu_type="H800", size=16, id="z"),
        evaluator.Island(gpu_type="H800", size=16, id="aa"),
        evaluator.Island(gpu_type="H800", size=16, id="ab"),
        evaluator.Island(gpu_type="H800", size=16, id="ac"),
        evaluator.Island(gpu_type="H800", size=16, id="ad"),
        evaluator.Island(gpu_type="H800", size=16, id="ae"),
        evaluator.Island(gpu_type="H800", size=16, id="af"),
        evaluator.Island(gpu_type="H800", size=16, id="ag"),
        evaluator.Island(gpu_type="H800", size=16, id="ah"),
        evaluator.Island(gpu_type="H800", size=16, id="ai"),
        evaluator.Island(gpu_type="H800", size=16, id="aj"),
        evaluator.Island(gpu_type="H800", size=16, id="ak"),
    ]
    islands = {island.id: island for island in islands}

    # load the trace PDF
    trace_pdf = evaluator.load_trace_pdf("traces/generated_trace_pdf.csv")

    # Pack the prefill bins
    model, prefill_throughput, decode_throughput, delta, objective = packer.solve_linear(islands, trace_pdf, resolution=500, print_debug=False)

    # check if the model is not None
    if model is not None:
        print("\n=== Packer Results ===")
        print(f"Prefill Throughput: {prefill_throughput:.4f} requests/s")
        print(f"Decode Throughput: {decode_throughput:.4f} requests/s")
        print(f"Deviation: {delta:.4f}")
        print(f"Objective: {objective:.4f}")

        # add assignments to the islands
        for (island_id, range_id), value in model['prefill']['assignment'].items():
            if value == 0:
                continue
            islands[island_id].assign_prefill(
                (
                    evaluator.Bin(
                        min=model['ranges'][range_id]['min'],
                        max=model['ranges'][range_id]['max'],
                        dp=model['prefill']['configs'][island_id]['dp'],
                        tp=model['prefill']['configs'][island_id]['tp'],
                        bs=None,  # prefill does not use batch size
                    ),
                    value
                )
            )
        for (island_id, range_id), value in model['decode']['assignment'].items():
            if value == 0:
                continue
            islands[island_id].assign_decode(
                (
                    evaluator.Bin(
                        min=model['ranges'][range_id]['min'],
                        max=model['ranges'][range_id]['max'],
                        dp=model['decode']['configs'][island_id]['dp'],
                        tp=model['decode']['configs'][island_id]['tp'],
                        bs=model['decode']['configs'][island_id]['ranges'][range_id]['batch_size'],
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
            line = (
                "Island,"
                "Range,"
                "Prefill_TP,"
                "Prefill_DP,"
                "Prefill_Assignment,"
                "Prefill_Benchmark,"
                "Prefill_Assignment_Benchmark,"
                "Decode_TP,"
                "Decode_DP,"
                "Decode_Batch_Size,"
                "Decode_Assignment,"
                "Decode_Benchmark,"
                "Decode_Assignment_Benchmark,"
                "Mode\n"
            )
            f.write(line)
            for island_id in islands.keys():
                for range_id in model['ranges'].keys():
                    line = (
                        f"{island_id},"
                        f"{range_id},"
                        f"{model['prefill']['configs'][island_id]['tp']},"
                        f"{model['prefill']['configs'][island_id]['dp']},"
                        f"{model['prefill']['assignment'][(island_id, range_id)]},"
                        f"{model['prefill']['configs'][island_id]['ranges'][range_id]},"
                        f"{model['prefill']['assignment'][(island_id, range_id)] * model['prefill']['configs'][island_id]['ranges'][range_id]},"
                        f"{model['decode']['configs'][island_id]['tp']},"
                        f"{model['decode']['configs'][island_id]['dp']},"
                        f"{model['decode']['configs'][island_id]['ranges'][range_id]['batch_size']},"
                        f"{model['decode']['assignment'][(island_id, range_id)]},"
                        f"{model['decode']['configs'][island_id]['ranges'][range_id]['throughput']},"
                        f"{model['decode']['assignment'][(island_id, range_id)] * model['decode']['configs'][island_id]['ranges'][range_id]['throughput']},"
                        f"{model['modes'][(island_id, range_id)]}\n"
                    )
                    f.write(line)
        # pass to evaluator
        evaluator = evaluator.Evaluator(gpu_types)
        prefill_throughput, decode_throughput = evaluator.evaluate(islands, trace_pdf, print_debug=False)

        print("\n=== Evaluator Results ===")
        print(f"Prefill Throughput: {prefill_throughput:.4f} requests/s")
        print(f"Decode Throughput: {decode_throughput:.4f} requests/s")

    # do not print the model
    else:
        print("No solution found.")