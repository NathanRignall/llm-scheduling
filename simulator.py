import pandas as pd
import math
from functools import reduce

NVL_GPU_LIST = [72, 144, 576]

class ModelArgs:
    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    vocab_size: int = 129280
    dim: int = 7168
    inter_dim: int = 18432
    moe_inter_dim: int = 2048
    n_layers: int = 61
    n_dense_layers: int = 3
    n_heads: int = 128
    # moe
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    n_activated_experts: int = 8
    n_expert_groups: int = 8
    n_limited_groups: int = 4
    route_scale: float = 2.5
    # mla
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.

class GPU_perf:
    def __init__(self, gpu_type, sm, comm_sm, gpu_per_node, fp16_flops, fp8_flops, fp4_flops, mem, mem_bw, nvlink_bw, pcie_bw, discount_rate, price):
        self.gpu_type = gpu_type
        self.sm = sm
        self.gpu_per_node = gpu_per_node
        self.comm_sm = comm_sm
        self.fp16_flops = fp16_flops
        self.fp8_flops = fp8_flops
        self.fp4_flops = fp4_flops
        self.mem = mem
        self.mem_bw = mem_bw
        self.nvlink_bw = nvlink_bw
        self.pcie_bw = pcie_bw
        self.price = price
        self.discount_rate = discount_rate

    def get_fp16_flops(self):
        return self.fp16_flops * self.discount_rate * (self.sm - self.comm_sm) / self.sm

    def get_fp8_flops(self):
        return self.fp8_flops * self.discount_rate * (self.sm - self.comm_sm) / self.sm

    def get_fp4_flops(self):
        return self.fp4_flops * self.discount_rate * (self.sm - self.comm_sm) / self.sm

    def get_mem_bw(self):
        return self.mem_bw * self.discount_rate

    def get_nvlink_bw(self):
        return self.nvlink_bw * self.discount_rate

    def get_pcie_bw(self):
        return self.pcie_bw * self.discount_rate

    def get_price(self):
        return self.price

def get_gpu_info(filename='./device/gpuinfo.csv', discount_rate=0.85, device_list=[], decoding_mode=False):
    gpu_dict = {}
    df = pd.read_csv(filename)
    if decoding_mode:
        df['comm_sm'] = 0
    for _, c in df.iterrows():
        key = c['gpu_type']
        gpu = GPU_perf(
            gpu_type=c['gpu_type'],
            sm=c['sm'], comm_sm=c['comm_sm'],
            fp16_flops=c['fp16'],
            fp8_flops=c['fp8'],
            fp4_flops=c['fp4'],
            mem=c['mem'],
            mem_bw=c['mem_bw'],
            nvlink_bw=c['nvlink_bw'],
            pcie_bw=c['pcie_bw'],
            gpu_per_node=c['gpu_per_node'],
            price=c['price'],
            discount_rate=discount_rate)
        if (len(device_list) == 0) | (key in device_list):
            gpu_dict[key] = gpu
    return gpu_dict

def _prefill_moe(args: ModelArgs, gpu: GPU_perf, seq_len, tp_num, dp_num):
    load_time = moe_expert_mem(args) / gpu.get_mem_bw()
    gemm_flops = gpu.get_fp4_flops() if gpu.get_fp4_flops() != 0 else gpu.get_fp8_flops()
    num_device = tp_num * dp_num
    num_shared_token = dp_num * seq_len / num_device
    shared_flops = moe_expert_flops(args, num_shared_token)
    shared_time = shared_flops / gemm_flops + load_time

    num_routed_token = seq_len * dp_num * args.n_activated_experts / num_device
    routed_flops = moe_expert_flops(args, num_routed_token)
    expert_num = math.ceil(args.n_routed_experts) / dp_num / tp_num
    routed_time = routed_flops / gemm_flops + load_time * expert_num

    return shared_time, routed_time

def _prefill_alltoall(args: ModelArgs, gpu, seq_len, tp_num, static_latency=0.05):
    if gpu.gpu_per_node == 8:
        dp = gpu.gpu_per_node / tp_num
        dispatch_node = 4
        dispatch_size = (dispatch_node - 1) * dp * seq_len * \
            args.n_activated_experts / gpu.gpu_per_node * args.dim / 1024/1024
        comm_bw = gpu.get_pcie_bw() * gpu.gpu_per_node
    else:
        # NVL72
        expert_num = math.ceil(args.n_routed_experts / gpu.gpu_per_node)
        dispatch_prob = (args.n_routed_experts - expert_num) / \
            args.n_routed_experts
        dispatch_size = dispatch_prob * args.n_activated_experts * seq_len / tp_num * args.dim / 1024/1024
        comm_bw = gpu.get_nvlink_bw()

    combine_size = 2 * dispatch_size  # fp16
    if gpu.get_fp4_flops != 0:
        dispatch_size = dispatch_size / 2
    dispatch_time = dispatch_size / comm_bw + static_latency
    combine_time = combine_size / comm_bw + static_latency
    return dispatch_time, combine_time

def _prefill_time(args: ModelArgs, gpu, seq_len, kv_cache_rate, tp_num, dp_num):
    dense_mla, tp_mla = mla_elapse_time(args, gpu, seq_len, kv_cache_rate, tp_num=tp_num, decoding_mode=False, enable_gemm_fp4=True)
    dense_mlp = _prefill_dense_mlp(args, gpu, seq_len)
    shared, routed = _prefill_moe(args, gpu, seq_len, tp_num, dp_num)
    dispatch, combine = _prefill_alltoall(args, gpu, seq_len, tp_num)
    
    return dense_mla, dense_mlp, tp_mla[tp_num], shared, combine, routed, dispatch

def prefill_time(args: ModelArgs, gpu, seq_len, kv_cache_rate, tp_num, dp_num):
    n_sparse_layers = args.n_layers - args.n_dense_layers

    dense_mla, dense_mlp, tp_mla, shared, combine, routed, dispatch = _prefill_time(args, gpu, seq_len, kv_cache_rate, tp_num, dp_num)
    overlap1 = combine - (tp_mla + shared)
    overlap2 = dispatch - routed

    comp_time = args.n_dense_layers * (dense_mla + dense_mlp) + n_sparse_layers * (tp_mla + shared + routed)
    comm_time = n_sparse_layers * (combine + dispatch)
    sum_time = comp_time
    if overlap1 > 0:
        sum_time += overlap1 * n_sparse_layers
    if overlap2 > 0:
        sum_time += overlap2 * n_sparse_layers

    return sum_time

# decode

def _decoding_batchsize(args: ModelArgs, gpu: GPU_perf, seq_len, decode_len, tp_num, expert_num):
    mem_util_rate = 0.9  # torch/activation等其它开销的折扣
    mla = 187.17  # MLA的参数(单位M)
    expert_mem = 44.05  # expert的参数(单位M)
    others_parameter = 2.91  # 其它参数2.91GB
    kv_cache = (seq_len+decode_len) * (args.kv_lora_rank + args.qk_rope_head_dim) * args.n_layers * tp_num
    mem = gpu.mem * mem_util_rate - others_parameter - mla * args.n_layers / tp_num /1024
    mem -= expert_mem * (args.n_layers - args.n_dense_layers) * expert_num / 1024
    return mem * 1024 * 1024 * 1024 / kv_cache

def mla_flops(q_len, kv_len, args: ModelArgs, kv_cache_rate):
    # calculate MACs and estimate Flops approx. 2xMAC.
    q_down_proj = q_len * args.dim * args.q_lora_rank  # wq_a
    q_up_proj = q_len * args.q_lora_rank * args.n_heads * \
        (args.qk_nope_head_dim + args.qk_rope_head_dim)  # wq_b
    kv_down_proj = kv_len * args.dim * \
        (args.kv_lora_rank + args.qk_rope_head_dim)  # wkv_a
    k_up_proj = kv_len * args.kv_lora_rank * \
        args.n_heads * args.qk_nope_head_dim  # w_uk
    v_up_proj = kv_len * args.kv_lora_rank * args.n_heads * args.v_head_dim  # w_uv

    kv_down_proj = kv_down_proj * (1 - kv_cache_rate)
    gemm_sum = q_down_proj + q_up_proj + kv_down_proj + k_up_proj + v_up_proj

    # 把它看成一个标准的args.n_heads的MHA
    mha = args.n_heads * (q_len * args.qk_rope_head_dim * kv_len  # QK_score_rope
                          + q_len * args.qk_nope_head_dim * kv_len  # QK_score_nope
                          + q_len * kv_len * args.v_head_dim)  # ScoreV
    wo = q_len * args.n_heads * args.v_head_dim * args.dim  # wo
    attn_sum = mha + wo
    # return flops by 2* Sum(MACs)
    GEMM_FP8_FLOPS = gemm_sum * 2/1e9
    ATTN_FP16_FLOPS = attn_sum * 2/1e9

    return GEMM_FP8_FLOPS+ATTN_FP16_FLOPS, GEMM_FP8_FLOPS, ATTN_FP16_FLOPS

def mla_matabsob_flops(q_len, kv_len, args: ModelArgs, kv_cache_rate=0):
    # calculate MACs and estimate Flops approx. 2xMAC.
    q_down_proj = q_len * args.dim * args.q_lora_rank  # wq_a
    q_rope_up_proj = q_len * args.q_lora_rank * \
        args.n_heads * args.qk_rope_head_dim  # wq_b_rope
    q_absorb = q_len * args.n_heads * (args.q_lora_rank * args.qk_nope_head_dim  # wq_b
                                       + args.qk_nope_head_dim * args.kv_lora_rank)  # w_uk

    kv_down_proj = kv_len * args.dim * \
        (args.kv_lora_rank + args.qk_rope_head_dim)  # wkv_a
    kv_down_proj = kv_down_proj * (1 - kv_cache_rate)  # KV-Cache命中率修正
    gemm_sum = q_down_proj + q_rope_up_proj + q_absorb + kv_down_proj

    # 把它看成一个标准的args.n_heads的MQA
    mqa = args.n_heads * (q_len * args.qk_rope_head_dim * kv_len  # Score_rope
                          + q_len * args.kv_lora_rank * kv_len  # Score_nope
                          + q_len * kv_len * args.kv_lora_rank)  # Score V

    attn_up_proj = q_len * args.n_heads * args.v_head_dim * args.kv_lora_rank
    o_proj = q_len * args.n_heads * args.v_head_dim * args.dim
    attn_sum = mqa + attn_up_proj + o_proj

    # return flops by 2* Sum(MACs)
    gemm_sum = gemm_sum * 2/1e9
    attn_sum = attn_sum * 2/1e9

    return gemm_sum + attn_sum, gemm_sum, attn_sum

def mla_mem(args: ModelArgs):
    q_down_proj = args.dim * args.q_lora_rank  # wq_a
    q_up_proj = args.q_lora_rank * args.n_heads * \
        (args.qk_nope_head_dim + args.qk_rope_head_dim)  # wq_b
    kv_down_proj = args.dim * \
        (args.kv_lora_rank + args.qk_rope_head_dim)  # wkv_a
    k_up_proj = args.kv_lora_rank * args.n_heads * args.qk_nope_head_dim  # w_uk
    v_up_proj = args.kv_lora_rank * args.n_heads * args.v_head_dim  # w_uv
    wo = args.n_heads * args.v_head_dim * args.dim  # wo
    return (q_down_proj + q_up_proj + k_up_proj + kv_down_proj + v_up_proj + wo)/1024/1024

def mla_elapse_time(args: ModelArgs, gpu: GPU_perf, seq_len, kv_cache_rate, tp_num, decoding_mode=True, batchsize=1, enable_gemm_fp4=True, min_ar_time=0.015, mla_discount=0.7, mla_kernel_static_time=0.05):
    if decoding_mode:
        # decode
        _, gemm_flops, attn_fp16_flops = mla_matabsob_flops(1, seq_len, args, 1)
        gemm_flops *= batchsize
        attn_fp16_flops *= batchsize
    else:
        # prefill
        _, gemm_flops, attn_fp16_flops = mla_flops(seq_len, seq_len, args, kv_cache_rate)

    gemm_fp8_t = gemm_flops / gpu.get_fp8_flops() / mla_discount
    attn_fp16_t = attn_fp16_flops / gpu.get_fp16_flops() / mla_discount

    # load weight
    load_t = mla_mem(args) / gpu.get_mem_bw()

    total = gemm_fp8_t + attn_fp16_t + load_t

    if enable_gemm_fp4:
        if gpu.get_fp4_flops() == 0:
            None
        else:
            gemm_fp4_t = gemm_flops / gpu.get_fp4_flops()
            total = gemm_fp4_t + attn_fp16_t

    ar_len = batchsize if decoding_mode else seq_len
    all_reduce_comm_size = ar_len * args.dim * 2 / 1024 / 1024  # fp16 take 2Bytes
    all_reduce_t = all_reduce_comm_size / gpu.get_nvlink_bw() + min_ar_time

    tp_time = {}
    if tp_num == 1:
        tp_time[tp_num] = total + mla_kernel_static_time
    else:
        tp_time[tp_num] = total / tp_num + all_reduce_t + mla_kernel_static_time

    return total, tp_time

# Prefill

def prefill_mla(args: ModelArgs, gpu_dict, seq_len, kv_cache_rate):
    df = pd.DataFrame(columns=['GPU', 'TP1', 'TP4', 'TP8'])
    for key in gpu_dict.keys():
        tp1, tp_list = mla_elapse_time(args, gpu_dict[key], seq_len, kv_cache_rate, tp=[4, 8], decoding_mode=False, enable_gemm_fp4=True)
        df.loc[len(df)] = [gpu_dict[key].gpu_type, tp1] + list(tp_list.values())
    return df

def densmlp_flops(args: ModelArgs, seq_len):
    return 3 * seq_len * args.dim * args.inter_dim * 2/1e9

def densmlp_mem(args: ModelArgs):
    return 3 * args.dim * args.inter_dim / 1024/1024

def _prefill_dense_mlp(args: ModelArgs, gpu: GPU_perf, seq_len):
    gemm_flops = densmlp_flops(args, seq_len)
    if gpu.get_fp4_flops() != 0:
        gemm_time = gemm_flops / gpu.get_fp4_flops()
    else:
        gemm_time = gemm_flops / gpu.get_fp8_flops()

    load_time = densmlp_mem(args) / gpu.get_mem_bw()
    gemm_time = gemm_time + load_time
    return gemm_time

def moe_expert_flops(args: ModelArgs, seq_len):
    return 3 * seq_len * args.dim * args.moe_inter_dim * 2/1e9

def moe_expert_mem(args: ModelArgs):
    return 3 * args.dim * args.moe_inter_dim / 1024 / 1024

def decode_mla(args: ModelArgs, gpu, tp_num, bs_num, seq_len, decode_len, expert_num=2):
    kv_cache = seq_len * (args.kv_lora_rank + args.qk_rope_head_dim) * bs_num
    load_kv_time = kv_cache / 1024 / 1024 / 1024 / gpu.get_mem_bw() * 1000
    dense_mla, sparse_mla = mla_elapse_time(args, gpu, seq_len, kv_cache_rate=1, tp_num=tp_num, batchsize=bs_num, decoding_mode=True, enable_gemm_fp4=True)
    return load_kv_time, dense_mla, sparse_mla[tp_num]

def decode_dense_mlp(args: ModelArgs, gpu, tp_num, bs_num, seq_len, decode_len, expert_num=2):
    gemm_time = _prefill_dense_mlp(args, gpu, bs_num)
    return gemm_time

def n_pow2_range(n:int):
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n = n+1
    return n

def _decode_moe_expert(args: ModelArgs, gpu: GPU_perf, bs, 
                       gemm_group_per_device, device_num):
    load_time = moe_expert_mem(args) / gpu.get_mem_bw()
    if gpu.get_fp4_flops() != 0:
        load_time = load_time /2
    gpu_flops = gpu.get_fp4_flops() if gpu.get_fp4_flops() != 0 else gpu.get_fp8_flops()
    
    total_expert = gemm_group_per_device * device_num
    m_per_group = bs * args.n_activated_experts * device_num / total_expert

    #data from hs's profiling result
    flops_discounts = {
        1: 0.05,
        2: 0.05,
        4: 0.05,
        8: 0.05,
        16: 0.08,
        32: 0.1,
        64: 0.2,
        128: 0.35,
        256: 0.4,
        512: 0.6,
        1024: 0.7,
        2048: 0.7,
        4096: 0.7,
        8192: 0.7,
        16384: 0.7,
        32768: 0.7,
        65536: 0.7
    }

    # H20 exception based on hs's result
    if gpu.gpu_type.find('H20')!= -1 :
        flops_discounts = {
        1: 0.06,
        2: 0.06,
        4: 0.06,
        8: 0.12,
        16: 0.25,
        32: 0.45,
        64: 0.8,
        128: 0.9,
        256: 1.0,
        512: 1.0,
        1024: 1.0,
        2048: 1.0,
        4096: 1.0,
        8192: 1.0,
        16384: 1.0,
        32768: 1.0,
        65536: 1.0
    }

    gpu_flops = gpu_flops * flops_discounts[n_pow2_range(int(m_per_group))]
    
    shared_flops = moe_expert_flops(args, bs)
    shared_time = shared_flops / gpu_flops + load_time

    num_routed_token = bs * args.n_activated_experts
    routed_flops = moe_expert_flops(args, num_routed_token)
    routed_time = routed_flops / gpu_flops + load_time * gemm_group_per_device
    return shared_time, routed_time

def decode_moe_expert(args: ModelArgs, gpu, tp_num, bs_num, seq_len, decode_len, gemm_group_per_device, device_num, mbs=2):
    shared_time, routed_time = _decode_moe_expert(args, gpu, bs_num / mbs, gemm_group_per_device=gemm_group_per_device, device_num=device_num)
    shared_time *= mbs
    routed_time *= mbs
    return shared_time, routed_time

def _moe_a2a(args: ModelArgs, gpu: GPU_perf, bs, expert_num, device_num, fp8_combine=False, static_latency=0.005, mbs=2):
    dispatch_size = bs * args.dim * args.n_activated_experts / 1024/1024
    if fp8_combine & (gpu.get_fp4_flops() != 0):
        combine_size = dispatch_size
    else:
        combine_size = dispatch_size * 2  # FP16
    if gpu.gpu_per_node == 8:
        comm_bw = gpu.get_pcie_bw()
        # single host deployment
        if args.n_routed_experts / (expert_num - 1) == gpu.gpu_per_node:
            comm_bw = gpu.get_nvlink_bw()
    #NVL72 /144 / 576
    elif (gpu.gpu_per_node in NVL_GPU_LIST) & (device_num >  gpu.gpu_per_node):
            comm_bw = gpu.get_pcie_bw()
    else:
        comm_bw = gpu.get_nvlink_bw()

    dispatch_time = dispatch_size / comm_bw + static_latency * mbs
    combine_time = combine_size / comm_bw + static_latency * mbs
    return dispatch_time, combine_time

def decode_time(args: ModelArgs, gpu, dp_num, tp_num, bs_num, seq_len, decode_len, gemm_group_per_device, device_num, tps_limit=0, mbs=2, fp8_combine=False, print_flag=False):
    expert_per_device = gemm_group_per_device + 1  # add shared expert

    max_bs = _decoding_batchsize(args, gpu, seq_len, decode_len, expert_num=expert_per_device, tp_num=tp_num)

    if bs_num > max_bs:
        print(f"Batch size {bs_num} exceeds the limit {max_bs}.")
        return

    load_kv_time, dense_mla, sparse_mla = decode_mla(args, gpu, tp_num, bs_num, seq_len, decode_len, expert_num=expert_per_device)
    dense_mlp = _prefill_dense_mlp(args, gpu, bs_num)
    shared_expert, routed_expert = decode_moe_expert(args, gpu, tp_num, bs_num, seq_len, decode_len, gemm_group_per_device=gemm_group_per_device, device_num=device_num, mbs=mbs)
    dispatch, combine = _moe_a2a(args, gpu, bs_num, expert_num=expert_per_device, device_num=device_num, fp8_combine=fp8_combine, mbs=mbs)

    # compute times
    dense_mla = dense_mla + load_kv_time
    sparse_mla = sparse_mla + load_kv_time
    comp_sum = sparse_mla + shared_expert + routed_expert
    comm_sum = dispatch + combine
    delta = comm_sum - sparse_mla - shared_expert

    tpot_o = (dense_mla + dense_mlp) * args.n_dense_layers + \
        (sparse_mla + shared_expert + routed_expert) * (args.n_layers - args.n_dense_layers) / dp_num
    tpot = tpot_o if delta < 0 else tpot_o + delta * (args.n_layers - args.n_dense_layers) / dp_num

    tps = 1000 / tpot
    tps_o = 1000 / tpot_o

    total = tps * bs_num
    total_o = tps_o * bs_num
    comm_impact = (total_o - total) / total_o

    if print_flag:
        print(f"DenseMLA: {round(dense_mla, 6)}ms, DenseMLP: {round(dense_mlp, 6)}ms, SparseMLA: {round(sparse_mla, 6)}ms, SharedExpert: {round(shared_expert, 6)}ms, RoutedExpert: {round(routed_expert, 6)}ms, Dispatch: {round(dispatch, 6)}ms, Combine: {round(combine, 6)}ms")
        print(f"CompSum: {round(comp_sum, 6)}ms, CommSum: {round(comm_sum, 6)}ms, Delta: {round(delta, 6)}ms")
        print(f"tpot: {round(tpot, 6)}ms, tpot_o: {round(tpot_o, 6)}ms")
        print(f"tps: {round(tps, 6)}ms, tps_o: {round(tps_o, 6)}ms")
        print(f"CommImpact: {round(comm_impact, 6)}ms")

    return total

# Example usage
if __name__ == "__main__":
    args = ModelArgs()
    gpu_dict = get_gpu_info('./device/gpu_info.csv', decoding_mode=True, discount_rate=0.85)
    
    dp_num = 2
    tp_num = 4
    bs_num = 8

    seq_len = 4096
    decode_len = 512
    
    gemm_group_per_device = 2
    device_num = 8

    # Example usage of decoding time
    total_time = decode_time(args, gpu_dict['H200'], dp_num, tp_num, bs_num, seq_len, decode_len, gemm_group_per_device, device_num)
    print(f"Total decoding time: {total_time} ms")