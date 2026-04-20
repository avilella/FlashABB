import statistics
import time

from matplotlib import pyplot as plt
import pandas as pd

import torch
from torch.nn import Linear

from abodybuilder3.openfold.model.structure_module import InvariantPointAttention
from abodybuilder3.openfold.model.flashpoint_attention import FlashpointAttention
from abodybuilder3.openfold.utils.rigid_utils import Rigid

torch.set_float32_matmul_precision('medium')

device = torch.device('cuda')
# device = torch.device('cpu')

embed_dim = 128

ipa_args = {
    'c_s': embed_dim,
    'c_z': embed_dim,
    'c_hidden': 16,
    'no_heads': 12,
    'no_qk_points': 4,
    'no_v_points': 8
}

fpa_args = {arg: ipa_args[arg] for arg in ipa_args if arg != 'c_z'}

ipa = InvariantPointAttention(**ipa_args).to(device)
fpa = FlashpointAttention(**fpa_args).to(device)
linear_in_node = Linear(23, embed_dim).to(device)
linear_in_edge = Linear(132, embed_dim).to(device)


def make_batch(pairless, mult=1, num_copies=20):
    heavy = "QVQLVQSGAEVKKPGSSVKVSCKASGGTFSSLAISWVRQAPGQGLEWMGGIIPIFGTANYAQKFQGRVTITADESTSTAYMELSSLRSEDTAVYYCARGGSVSGTLVDFDIWGQGTMVTVSS"
    light = "DIQMTQSPSTLSASVGDRVTITCRASQSISSWLAWYQQKPGKAPKLLIYKASSLESGVPSRFSGSGSGTEFTLTISSLQPDDFATYYCQQYNIYPITFGGGTKVEIK"
    n = 60
    heavy_chunks = [light[i:i+n] for i in range(0, len(light), n)]
    print('\n'.join(heavy_chunks))
    raise OSError
    if pairless:
        from abodybuilder3.no_pair_utils import string_to_input
    else:
        from abodybuilder3.utils import string_to_input
    heavy = heavy * mult
    light = light * mult
    ab_input_batch = {}
    for i in range(num_copies):
        ab_input = string_to_input(heavy=heavy, light=light)
        ab_input_dict = {
            key: (value.unsqueeze(0).to(device) if key not in ["single", "pair"] else value.to(device))
            for key, value in ab_input.items()
        }
        if len(ab_input_batch) == 0:
            ab_input_batch = ab_input_dict
        else:
            for key in ab_input_dict:
                ab_input_batch[key] = torch.cat((ab_input_batch[key], ab_input_dict[key]), 0)
    return ab_input_batch


def run_test(pairless, mult=1, num_copies=20):
    torch.cuda.reset_peak_memory_stats()
    ab_input_batch = make_batch(pairless, mult, num_copies)
    s = ab_input_batch['single']
    s = linear_in_node(s)
    if not pairless:
        z = ab_input_batch['pair']
        z = linear_in_edge(z)
    res_idx = ab_input_batch['residue_index']
    mask = s.new_ones(s.shape[:-1])
    rigids = Rigid.identity(
        s.shape[:-1],
        s.dtype,
        s.device,
        # self.training,
        False,
        fmt="quat",
    )

    times = []
    with torch.no_grad():
        for i in range(100):
            start_t = time.time() * 1000
            if pairless:
                out = fpa(s, res_idx, rigids, mask)
            else:
                out = ipa(s, z, rigids, mask)
            times.append(time.time() * 1000 - start_t)
    t = statistics.median(times)
    utilize_gpu = float(torch.cuda.max_memory_allocated(device=device)) / 1024.**3
    return t, utilize_gpu


def run_test_back(pairless, mult=1, num_copies=20):
    torch.cuda.reset_peak_memory_stats()
    ab_input_batch = make_batch(pairless, mult, num_copies)
    s = ab_input_batch['single']
    s = linear_in_node(s)
    if not pairless:
        z = ab_input_batch['pair']
        z = linear_in_edge(z)
    res_idx = ab_input_batch['residue_index']
    mask = s.new_ones(s.shape[:-1])
    rigids = Rigid.identity(
        s.shape[:-1],
        s.dtype,
        s.device,
        # self.training,
        True,
        fmt="quat",
    )

    times = []
    if pairless:
        out = fpa(s, res_idx, rigids, mask)
    else:
        out = ipa(s, z, rigids, mask)
    loss = out.sum()
    for i in range(100):
        start_t = time.time() * 1000
        loss.backward(retain_graph=True)
        times.append(time.time() * 1000 - start_t)
    t = statistics.median(times)
    utilize_gpu = float(torch.cuda.max_memory_allocated(device=device)) / 1024.**3
    # utilize_gpu = 0
    return t, utilize_gpu


def seq_len_test(pairless, max_mult=8, num_copies=1):
    heavy = "QVQLVQSGAEVKKPGSSVKVSCKASGGTFSSLAISWVRQAPGQGLEWMGGIIPIFGTANYAQKFQGRVTITADESTSTAYMELSSLRSEDTAVYYCARGGSVSGTLVDFDIWGQGTMVTVSS"
    light = "DIQMTQSPSTLSASVGDRVTITCRASQSISSWLAWYQQKPGKAPKLLIYKASSLESGVPSRFSGSGSGTEFTLTISSLQPDDFATYYCQQYNIYPITFGGGTKVEIK"
    suffix = 'pairless' if pairless else 'paired'
    times = []
    mems = []
    tokens = []
    for mult in range(1,max_mult+1):
        t, mem = run_test(pairless, mult=mult, num_copies=num_copies)
        times.append(t)
        mems.append(mem)
        tokens.append((len(heavy) + len(light))*mult)
    df = pd.DataFrame({'tokens': tokens, 'time': times, 'mem': mems})
    df.to_csv(f'speed_{suffix}_mult-{max_mult}_bsz-{num_copies}.csv', index=False)


def backward_seq_len_test(pairless, max_mult=8, num_copies=1):
    heavy = "QVQLVQSGAEVKKPGSSVKVSCKASGGTFSSLAISWVRQAPGQGLEWMGGIIPIFGTANYAQKFQGRVTITADESTSTAYMELSSLRSEDTAVYYCARGGSVSGTLVDFDIWGQGTMVTVSS"
    light = "DIQMTQSPSTLSASVGDRVTITCRASQSISSWLAWYQQKPGKAPKLLIYKASSLESGVPSRFSGSGSGTEFTLTISSLQPDDFATYYCQQYNIYPITFGGGTKVEIK"
    suffix = 'pairless' if pairless else 'paired'
    times = []
    mems = []
    tokens = []
    for mult in range(1,max_mult+1):
        t, mem = run_test_back(pairless, mult=mult, num_copies=num_copies)
        times.append(t)
        mems.append(mem)
        tokens.append((len(heavy) + len(light))*mult)
    df = pd.DataFrame({'tokens': tokens, 'time': times, 'mem': mems})
    df.to_csv(f'backward_speed_{suffix}_mult-{max_mult}_bsz-{num_copies}.csv', index=False)


def add_labels(metric):
    if metric == 'mem':
        plt.ylabel('Memory (GB)', fontsize=16)
    elif metric == 'time':
        plt.ylabel('Time (ms)', fontsize=16)


def plot_seq_len(metric='mem'):
    plt.clf()
    df_paired = pd.read_csv('speed_paired_mult-8_bsz-1.csv')
    df_pairless = pd.read_csv('speed_pairless_mult-8_bsz-1.csv')

    plt.plot(df_paired['tokens'], df_paired[metric], label='Invariant Point Attention')
    plt.plot(df_pairless['tokens'], df_pairless[metric], label='Flashpoint Attention')
    add_labels(metric)
    plt.xlabel('Number of tokens', fontsize=16)
    plt.legend(fontsize=16)
    plt.savefig(f'scaling_{metric}.pdf')


def plot_typical(metric='mem'):
    plt.clf()
    df_paired = pd.read_csv('speed_paired_mult-1_bsz-32.csv')
    df_pairless = pd.read_csv('speed_pairless_mult-1_bsz-32.csv')
    df_paired_back = pd.read_csv('backward_speed_paired_mult-1_bsz-32.csv')
    df_pairless_back = pd.read_csv('backward_speed_pairless_mult-1_bsz-32.csv')

    bar_container = plt.bar(
        [
            'IPA forward',
            'FPA forward',
            'IPA backward',
            'FPA backward',
        ],
        [
            df_paired[metric][0],
            df_pairless[metric][0],
            df_paired_back[metric][0],
            df_pairless_back[metric][0],
        ],
        color=[
            'tab:blue',
            'tab:orange',
            'tab:blue',
            'tab:orange',
        ]
    )
    add_labels(metric)
    plt.xticks(fontsize=12)
    plt.bar_label(bar_container, fmt='{:,.2f}')
    # plt.xlabel('Method')
    plt.savefig(f'typical_{metric}.pdf')


if __name__ == '__main__':
    # # with torch.backends.cuda.sdp_kernel(enable_math=False):
    # from torch.nn.attention import SDPBackend, sdpa_kernel
    # # Only enable flash attention backend
    # # with torch.nn.attention.sdpa_kernel(enable_math=False):
    # # with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    #     # scaled_dot_product_attention(...)
    seq_len_test(True, max_mult=1, num_copies=32)
    seq_len_test(False, max_mult=1, num_copies=32)
    backward_seq_len_test(True, max_mult=1, num_copies=32)
    backward_seq_len_test(False, max_mult=1, num_copies=32)
    seq_len_test(True, max_mult=8, num_copies=1)
    seq_len_test(False, max_mult=8, num_copies=1)
    plot_seq_len('mem')
    plot_seq_len('time')
    plot_typical('mem')
    plot_typical('time')
