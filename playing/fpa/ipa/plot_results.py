"""Plot speed comparison results between IPA and FPA."""
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import os

# Consistent styling
COLORS = {
    'ipa': 'lightblue',
    'fpa': 'lightsalmon',
}
FONTSIZE_LABEL = 14
FONTSIZE_TITLE = 16
FONTSIZE_TICK = 12
FONTSIZE_LEGEND = 11
DPI = 150


def plot_seq_len(metric='mem', data_dir='.'):
    """Plot scaling with sequence length for IPA vs FPA."""
    fig, ax = plt.subplots(figsize=(10, 6))

    paired_file = os.path.join(data_dir, 'speed_paired_mult-8_bsz-1.csv')
    pairless_file = os.path.join(data_dir, 'speed_pairless_mult-8_bsz-1.csv')

    if not os.path.exists(paired_file) or not os.path.exists(pairless_file):
        print(f"Missing data files in {data_dir}")
        return

    df_paired = pd.read_csv(paired_file)
    df_pairless = pd.read_csv(pairless_file)

    ax.plot(df_paired['tokens'], df_paired[metric],
            label='Invariant Point Attention', color=COLORS['ipa'],
            linewidth=2, marker='o', markersize=6)
    ax.plot(df_pairless['tokens'], df_pairless[metric],
            label='Flashpoint Attention', color=COLORS['fpa'],
            linewidth=2, marker='s', markersize=6)

    if metric == 'mem':
        ax.set_ylabel('Memory (GB)', fontsize=FONTSIZE_LABEL)
    elif metric == 'time':
        ax.set_ylabel('Time (ms)', fontsize=FONTSIZE_LABEL)

    ax.set_xlabel('Number of tokens', fontsize=FONTSIZE_LABEL)
    ax.set_title(f'Sequence Length Scaling ({metric.capitalize()})', fontsize=FONTSIZE_TITLE)
    ax.tick_params(axis='x', labelsize=FONTSIZE_TICK)
    ax.tick_params(axis='y', labelsize=FONTSIZE_TICK)
    ax.legend(fontsize=FONTSIZE_LEGEND)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, f'scaling_{metric}.pdf'), dpi=DPI)
    plt.savefig(os.path.join(data_dir, f'scaling_{metric}.png'), dpi=DPI)
    print(f"Saved scaling_{metric}.pdf and scaling_{metric}.png")
    plt.close()


def plot_typical(metric='mem', data_dir='.'):
    """Plot bar comparison for typical antibody (batch of 32)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    files = {
        'paired': os.path.join(data_dir, 'speed_paired_mult-1_bsz-32.csv'),
        'pairless': os.path.join(data_dir, 'speed_pairless_mult-1_bsz-32.csv'),
        'paired_back': os.path.join(data_dir, 'backward_speed_paired_mult-1_bsz-32.csv'),
        'pairless_back': os.path.join(data_dir, 'backward_speed_pairless_mult-1_bsz-32.csv'),
    }

    for name, path in files.items():
        if not os.path.exists(path):
            print(f"Missing {path}")
            return

    df_paired = pd.read_csv(files['paired'])
    df_pairless = pd.read_csv(files['pairless'])
    df_paired_back = pd.read_csv(files['paired_back'])
    df_pairless_back = pd.read_csv(files['pairless_back'])

    names = ['IPA\nForward', 'FPA\nForward', 'IPA\nBackward', 'FPA\nBackward']
    values = [
        df_paired[metric].iloc[0],
        df_pairless[metric].iloc[0],
        df_paired_back[metric].iloc[0],
        df_pairless_back[metric].iloc[0],
    ]
    colors = [COLORS['ipa'], COLORS['fpa'], COLORS['ipa'], COLORS['fpa']]

    bars = ax.bar(names, values, color=colors, edgecolor='black', linewidth=1)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        if metric == 'time':
            label = f'{value:.1f}ms'
        else:
            label = f'{value:.2f}GB'
        ax.text(bar.get_x() + bar.get_width()/2, height * 1.02,
                label, ha='center', va='bottom', fontsize=FONTSIZE_TICK)

    if metric == 'mem':
        ax.set_ylabel('Memory (GB)', fontsize=FONTSIZE_LABEL)
    elif metric == 'time':
        ax.set_ylabel('Time (ms)', fontsize=FONTSIZE_LABEL)

    ax.set_title(f'Typical Antibody Batch (32 sequences)', fontsize=FONTSIZE_TITLE)
    ax.tick_params(axis='x', labelsize=FONTSIZE_TICK)
    ax.tick_params(axis='y', labelsize=FONTSIZE_TICK)

    # Add legend
    legend_elements = [
        Patch(facecolor=COLORS['ipa'], edgecolor='black', label='Invariant Point Attention'),
        Patch(facecolor=COLORS['fpa'], edgecolor='black', label='Flashpoint Attention'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=FONTSIZE_LEGEND)

    # Add some headroom for labels
    ax.set_ylim(top=ax.get_ylim()[1] * 1.15)

    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, f'typical_{metric}.pdf'), dpi=DPI)
    plt.savefig(os.path.join(data_dir, f'typical_{metric}.png'), dpi=DPI)
    print(f"Saved typical_{metric}.pdf and typical_{metric}.png")
    plt.close()


def plot_combined(data_dir='.'):
    """Combined figure with scaling (time, memory) and speedup."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- Panel 1: Time scaling ---
    ax = axes[0]
    paired_file = os.path.join(data_dir, 'speed_paired_mult-8_bsz-1.csv')
    pairless_file = os.path.join(data_dir, 'speed_pairless_mult-8_bsz-1.csv')

    if os.path.exists(paired_file) and os.path.exists(pairless_file):
        df_paired = pd.read_csv(paired_file)
        df_pairless = pd.read_csv(pairless_file)

        ax.plot(df_paired['tokens'], df_paired['time'],
                label='Invariant Point Attention', color=COLORS['ipa'],
                linewidth=2, marker='o', markersize=6)
        ax.plot(df_pairless['tokens'], df_pairless['time'],
                label='Flashpoint Attention', color=COLORS['fpa'],
                linewidth=2, marker='s', markersize=6)

        ax.set_ylabel('Time (ms)', fontsize=FONTSIZE_LABEL)
        ax.set_xlabel('Number of tokens', fontsize=FONTSIZE_LABEL)
        ax.set_title('Time Scaling', fontsize=FONTSIZE_TITLE)
        ax.tick_params(axis='x', labelsize=FONTSIZE_TICK)
        ax.tick_params(axis='y', labelsize=FONTSIZE_TICK)
        ax.legend(fontsize=FONTSIZE_LEGEND - 1)
        ax.grid(True, alpha=0.3)

    # --- Panel 2: Memory scaling ---
    ax = axes[1]
    if os.path.exists(paired_file) and os.path.exists(pairless_file):
        ax.plot(df_paired['tokens'], df_paired['mem'],
                label='Invariant Point Attention', color=COLORS['ipa'],
                linewidth=2, marker='o', markersize=6)
        ax.plot(df_pairless['tokens'], df_pairless['mem'],
                label='Flashpoint Attention', color=COLORS['fpa'],
                linewidth=2, marker='s', markersize=6)

        ax.set_ylabel('Memory (GB)', fontsize=FONTSIZE_LABEL)
        ax.set_xlabel('Number of tokens', fontsize=FONTSIZE_LABEL)
        ax.set_title('Memory Scaling', fontsize=FONTSIZE_TITLE)
        ax.tick_params(axis='x', labelsize=FONTSIZE_TICK)
        ax.tick_params(axis='y', labelsize=FONTSIZE_TICK)
        ax.legend(fontsize=FONTSIZE_LEGEND - 1)
        ax.grid(True, alpha=0.3)

    # --- Panel 3: Speedup ---
    ax = axes[2]
    files = {
        'paired': os.path.join(data_dir, 'speed_paired_mult-1_bsz-32.csv'),
        'pairless': os.path.join(data_dir, 'speed_pairless_mult-1_bsz-32.csv'),
        'paired_back': os.path.join(data_dir, 'backward_speed_paired_mult-1_bsz-32.csv'),
        'pairless_back': os.path.join(data_dir, 'backward_speed_pairless_mult-1_bsz-32.csv'),
    }

    all_exist = all(os.path.exists(p) for p in files.values())
    if all_exist:
        df_paired = pd.read_csv(files['paired'])
        df_pairless = pd.read_csv(files['pairless'])
        df_paired_back = pd.read_csv(files['paired_back'])
        df_pairless_back = pd.read_csv(files['pairless_back'])

        time_speedup_forward = df_paired['time'].iloc[0] / df_pairless['time'].iloc[0]
        time_speedup_backward = df_paired_back['time'].iloc[0] / df_pairless_back['time'].iloc[0]
        mem_reduction_forward = df_paired['mem'].iloc[0] / df_pairless['mem'].iloc[0]
        mem_reduction_backward = df_paired_back['mem'].iloc[0] / df_pairless_back['mem'].iloc[0]

        names = ['Fwd\nTime', 'Bwd\nTime', 'Fwd\nMem', 'Bwd\nMem']
        values = [time_speedup_forward, time_speedup_backward, mem_reduction_forward, mem_reduction_backward]
        colors = ['lightgreen', 'lightgreen', 'lightskyblue', 'lightskyblue']

        bars = ax.bar(names, values, color=colors, edgecolor='black', linewidth=1)

        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height * 1.02,
                    f'{value:.1f}x', ha='center', va='bottom', fontsize=FONTSIZE_TICK)

        ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_ylabel('Ratio (IPA / FPA)', fontsize=FONTSIZE_LABEL)
        ax.set_title('FPA Improvement', fontsize=FONTSIZE_TITLE)
        ax.tick_params(axis='x', labelsize=FONTSIZE_TICK)
        ax.tick_params(axis='y', labelsize=FONTSIZE_TICK)

        legend_elements = [
            Patch(facecolor='lightgreen', edgecolor='black', label='Time speedup'),
            Patch(facecolor='lightskyblue', edgecolor='black', label='Memory reduction'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=FONTSIZE_LEGEND - 1)
        ax.set_ylim(top=ax.get_ylim()[1] * 1.15)

    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, 'combined_comparison.pdf'), dpi=DPI)
    plt.savefig(os.path.join(data_dir, 'combined_comparison.png'), dpi=DPI)
    print("Saved combined_comparison.pdf and combined_comparison.png")
    plt.close()


def plot_speedup(data_dir='.'):
    """Plot speedup ratios (IPA/FPA) for forward and backward passes."""
    fig, ax = plt.subplots(figsize=(8, 6))

    files = {
        'paired': os.path.join(data_dir, 'speed_paired_mult-1_bsz-32.csv'),
        'pairless': os.path.join(data_dir, 'speed_pairless_mult-1_bsz-32.csv'),
        'paired_back': os.path.join(data_dir, 'backward_speed_paired_mult-1_bsz-32.csv'),
        'pairless_back': os.path.join(data_dir, 'backward_speed_pairless_mult-1_bsz-32.csv'),
    }

    for name, path in files.items():
        if not os.path.exists(path):
            print(f"Missing {path}")
            return

    df_paired = pd.read_csv(files['paired'])
    df_pairless = pd.read_csv(files['pairless'])
    df_paired_back = pd.read_csv(files['paired_back'])
    df_pairless_back = pd.read_csv(files['pairless_back'])

    # Calculate speedups
    time_speedup_forward = df_paired['time'].iloc[0] / df_pairless['time'].iloc[0]
    time_speedup_backward = df_paired_back['time'].iloc[0] / df_pairless_back['time'].iloc[0]
    mem_reduction_forward = df_paired['mem'].iloc[0] / df_pairless['mem'].iloc[0]
    mem_reduction_backward = df_paired_back['mem'].iloc[0] / df_pairless_back['mem'].iloc[0]

    names = ['Forward\n(Time)', 'Backward\n(Time)', 'Forward\n(Memory)', 'Backward\n(Memory)']
    values = [time_speedup_forward, time_speedup_backward, mem_reduction_forward, mem_reduction_backward]
    colors = ['lightgreen', 'lightgreen', 'lightskyblue', 'lightskyblue']

    bars = ax.bar(names, values, color=colors, edgecolor='black', linewidth=1)

    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height * 1.02,
                f'{value:.1f}x', ha='center', va='bottom', fontsize=FONTSIZE_TICK)

    ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_ylabel('Speedup / Reduction Factor (IPA / FPA)', fontsize=FONTSIZE_LABEL)
    ax.set_title('FPA Improvement over IPA', fontsize=FONTSIZE_TITLE)
    ax.tick_params(axis='x', labelsize=FONTSIZE_TICK)
    ax.tick_params(axis='y', labelsize=FONTSIZE_TICK)

    # Add legend
    legend_elements = [
        Patch(facecolor='lightgreen', edgecolor='black', label='Time speedup'),
        Patch(facecolor='lightskyblue', edgecolor='black', label='Memory reduction'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=FONTSIZE_LEGEND)

    ax.set_ylim(top=ax.get_ylim()[1] * 1.15)

    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, 'speedup.pdf'), dpi=DPI)
    plt.savefig(os.path.join(data_dir, 'speedup.png'), dpi=DPI)
    print("Saved speedup.pdf and speedup.png")
    plt.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plot IPA vs FPA speed comparison results')
    parser.add_argument('--data-dir', type=str, default='.',
                        help='Directory containing CSV files from speed_test.py')
    args = parser.parse_args()

    plot_seq_len('mem', args.data_dir)
    plot_seq_len('time', args.data_dir)
    plot_typical('mem', args.data_dir)
    plot_typical('time', args.data_dir)
    plot_speedup(args.data_dir)
    plot_combined(args.data_dir)
