import torch

from matplotlib import pyplot as plt


torch.set_float32_matmul_precision("medium")
# torch.set_float32_matmul_precision("high")
# torch.set_float32_matmul_precision("highest")


def classic_dist(pts):
    rel_pos = pts[None,:,:] - pts[:,None,:]
    return rel_pos.norm(dim=-1) ** 2


def mat_dist(pts):
    sq_pts = pts**2
    ones = torch.ones_like(sq_pts)
    Q = torch.cat((sq_pts, -2*pts, ones), -1)
    K = torch.cat((ones, pts, sq_pts), -1).transpose(0,1)
    return torch.matmul(Q,K)


def run_experiment(dtype, device=torch.device('cuda'), scale=30.0):
    n = 1000
    pts = torch.randn((n, 3), device=device, dtype=dtype) * scale
    d1 = classic_dist(pts)
    d2 = mat_dist(pts)
    diff = (d1 - d2).abs().sqrt()
    avg_self = float(diff.diagonal().mean())
    # avg_self = float(diff.abs().mean())
    return avg_self


with torch.no_grad():
    errb16 = run_experiment(torch.bfloat16)
    err16 = run_experiment(torch.float16)
    err32 = run_experiment(torch.float32)

    fig, ax = plt.subplots(figsize=(8, 6))

    names = ['bfloat16', 'float16', 'float32']
    values = [errb16, err16, err32]
    # colors = ['lightblue', 'lightsalmon', 'lightgreen']
    colors = ['lightblue', 'lightblue', 'lightgreen']

    bars = ax.bar(names, values, color=colors, edgecolor='black', linewidth=1)

    ax.set_ylabel('Error of distance to self (Å)', fontsize=14)
    ax.set_title('Floating Point Precision Comparison', fontsize=16)
    ax.set_yscale('log')
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=11)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        # ax.text(bar.get_x() + bar.get_width()/2, height * 1.3,
        ax.text(bar.get_x() + bar.get_width()/2, height * 1.05,
                f'{value:.2f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('float_acc.pdf', dpi=150)
    plt.savefig('float_acc.png', dpi=150)
    print("Saved float_acc.pdf and float_acc.png")
