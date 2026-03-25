"""Train and evaluate MLP regressors on precomputed or live TAP embeddings."""

import os
import sys
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import TAPRegressor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

TAP_COLS = ["PSH", "PPC", "PNC", "SFvCSP"]
SEED = 42
DIR = os.path.dirname(__file__)


def load_embeddings(path):
    data = torch.load(path, weights_only=True)
    return data["embeddings"], data["targets"]


def split_data(embeddings, targets, seed=SEED):
    n = len(embeddings)
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]
    return (
        (embeddings[train_idx], targets[train_idx]),
        (embeddings[val_idx], targets[val_idx]),
        (embeddings[test_idx], targets[test_idx]),
    )


def train_model(
    train_data, val_data, input_dim, lr=1e-3, epochs=200, patience=15, device="cuda",
    use_wandb=False,
):
    train_emb, train_tgt = train_data
    val_emb, val_tgt = val_data

    # Z-score normalization on training targets
    tgt_mean = train_tgt.mean(dim=0)
    tgt_std = train_tgt.std(dim=0)
    train_tgt_norm = (train_tgt - tgt_mean) / tgt_std
    val_tgt_norm = (val_tgt - tgt_mean) / tgt_std

    train_loader = DataLoader(
        TensorDataset(train_emb, train_tgt_norm), batch_size=256, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_emb, val_tgt_norm), batch_size=512
    )

    model = TAPRegressor(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        n_batches = len(train_loader)
        for batch_idx, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss = loss.item()
            train_loss += batch_loss * x.size(0)
            print(f"\r  Epoch {epoch:3d}  batch {batch_idx}/{n_batches}  loss={batch_loss:.4f}", end="", flush=True)
        print()
        train_loss /= len(train_emb)

        # Validate — collect predictions in original scale
        model.eval()
        val_preds, val_actuals = [], []
        with torch.no_grad():
            for x, y in val_loader:
                pred_norm = model(x.to(device)).cpu()
                pred = pred_norm * tgt_std + tgt_mean
                val_preds.append(pred)
                # y is normalised; undo
                val_actuals.append(y * tgt_std + tgt_mean)
        val_preds = torch.cat(val_preds)
        val_actuals = torch.cat(val_actuals)
        val_loss = criterion(
            (val_preds - tgt_mean) / tgt_std,
            (val_actuals - tgt_mean) / tgt_std,
        ).item()

        maes, r2s = [], []
        for i in range(len(TAP_COLS)):
            p, a = val_preds[:, i], val_actuals[:, i]
            maes.append((p - a).abs().mean().item())
            ss_res = ((a - p) ** 2).sum().item()
            ss_tot = ((a - a.mean()) ** 2).sum().item()
            r2s.append(1 - ss_res / ss_tot if ss_tot > 0 else float("nan"))

        if epoch % 10 == 0 or epoch == 1:
            mae_str = "  ".join(f"{c}: {m:.3f}" for c, m in zip(TAP_COLS, maes))
            r2_str = "  ".join(f"{c}: {r:.3f}" for c, r in zip(TAP_COLS, r2s))
            star = " *" if val_loss < best_val_loss else ""
            print(
                f"  Epoch {epoch:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}{star}\n"
                f"    MAE  {mae_str}\n"
                f"    R²   {r2_str}"
            )

        if use_wandb:
            import wandb
            log = {"train/loss": train_loss, "val/loss": val_loss}
            for col, mae, r2 in zip(TAP_COLS, maes, r2s):
                log[f"val/MAE_{col}"] = mae
                log[f"val/R2_{col}"] = r2
            wandb.log(log, step=epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model, tgt_mean, tgt_std


def evaluate(model, test_data, tgt_mean, tgt_std, device="cuda"):
    test_emb, test_tgt = test_data
    model.eval()
    with torch.no_grad():
        pred_norm = model(test_emb.to(device)).cpu()
    # Unnormalize predictions
    pred = pred_norm * tgt_std + tgt_mean
    actual = test_tgt

    results = {}
    for i, col in enumerate(TAP_COLS):
        p, a = pred[:, i], actual[:, i]
        mse = ((p - a) ** 2).mean().item()
        mae = (p - a).abs().mean().item()
        ss_res = ((a - p) ** 2).sum().item()
        ss_tot = ((a - a.mean()) ** 2).sum().item()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        results[col] = {"MSE": mse, "MAE": mae, "R2": r2}
    return results


def compute_embeddings_live(encoder_name, device):
    """Compute embeddings in-memory without saving to disk."""
    from embed_sequences import load_data, embed_ablang2, embed_flabb

    seqs, targets = load_data()
    print(f"  Computing {encoder_name} embeddings for {len(seqs)} sequences...")
    if encoder_name == "ablang2":
        embeddings = embed_ablang2(seqs, device=device)
    else:
        embeddings = embed_flabb(seqs, device=device)
    return embeddings, targets


def run_one(name, input_dim, device, path=None, use_wandb=False):
    print(f"\n{'='*50}")
    print(f"  {name}  (input_dim={input_dim})")
    print(f"{'='*50}")

    if use_wandb:
        import wandb
        wandb.init(
            project="tap-regression",
            name=name,
            config={"encoder": name, "input_dim": input_dim, "mode": "frozen"},
        )

    if path is not None:
        embeddings, targets = load_embeddings(path)
    else:
        encoder = {"AbLang2": "ablang2", "FlashABB": "flabb"}[name]
        embeddings, targets = compute_embeddings_live(encoder, device)
    train_data, val_data, test_data = split_data(embeddings, targets)
    model, tgt_mean, tgt_std = train_model(
        train_data, val_data, input_dim, device=device, use_wandb=use_wandb,
    )

    # Save checkpoint
    ckpt_path = os.path.join(DIR, f"tap_{name.lower().replace(' ', '_')}.pt")
    torch.save(
        {"state_dict": model.state_dict(), "tgt_mean": tgt_mean, "tgt_std": tgt_std,
         "input_dim": input_dim},
        ckpt_path,
    )
    print(f"  Saved checkpoint to {ckpt_path}")

    results = evaluate(model, test_data, tgt_mean, tgt_std, device=device)

    if use_wandb:
        import wandb
        test_log = {}
        for col, r in results.items():
            test_log[f"test/MSE_{col}"] = r["MSE"]
            test_log[f"test/MAE_{col}"] = r["MAE"]
            test_log[f"test/R2_{col}"] = r["R2"]
        wandb.log(test_log)
        wandb.finish()

    return results


def print_comparison(all_results):
    print(f"\n{'='*70}")
    print("  Comparison: AbLang2 vs FlashABB")
    print(f"{'='*70}")
    header = f"{'Property':<10}"
    for name in all_results:
        header += f"  {name+' MSE':>14}  {name+' MAE':>14}  {name+' R²':>14}"
    print(header)
    print("-" * len(header))
    for col in TAP_COLS:
        row = f"{col:<10}"
        for name in all_results:
            r = all_results[name][col]
            row += f"  {r['MSE']:>14.4f}  {r['MAE']:>14.4f}  {r['R2']:>14.4f}"
        print(row)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--live", action="store_true",
        help="Compute embeddings on the fly instead of loading precomputed .pt files",
    )
    parser.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases")
    args = parser.parse_args()

    all_results = {}

    ablang_path = os.path.join(DIR, "embeddings_ablang2.pt")
    flabb_path = os.path.join(DIR, "embeddings_flabb.pt")

    if args.live or os.path.exists(ablang_path):
        path = None if args.live else ablang_path
        all_results["AbLang2"] = run_one("AbLang2", 480, args.device, path=path, use_wandb=args.wandb)
    else:
        print(f"Skipping AbLang2 — {ablang_path} not found. Run embed_sequences.py --model ablang2 or use --live.")

    if args.live or os.path.exists(flabb_path):
        path = None if args.live else flabb_path
        all_results["FlashABB"] = run_one("FlashABB", 128, args.device, path=path, use_wandb=args.wandb)
    else:
        print(f"Skipping FlashABB — {flabb_path} not found. Run embed_sequences.py --model flabb or use --live.")

    if len(all_results) > 0:
        print_comparison(all_results)


if __name__ == "__main__":
    main()
