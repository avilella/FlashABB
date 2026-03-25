"""End-to-end finetuning of AbLang2/FlashABB encoders with TAP regression head."""

import os
import sys
import copy
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model import TAPRegressor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

TAP_COLS = ["PSH", "PPC", "PNC", "SFvCSP"]
SEED = 42
DIR = os.path.dirname(__file__)


# ---------------------------------------------------------------------------
# Encoder wrappers (nn.Module so parameters are registered for the optimizer)
# ---------------------------------------------------------------------------

class FlashABBEncoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        from flash_abb.load_model import load_model
        self.flabb, _ = load_model("flash-abb")
        self.flabb.to(device)
        self._device = device
        self.embed_dim = 128

    def forward(self, seqs):
        from flash_abb.model.flash_abb import featurize
        features = featurize(seqs, self._device)
        output = self.flabb.model(
            {"single": features["single"]},
            features["aatype"],
            features["res_idx"],
            features["mask"],
        )
        single = output["single"]  # (B, L, 128)
        mask = features["mask"].unsqueeze(-1)  # (B, L, 1)
        return (single * mask).sum(dim=1)  # (B, 128)


class AbLang2Encoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        import ablang2
        ablang = ablang2.pretrained("ablang2-paired", device=device)
        self.AbRep = ablang.AbRep  # nn.Module — registered as submodule
        self._tokenizer = ablang.tokenizer
        self._device = device
        self.embed_dim = 480

    def forward(self, seqs):
        tokenized = self._tokenizer(
            seqs, pad=True, w_extra_tkns=False, device=self._device
        )
        rescoding = self.AbRep(tokenized).last_hidden_states  # (B, L, 480)
        return rescoding.sum(dim=1)  # (B, 480)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SeqDataset(Dataset):
    def __init__(self, seqs, targets):
        self.seqs = seqs
        self.targets = targets

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.targets[idx]


def seq_collate(batch):
    seqs, targets = zip(*batch)
    return list(seqs), torch.stack(targets)


# ---------------------------------------------------------------------------
# Data loading / splitting
# ---------------------------------------------------------------------------

def load_data():
    import pandas as pd
    csv_path = os.path.join(DIR, "OAS_paired_with_tap.csv")
    df = pd.read_csv(csv_path)
    seqs = [s.replace("/", "|") for s in df["full_seq"].tolist()]
    targets = torch.tensor(df[TAP_COLS].values, dtype=torch.float32)
    return seqs, targets


def split_indices(n, seed=SEED):
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    return perm[:n_train], perm[n_train : n_train + n_val], perm[n_train + n_val :]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_finetune(
    encoder,
    head,
    train_set,
    val_set,
    tgt_mean,
    tgt_std,
    encoder_lr=1e-5,
    head_lr=1e-3,
    epochs=50,
    patience=10,
    batch_size=16,
    device="cuda",
    use_wandb=False,
):
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, collate_fn=seq_collate
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, collate_fn=seq_collate
    )

    tgt_mean_d = tgt_mean.to(device)
    tgt_std_d = tgt_std.to(device)

    optimizer = torch.optim.Adam([
        {"params": encoder.parameters(), "lr": encoder_lr},
        {"params": head.parameters(), "lr": head_lr},
    ])
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_encoder_state = None
    best_head_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        encoder.train()
        head.train()
        train_loss = 0.0
        n_train = 0
        n_batches = len(train_loader)
        for batch_idx, (seqs, targets) in enumerate(train_loader, 1):
            targets_norm = (targets.to(device) - tgt_mean_d) / tgt_std_d
            pooled = encoder(seqs)
            pred = head(pooled)
            loss = criterion(pred, targets_norm)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss = loss.item()
            train_loss += batch_loss * len(seqs)
            n_train += len(seqs)
            print(f"\r  Epoch {epoch:3d}  batch {batch_idx}/{n_batches}  loss={batch_loss:.4f}", end="", flush=True)
        print()
        train_loss /= n_train

        # --- validation with per-property metrics in original scale ----------
        encoder.eval()
        head.eval()
        val_preds, val_actuals = [], []
        with torch.no_grad():
            for seqs, targets in val_loader:
                pooled = encoder(seqs)
                pred_norm = head(pooled)
                pred = pred_norm * tgt_std_d + tgt_mean_d
                val_preds.append(pred.cpu())
                val_actuals.append(targets)
        val_preds = torch.cat(val_preds)
        val_actuals = torch.cat(val_actuals)

        # normalised MSE for early stopping (same scale as training loss)
        val_loss = criterion(
            (val_preds - tgt_mean) / tgt_std,
            (val_actuals - tgt_mean) / tgt_std,
        ).item()

        # per-property MAE & R²
        maes, r2s = [], []
        for i in range(len(TAP_COLS)):
            p, a = val_preds[:, i], val_actuals[:, i]
            maes.append((p - a).abs().mean().item())
            ss_res = ((a - p) ** 2).sum().item()
            ss_tot = ((a - a.mean()) ** 2).sum().item()
            r2s.append(1 - ss_res / ss_tot if ss_tot > 0 else float("nan"))

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
            best_encoder_state = copy.deepcopy(encoder.state_dict())
            best_head_state = copy.deepcopy(head.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    encoder.load_state_dict(best_encoder_state)
    head.load_state_dict(best_head_state)
    return encoder, head


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(encoder, head, test_set, tgt_mean, tgt_std, batch_size=16, device="cuda"):
    loader = DataLoader(test_set, batch_size=batch_size, collate_fn=seq_collate)
    tgt_mean_d = tgt_mean.to(device)
    tgt_std_d = tgt_std.to(device)

    encoder.eval()
    head.eval()
    all_pred, all_actual = [], []
    with torch.no_grad():
        for seqs, targets in loader:
            pooled = encoder(seqs)
            pred_norm = head(pooled)
            pred = pred_norm * tgt_std_d + tgt_mean_d
            all_pred.append(pred.cpu())
            all_actual.append(targets)

    pred = torch.cat(all_pred)
    actual = torch.cat(all_actual)

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


# ---------------------------------------------------------------------------
# Run one encoder
# ---------------------------------------------------------------------------

def run_one(encoder_name, display_name, device, use_wandb=False, **train_kwargs):
    print(f"\n{'='*50}")
    print(f"  {display_name}  (finetuning)")
    print(f"{'='*50}")

    if use_wandb:
        import wandb
        wandb.init(
            project="tap-finetune",
            name=display_name,
            config={"encoder": encoder_name, **train_kwargs},
        )

    seqs, targets = load_data()
    train_idx, val_idx, test_idx = split_indices(len(seqs))

    train_seqs = [seqs[i] for i in train_idx]
    val_seqs = [seqs[i] for i in val_idx]
    test_seqs = [seqs[i] for i in test_idx]
    train_tgt, val_tgt, test_tgt = targets[train_idx], targets[val_idx], targets[test_idx]

    tgt_mean = train_tgt.mean(dim=0)
    tgt_std = train_tgt.std(dim=0)

    train_set = SeqDataset(train_seqs, train_tgt)
    val_set = SeqDataset(val_seqs, val_tgt)
    test_set = SeqDataset(test_seqs, test_tgt)

    if encoder_name == "flabb":
        encoder = FlashABBEncoder(device)
    else:
        encoder = AbLang2Encoder(device)
    head = TAPRegressor(encoder.embed_dim).to(device)

    encoder, head = train_finetune(
        encoder, head, train_set, val_set, tgt_mean, tgt_std,
        device=device, use_wandb=use_wandb, **train_kwargs,
    )

    ckpt_path = os.path.join(DIR, f"tap_ft_{encoder_name}.pt")
    torch.save({
        "encoder_state": encoder.state_dict(),
        "head_state": head.state_dict(),
        "tgt_mean": tgt_mean,
        "tgt_std": tgt_std,
        "input_dim": encoder.embed_dim,
    }, ckpt_path)
    print(f"  Saved to {ckpt_path}")

    results = evaluate(
        encoder, head, test_set, tgt_mean, tgt_std,
        batch_size=train_kwargs.get("batch_size", 16), device=device,
    )

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
    print("  Comparison (finetuned)")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encoder", choices=["ablang2", "flabb", "both"], default="both")
    parser.add_argument("--encoder_lr", type=float, default=1e-5)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases")
    args = parser.parse_args()

    train_kwargs = dict(
        encoder_lr=args.encoder_lr,
        head_lr=args.head_lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
    )

    all_results = {}
    if args.encoder in ("ablang2", "both"):
        all_results["AbLang2"] = run_one("ablang2", "AbLang2", args.device, use_wandb=args.wandb, **train_kwargs)
    if args.encoder in ("flabb", "both"):
        all_results["FlashABB"] = run_one("flabb", "FlashABB", args.device, use_wandb=args.wandb, **train_kwargs)

    if all_results:
        print_comparison(all_results)


if __name__ == "__main__":
    main()
