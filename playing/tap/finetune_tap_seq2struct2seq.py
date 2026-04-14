"""Finetuning seq2struct2seq model for TAP property prediction.

The seq2struct2seq model uses:
1. FlashABB (frozen) to predict structure from sequence
2. StructureModule (FlashpointAttention blocks) to process sequence + coordinates
3. TAP regression head (replacing the original generator layer)
"""

import os
import sys
import copy
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Add parent directory to path for flash_abb imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from finetune_tap import (
    TAP_COLS, SEED, DIR, SeqDataset, seq_collate,
    load_data, split_indices,
)

from seq2struct2seq_model.seq2struct2seq import BERTCoords


# ---------------------------------------------------------------------------
# Encoder wrapper for seq2struct2seq
# ---------------------------------------------------------------------------

class Seq2Struct2SeqEncoder(nn.Module):
    """Wrapper for BERTCoords that extracts per-residue embeddings."""

    def __init__(self, device, num_heads=12, num_layers=6, emb_size=512, dropout=0.0):
        super().__init__()
        self.device = device
        self.model = BERTCoords(
            num_heads=num_heads,
            num_layers=num_layers,
            emb_size=emb_size,
            dropout=dropout,
            use_coords=True,  # Use structure information
        )
        self.model = self.model.to(device)
        self.alphabet = self.model.alphabet
        self.embed_dim = emb_size

    def load_pretrained(self, weights_path):
        """Load pretrained weights."""
        state_dict = torch.load(weights_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state_dict)
        print(f"Loaded pretrained weights from {weights_path}")

    def freeze_structure_module(self):
        """Freeze the FlashABB and StructureModule (keep only TAP head trainable)."""
        self.model.folder.flabb.requires_grad_(False)
        self.model.ablang.AbLang.requires_grad_(False)
        self.model.encoder.requires_grad_(False)
        print("Froze FlashABB, AbLang2, and StructureModule")

    def forward(self, seqs):
        """
        Returns per-residue embeddings and mask.

        Args:
            seqs: List of sequences in "heavy|light" format

        Returns:
            per_residue_emb: (batch, seq_len-1, emb_size)  # -1 because separator is removed
            mask: (batch, seq_len-1)
        """
        # Tokenize sequences
        in_tokens = self.alphabet(seqs, pad=True, w_extra_tkns=False)
        pad_mask = in_tokens.eq(self.alphabet.pad_token)
        in_tokens = in_tokens.to(self.device)
        pad_mask = pad_mask.to(self.device)

        # Get per-residue embeddings (not logits)
        # Note: BERTCoords internally removes the separator token before processing
        per_residue_emb = self.model(
            seqs, in_tokens, pad_mask, return_emb=True
        )  # (batch, seq_len-1, emb_size)

        # Create mask that matches the embeddings (also remove separator)
        sep_mask = in_tokens != self.alphabet.sep_token
        src_shape = list(in_tokens.shape)
        src_shape[1] = src_shape[1] - 1

        # Apply same separator removal as BERTCoords does
        mask = ~pad_mask  # True where not padding
        mask = mask[sep_mask].view(src_shape)  # Remove separator position

        return per_residue_emb, mask


# ---------------------------------------------------------------------------
# TAP Regressor (same as in other scripts)
# ---------------------------------------------------------------------------

class TAPRegressor(nn.Module):
    """Per-residue MLP that predicts TAP contributions, then sum pools."""

    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 4),  # 4 TAP properties per residue
        )

    def forward(self, per_residue_emb, mask):
        """
        Args:
            per_residue_emb: (batch, seq_len, input_dim)
            mask: (batch, seq_len)
        Returns:
            (batch, 4) summed TAP predictions
        """
        # Apply MLP to each residue
        per_residue_tap = self.mlp(per_residue_emb)  # (batch, seq_len, 4)

        # Masked sum pooling
        mask_expanded = mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
        masked_tap = per_residue_tap * mask_expanded  # (batch, seq_len, 4)
        summed_tap = masked_tap.sum(dim=1)  # (batch, 4)

        return summed_tap


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
            per_residue_emb, mask = encoder(seqs)
            pred = head(per_residue_emb, mask)
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
                per_residue_emb, mask = encoder(seqs)
                pred_norm = head(per_residue_emb, mask)
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
            per_residue_emb, mask = encoder(seqs)
            pred_norm = head(per_residue_emb, mask)
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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encoder_lr", type=float, default=1e-5)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8, help="Smaller batch size due to structure computation")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num_layers", type=int, default=6, help="Number of StructureModule layers")
    parser.add_argument("--emb_size", type=int, default=512, help="Embedding size")
    parser.add_argument("--freeze_encoder", action="store_true", help="Freeze StructureModule (only train TAP head)")
    parser.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases")
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"  Seq2Struct2Seq (finetuning)")
    print(f"{'='*50}")

    if args.wandb:
        import wandb
        wandb.init(
            project="tap-finetune",
            name="Seq2Struct2Seq",
            config=vars(args),
        )

    # Load data
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

    # Create encoder and load pretrained weights
    encoder = Seq2Struct2SeqEncoder(
        device=args.device,
        num_layers=args.num_layers,
        emb_size=args.emb_size,
    )

    weights_path = os.path.join(DIR, "seq2struct2seq_model", "best_weights_seq2struct2seq.pt")
    # weights_path = os.path.join(DIR, "seq2struct2seq_model", "best_weights_seq2struct2seq_small.pt")
    encoder.load_pretrained(weights_path)

    if args.freeze_encoder:
        encoder.freeze_structure_module()
        print("  Training only TAP head (frozen encoder)")
    else:
        print("  Finetuning both encoder and TAP head")

    # Create TAP regression head
    head = TAPRegressor(encoder.embed_dim).to(args.device)

    # Train
    encoder, head = train_finetune(
        encoder, head, train_set, val_set, tgt_mean, tgt_std,
        encoder_lr=args.encoder_lr,
        head_lr=args.head_lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        device=args.device,
        use_wandb=args.wandb,
    )

    # Save checkpoint
    ckpt_path = os.path.join(DIR, "tap_ft_seq2struct2seq.pt")
    torch.save({
        "encoder_state": encoder.state_dict(),
        "head_state": head.state_dict(),
        "tgt_mean": tgt_mean,
        "tgt_std": tgt_std,
        "input_dim": encoder.embed_dim,
        "num_layers": args.num_layers,
        "emb_size": args.emb_size,
    }, ckpt_path)
    print(f"  Saved to {ckpt_path}")

    # Evaluate
    results = evaluate(
        encoder, head, test_set, tgt_mean, tgt_std,
        batch_size=args.batch_size, device=args.device,
    )

    print(f"\n{'='*50}")
    print("  Test Results")
    print(f"{'='*50}")
    for col in TAP_COLS:
        r = results[col]
        print(f"{col}:")
        print(f"  MSE: {r['MSE']:.4f}")
        print(f"  MAE: {r['MAE']:.4f}")
        print(f"  R²:  {r['R2']:.4f}")

    if args.wandb:
        import wandb
        test_log = {}
        for col, r in results.items():
            test_log[f"test/MSE_{col}"] = r["MSE"]
            test_log[f"test/MAE_{col}"] = r["MAE"]
            test_log[f"test/R2_{col}"] = r["R2"]
        wandb.log(test_log)
        wandb.finish()


if __name__ == "__main__":
    main()
