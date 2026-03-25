"""Precompute and cache sum-pooled embeddings from AbLang2 or FlashABB."""

import argparse
import sys
import os

import pandas as pd
import torch
from tqdm import tqdm

# Add project root to path so we can import flash_abb
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

TAP_COLS = ["PSH", "PPC", "PNC", "SFvCSP"]
CSV_PATH = os.path.join(os.path.dirname(__file__), "OAS_paired_with_tap.csv")


def load_data():
    df = pd.read_csv(CSV_PATH)
    # Convert separator from / to |
    seqs = [s.replace("/", "|") for s in df["full_seq"].tolist()]
    targets = torch.tensor(df[TAP_COLS].values, dtype=torch.float32)
    return seqs, targets


@torch.no_grad()
def embed_ablang2(seqs, batch_size=64, device="cuda"):
    import ablang2

    ablang = ablang2.pretrained("ablang2-paired", device=device)
    all_pooled = []
    for i in tqdm(range(0, len(seqs), batch_size), desc="AbLang2"):
        batch_seqs = seqs[i : i + batch_size]
        tokenized = ablang.tokenizer(
            batch_seqs, pad=True, w_extra_tkns=False, device=device
        )
        rescoding = ablang.AbRep(tokenized).last_hidden_states  # (B, L, 480)
        pooled = rescoding.sum(dim=1)  # (B, 480)
        all_pooled.append(pooled.cpu())
    return torch.cat(all_pooled, dim=0)


@torch.no_grad()
def embed_flabb(seqs, batch_size=50, device="cuda"):
    from flash_abb import pretrained

    flabb = pretrained(model_to_use="flash-abb", device=device)
    all_pooled = []
    for i in tqdm(range(0, len(seqs), batch_size), desc="FlashABB"):
        batch_seqs = seqs[i : i + batch_size]
        result = flabb(batch_seqs)
        single = result.output["single"]  # (B, L, 128)
        # Mask out padding before summing
        mask = result.mask.unsqueeze(-1)  # (B, L, 1)
        pooled = (single * mask).sum(dim=1)  # (B, 128)
        all_pooled.append(pooled.cpu())
    return torch.cat(all_pooled, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["ablang2", "flabb"],
        required=True,
        help="Which encoder to use",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    seqs, targets = load_data()
    print(f"Loaded {len(seqs)} sequences")

    if args.model == "ablang2":
        embeddings = embed_ablang2(seqs, args.batch_size, args.device)
        out_path = os.path.join(os.path.dirname(__file__), "embeddings_ablang2.pt")
    else:
        embeddings = embed_flabb(seqs, args.batch_size, args.device)
        out_path = os.path.join(os.path.dirname(__file__), "embeddings_flabb.pt")

    torch.save({"embeddings": embeddings, "targets": targets}, out_path)
    print(f"Saved {embeddings.shape} embeddings to {out_path}")


if __name__ == "__main__":
    main()
