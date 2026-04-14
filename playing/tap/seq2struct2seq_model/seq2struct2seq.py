from torch import Tensor
import torch
import torch.nn as nn
import math
from torch.nn.modules import TransformerEncoder, TransformerEncoderLayer
import sys
import os

# Add flash_abb to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

# from flash_abb.load_model import load_model
from flash_abb import pretrained
import ablang2

from .fpa_transformer.internal_structure_transformer import StructureModule

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from transformers import RobertaTokenizer

CACHED_TOKENIZERS = {}

def load_cached_tokenizer(tokenizer_key, cache=True):
    if not cache or tokenizer_key not in CACHED_TOKENIZERS:
        CACHED_TOKENIZERS[tokenizer_key] = RobertaTokenizer.from_pretrained(tokenizer_key)
    return CACHED_TOKENIZERS[tokenizer_key]

# 3D BERT
class BERTCoords(nn.Module):
    def __init__(
        self,
        num_heads=12,
        num_layers=8,
        emb_size=128,
        dropout: float = 0.1,
        use_coords=True,
    ):
        super(BERTCoords, self).__init__()

        self.emb_size = emb_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        # Load FlashABB model
        # self.folder, _ = load_model("flash-abb")
        self.folder = pretrained()
        # Load AbLang2 for tokenization
        self.ablang = ablang2.pretrained(model_to_use='ablang2-paired', random_init=False, device=self.folder.device)
        self.alphabet = self.ablang.tokenizer
        # Freeze base models
        # self.folder.requires_grad_(False)
        self.folder.flabb.requires_grad_(False)
        self.ablang.AbLang.requires_grad_(False)
        self.encoder = StructureModule(
            no_blocks = self.num_layers,
            embed_dim = self.emb_size,
            c_s = len(self.alphabet.aa_to_token),
            padding_idx=self.alphabet.pad_token,
            c_ipa = 16,
            no_heads_ipa = self.num_heads,
            dropout_rate = dropout,
        )

        self.norm = nn.LayerNorm(self.emb_size)
        self.generator = nn.Linear(self.emb_size, len(self.alphabet.aa_to_token))
        self.use_coords=use_coords

    def forward(
        self,
        src_seq,
        src: Tensor,
        mask=None,
        return_emb=False,
        src_seq_masked=None,
        return_attn_weights=False,
        # use_coords=False,
        # use_coords=True,
    ):
        sep_mask = src != self.alphabet.sep_token
        src_shape = list(src.shape)
        src_shape[1] = src.shape[1] - 1
        if self.use_coords:
            coords = self.folder(src_seq).bb_coords / 10
            # coords = self.folder(src_seq_masked).bb_coords / 10

        src_emb, rigids, attn_weights = self.encoder(
            src[sep_mask].view(src_shape),
            # pre_src,
            coords=coords if self.use_coords else None,
            return_attn_weights=return_attn_weights
        )
        if return_emb:
            return self.norm(src_emb)

        # if return_emb:
        #     return self.norm(src_emb), attn_weights
        logits = self.generator(self.norm(src_emb))

        if return_attn_weights:
            return logits, attn_weights
        return logits, rigids
