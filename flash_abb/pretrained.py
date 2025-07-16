import numpy as np
import torch

from .load_model import load_model
from .model.flash_abb import tokenize, FlashABBResult


class pretrained:

    def __init__(self, model_to_use="flash-abb", random_init=False, device='cuda'):
        super().__init__()
        
        self.used_device = torch.device(device)

        self.flabb, self.hparams = load_model(model_to_use, random_init=random_init)
        self.flabb.to(self.used_device)
        self.flabb.eval() # Default

    def freeze(self):
        self.flabb.eval()

    def unfreeze(self):
        self.flabb.train()

    def __call__(self, seqs, batch_size=50):
        # TODO: batching + masking
        device = torch.device('cuda')
        encoded_seqs, single_aa, res_idxs = tokenize(seqs)
        encoded_seqs = encoded_seqs.unsqueeze(0).to(device).float()
        res_idxs = res_idxs.unsqueeze(0).to(device)
        single_aa = single_aa.unsqueeze(0).to(device)
        pred = self.flabb.model({'single': encoded_seqs}, single_aa, res_idxs, None)
        result = FlashABBResult(seqs, pred)
        return result
