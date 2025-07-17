import numpy as np
import torch

from .load_model import load_model
from .model.flash_abb import featurize, FlashABBResult


class pretrained:

    def __init__(self, model_to_use="flash-abb", random_init=False, device='cuda'):
        super().__init__()
        
        self.used_device = torch.device(device)

        self.flabb, self.hparams = load_model(model_to_use, random_init=random_init)
        self.flabb.to(self.used_device)
        self.flabb.eval() # Default
        self.device = torch.device(device)

    def freeze(self):
        self.flabb.eval()

    def unfreeze(self):
        self.flabb.train()

    def __call__(self, seqs, batch_size=50):
        features = featurize(seqs, self.device)
        pred = self.flabb.model(
            {'single': features['single']},
            features['aatype'],
            features['res_idx'],
            features['mask']
        )
        result = FlashABBResult(seqs, pred, features['mask'])
        return result
