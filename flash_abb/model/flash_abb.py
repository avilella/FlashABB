import torch
from torch import nn
import numpy as np
from .openfold.np import residue_constants
from .structure_transformer import StructureModule


def atom14_to_atom37(position, aatype):
    from .openfold.utils.feats import (
        atom14_to_atom37 as openfold_atom14_to_atom37,
    )
    from .openfold.data.data_transforms import make_atom14_masks
    position = position.cpu()
    aatype = aatype.cpu()
    batch = make_atom14_masks({"aatype": aatype.squeeze().to(position.device)})
    return openfold_atom14_to_atom37(position.cpu(), batch)


def tokenize(seqs):
    heavy, light = seqs[0].split('|')
    heavy_encoded = torch.tensor([residue_constants.restype_order_with_x[aa] for aa in heavy])
    light_encoded = torch.tensor([residue_constants.restype_order_with_x[aa] for aa in light])
    single_aa_type = torch.cat((heavy_encoded, light_encoded), dim=-1)
    single_aa = torch.nn.functional.one_hot(single_aa_type, 21)
    heavy_heavy = torch.ones_like(heavy_encoded)
    light_heavy = torch.zeros_like(light_encoded)
    is_heavy = torch.cat((heavy_heavy, light_heavy), dim=-1)
    single_chain = torch.nn.functional.one_hot(is_heavy.long(), 2)
    single = torch.cat((single_aa, single_chain), dim=-1)
    heavy_index = torch.arange(len(heavy))
    light_index = torch.arange(len(light)) + 500
    res_index = torch.cat((heavy_index, light_index), dim=-1)
    return single, single_aa_type, res_index


class FlashABB(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.model = StructureModule(**params)


class FlashABBResult:
    def __init__(self, seqs, output):
        self.seqs = seqs
        self.output = output

    # @classmethod
    @property
    def coords(self):
        return self.output['positions'][-1,...]

    def to_pdbs(self, names):
        from .openfold.np.protein import Protein, to_pdb

        _, aatype, residue_idx = tokenize(self.seqs)
        residue_idx = residue_idx.unsqueeze(0)
        aatype = aatype.unsqueeze(0)
        coords = self.coords[0]
        coords = atom14_to_atom37(coords, aatype)
        coords = coords.detach().cpu().numpy()
        residue_idx = residue_idx[0,...].detach().cpu().numpy()
        aatype = aatype[0,...].long().detach().cpu().numpy()
        atom_mask = np.ones_like(coords[...,0])
        b_factors = np.zeros_like(atom_mask)
        prot = Protein(
            aatype=aatype,
            atom_positions=coords,
            atom_mask=atom_mask,
            residue_index=residue_idx + 1,
            b_factors=b_factors,
            chain_index=(residue_idx < 500).astype(int),
        )
        pdb_lines = to_pdb(prot)
        with open(names[0] + '.pdb', 'w') as f:
            f.write(pdb_lines)
