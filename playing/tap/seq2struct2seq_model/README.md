# Seq2Struct2Seq Model for TAP Property Prediction

This directory contains the seq2struct2seq model adapted from `/data/localhost/Code/struct_from_seq` for TAP property prediction.

## Model Architecture

The seq2struct2seq model uses a 3-stage pipeline:

1. **FlashABB (frozen)**: Predicts antibody structure from sequence
   - Input: Antibody sequence in "heavy|light" format
   - Output: Backbone coordinates (Ca, C, N atoms)

2. **StructureModule (trainable)**: Processes sequence + structure using FlashpointAttention blocks
   - Input: Tokenized sequence + predicted coordinates
   - Output: Per-residue embeddings (batch, seq_len, emb_size)
   - Uses Invariant Point Attention (IPA) to incorporate 3D geometric information

3. **TAP Regression Head (trainable)**: Predicts TAP properties
   - Input: Per-residue embeddings
   - Architecture: Per-residue MLP → masked sum pooling
   - Output: 4 TAP properties (PSH, PPC, PNC, SFvCSP)

## Files

- `best_weights_seq2struct2seq.pt` - Pretrained model weights (13 MB)
- `seq2struct2seq.py` - BERTCoords model definition
- `fpa_transformer/` - StructureModule and FlashpointAttention implementation
  - `internal_structure_transformer.py` - Main StructureModule class
  - `flashpoint_attention.py` - FlashpointAttention (IPA) blocks
  - Note: Uses openfold utilities from `flash_abb.model.openfold` (no duplication)

## Usage

### Training

```bash
# Finetune both StructureModule and TAP head
python finetune_tap_seq2struct2seq.py --device cuda

# Freeze StructureModule, train only TAP head (faster, less memory)
python finetune_tap_seq2struct2seq.py --freeze_encoder

# With W&B logging
python finetune_tap_seq2struct2seq.py --wandb
```

### Evaluation

```bash
# Evaluate on OAS test set
python eval_finetune_seq2struct2seq.py --device cuda

# Evaluate on therapeutic antibodies
python eval_finetune_seq2struct2seq.py --therapeutic --device cuda
```

## Hyperparameters

Default configuration (from pretrained model):
- `num_layers`: 6 (number of StructureModule blocks)
- `emb_size`: 512 (per-residue embedding dimension)
- `num_heads`: 12 (number of IPA heads)
- `c_ipa`: 16 (IPA hidden dimension)
- `dropout`: 0.0 (dropout rate)

Training parameters:
- `encoder_lr`: 1e-5 (learning rate for StructureModule)
- `head_lr`: 1e-3 (learning rate for TAP head)
- `batch_size`: 16 (same as other models)
- `epochs`: 50
- `patience`: 10 (early stopping)

## Key Differences from Other Models

1. **Structure-aware**: Uses 3D coordinates predicted by FlashABB
2. **Invariant Point Attention**: IPA blocks explicitly model geometric relationships
3. **Pretrained on sequence recovery**: Model was pretrained to recover masked sequences given structure
4. **Larger architecture**: 512-dim embeddings vs 128 (FlashABB) or 480 (AbLang2)
5. **Computational cost**: Slower due to IPA computation (FlashABB structure prediction is memory-efficient)

## Expected Performance

The model should capture both sequence and structural information, potentially improving predictions for properties that depend on:
- 3D spatial arrangement of residues
- Geometric features (e.g., surface patches, charge distribution)
- Structural context beyond linear sequence

Preliminary expectations:
- Better performance on PPC/PNC (charge patches require 3D understanding)
- Competitive with hybrid model on other properties
- May require more training time due to larger parameter count
