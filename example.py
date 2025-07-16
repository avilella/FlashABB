from flash_abb import pretrained
import torch

flabb = pretrained(model_to_use='flash-abb', device='cuda')

seq = [
'EVQLLESGGEVKKPGASVKVSCRASGYTFRNYGLTWVRQAPGQGLEWMGWISAYNGNTNYAQKFQGRVTLTTDTSTSTAYMELRSLRSDDTAVYFCARDVPGHGAAFMDVWGTGTTVTVSS', # The heavy chain (VH) needs to be the first element
'DIQLTQSPLSLPVTLGQPASISCRSSQSLEASDTNIYLSWFQQRPGQSPRRLIYKISNRDSGVPDRFSGSGSGTHFTLRISRVEADDVAVYYCMQGTHWPPAFGQGTKVDIK' # The light chain (VL) needs to be the second element
]
seqs = [f'{seq[0]}|{seq[1]}']
names = ['test1']

with torch.no_grad():
    result = flabb(seqs)

result.to_pdbs(names)
