import torch
import torch.nn as nn

class AbsaDataset(torch.utils.data.Dataset):
    pass

def create_target_string(triplets : list[dict]):
    result = []
    triplets = sorted(triplets, key=lambda x : x.get("start_idx"))
    for triplet in triplets:
        result.append(f"[ASP] {triplet.get("aspect")} [OP] {triplet.get('opinion')} [SENT] {triplet.get('sentiment')} [EOS]")
    
    return result