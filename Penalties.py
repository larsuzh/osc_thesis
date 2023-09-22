import torch

def negative_penalty(weights, threshold=0.0):
    # Calculate the negative weights and their absolute values
    negative_weights = weights[weights < threshold]
    abs_negative_weights = torch.abs(negative_weights)
    
    # Calculate the penalty based on the negative weights' absolute values
    # and their frequency
    if abs_negative_weights.numel() == 0:
        penalty = 0.0
    else:
        penalty = torch.sum(abs_negative_weights) * ( negative_weights.numel() / weights.numel() )
    
    return penalty
