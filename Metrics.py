import torch


def accuracy(prediction, target, is_ood=False):
    with torch.no_grad():
        if is_ood:
            predictions = (prediction >= 0).int()
            correct = torch.sum(predictions == target).item()
            total = len(target)
            return torch.tensor((correct, total))
        
        known = target >= 0
        total = torch.sum(known, dtype=int)
        if total:
            correct = torch.sum(
                torch.max(prediction[known], axis=1).indices == target[known], dtype=int
            )
        else:
            correct = 0
    return torch.tensor((correct, total))


def sphere(representation, target, sphere_radius=None, is_ood=False):
    # FIXME
    if is_ood:
        return None

    with torch.no_grad():
        known = target >= 0

        magnitude = torch.norm(representation, p=2, dim=1)

        sum = torch.sum(magnitude[~known])
        total = torch.sum(~known)

        if sphere_radius is not None:
            sum += torch.sum(torch.clamp(sphere_radius - magnitude, min=0.0))
            total += torch.sum(known)

    return torch.tensor((sum, total))


def confidence(logits, target, negative_offset=0.1, is_ood=False):
    with torch.no_grad():
        if is_ood:
            known = target > 0
            pred = torch.sigmoid(logits)
            confidence = 0.0
            if torch.sum(known):
                confidence += torch.sum(pred[known]).item() 
            if torch.sum(~known):
                confidence += torch.sum(1 - pred[~known]).item() 
            return torch.tensor((confidence, len(target)))
        
        known = target >= 0
        pred = torch.nn.functional.softmax(logits, dim=1)
        confidence = 0.0
        if torch.sum(known):
            confidence += torch.sum(pred[known, target[known]])
        if torch.sum(~known):
            confidence += torch.sum(
                1.0 + negative_offset - torch.max(pred[~known], dim=1)[0]
            )

    return torch.tensor((confidence, len(logits)))
