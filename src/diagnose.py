import torch

def compute_erf(model, sequence_length=600, device='cpu'):
    model.eval()
    model = model.to(device)

    # Create input and require gradient on it
    x = torch.zeros(1, sequence_length, requires_grad=False).to(device)
    x = x.unsqueeze(0)  # (1, seq_len)

    # We need grad w.r.t. input
    x = torch.zeros(sequence_length, requires_grad=True, device=device)

    # Forward pass
    preds, _ = model(x)

    # Backprop from the last prediction only
    output = preds[0, -1, 0]  # scalar: last timestep prediction
    output.backward()

    # Gradient magnitude w.r.t. each input position
    sensitivity = x.grad.abs().cpu().numpy()

    return sensitivity