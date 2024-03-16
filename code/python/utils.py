def print_parameters(model):
    params = model.parameters()
    print(f"{sum(p.numel() for p in params)} parameters")
    print(f"{sum(p.numel() for p in params if p.requires_grad)} required parameters")
