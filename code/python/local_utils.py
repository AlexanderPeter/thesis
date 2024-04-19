def print_parameters(model, verbose=False):
    param_count_total = 0
    param_count_required = 0
    for name, parameter in model.named_parameters():
        param_count = parameter.numel()
        param_count_total += param_count
        if parameter.requires_grad:
            param_count_required += param_count
        if verbose:
            print(f"{name}: {param_count}")

    print(f"Total parameters: {param_count_total}")
    print(f"Required parameters: {param_count_required} ")
    return param_count_total, param_count_required
