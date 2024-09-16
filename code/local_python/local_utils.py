import os
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.utils.data as data
from collections import OrderedDict
from transformers import ViTModel
from torchvision.models import resnet50
from torchvision import transforms

from local_python.dataframe_image_dataset import DataframeImageDataset


def print_parameters(model, verbose=False, required_parameters=False):
    print(f"Number of entries: {len(model.state_dict())}")
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
    if required_parameters:
        print(f"Required parameters: {param_count_required} ")


def load_headless_model(checkpoint_path, ignore_key_prefix=True, use_ssl_library=True):
    if use_ssl_library:
        from ssl_library.src.pkg.embedder import Embedder
        from ssl_library.src.pkg.wrappers import Wrapper

    model_architecture = os.path.basename(os.path.dirname(checkpoint_path))
    print(
        f"Loading model with architecture '{model_architecture}' from {checkpoint_path}"
    )
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    checkpoint_keys = list(checkpoint.keys())

    if model_architecture == "resnet50":
        model = resnet50(weights=None)
        model = nn.Sequential(*list(model.children())[:-1])
        if use_ssl_library:
            model = Wrapper(model=model)
    elif model_architecture == "vit_t16_v1":
        if use_ssl_library:
            model = Embedder.load_pretrained("imagenet_vit_tiny")
            model.head = nn.Sequential()
        else:
            model = ViTModel.from_pretrained("WinKawaks/vit-tiny-patch16-224")
            model.head = nn.Sequential()
    elif model_architecture == "vit_t16_v2":
        if use_ssl_library:
            model = Embedder.load_pretrained("vit_tiny_random")
            model.head = nn.Sequential()
        else:
            model = timm.create_model("vit_tiny_patch16_224", pretrained=False)
            model.head = nn.Sequential()
    else:
        raise Exception(
            f"Model architecture '{model_architecture}' is not implemented yet"
        )

    model_keys = list(model.state_dict().keys())
    assert len(checkpoint_keys) == len(
        model_keys
    ), f"Checkpoint does not match architecture!"

    if (
        ignore_key_prefix
        and (checkpoint_keys[0] != model_keys[0])
        and checkpoint_keys[0].endswith(model_keys[0])
    ):
        prefix_index = checkpoint_keys[0].index(model_keys[0])
        prefix = checkpoint_keys[0][:prefix_index]
        print(f"Ignoring prefix '{prefix}'")
        checkpoint_new = OrderedDict()
        for key, value in checkpoint.items():
            assert key.startswith(prefix), f"Prefix '{prefix}' not found in key '{k}'"
            checkpoint_new[key[prefix_index:]] = value
        checkpoint = checkpoint_new

    model.load_state_dict(checkpoint, strict=True)

    for param in model.parameters():
        param.requires_grad = False

    return model


def load_dataloader(
    data_dir,
    batch_size=16,
    img_size=224,
    normalise_mean=(0.485, 0.456, 0.406),  # ImageNet
    normalise_std=(0.229, 0.224, 0.225),  # ImageNet
):
    df = pd.read_csv(data_dir)
    transform = transforms.Compose(
        [
            # NOTE: ResNet50_Weights.IMAGENET1K_V1 also uses these resize and crop values
            transforms.Resize((256, 256)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(normalise_mean, normalise_std),
        ]
    )

    # NOTE: DataframeImageDataset uses pil_loader as default, which executes Image.convert("RGB") implicitly
    ds_full = DataframeImageDataset(
        df,
        filepath_column="filepath",
        label_columns=["target_code", "set"],
        transform=transform,
    )

    dl_full = data.DataLoader(
        ds_full,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return dl_full
