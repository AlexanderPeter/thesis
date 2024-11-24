import json
import numpy as np
import os
import pandas as pd
import random
import timm
import torch
import torch.nn as nn
import torch.utils.data as data
from collections import OrderedDict
from transformers import ViTModel
from torchvision.models import resnet50
from transformers import set_seed as transformers_set_seed


def number_to_string(value, nan_value="None"):
    if str(value) != "nan" and value is not None:
        return str(int(value))
    else:  # NaN
        return nan_value


def set_seed(seed, verbose=True):
    if verbose:
        print(f"Setting seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers_set_seed(seed)


def print_parameters(model, verbose=False):
    if verbose:
        print(f"Number of entries: {len(model.state_dict())}")

    param_count_total = 0
    param_count_trainable = 0
    for name, parameter in model.named_parameters():
        param_count = parameter.numel()
        param_count_total += param_count
        if parameter.requires_grad:
            param_count_trainable += param_count
        if verbose:
            print(f"{name}: {param_count}")

    print(f"Trainable parameters: {param_count_trainable}/{param_count_total}")


def load_model(
    checkpoint_path, ignore_key_prefix=True, use_ssl_library=True, freeze=True
):
    if use_ssl_library:
        from ssl_library.src.pkg.embedder import Embedder
        from ssl_library.src.pkg.wrappers import Wrapper

    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    checkpoint_keys = list(checkpoint.keys())
    number_of_keys = len(checkpoint_keys)

    if 150 == number_of_keys:
        print("Loading vit_tiny_patch16_224 from timm-library")
        if use_ssl_library:
            model = Embedder.load_pretrained("vit_tiny_random")
        else:
            model = timm.create_model("vit_tiny_patch16_224", pretrained=False)
        model.head = nn.Sequential()
    elif 200 == number_of_keys:
        print("Loading vit_tiny_patch16_224 from transformers-library")
        if use_ssl_library:
            model = Embedder.load_pretrained("imagenet_vit_tiny")
        else:
            model = ViTModel.from_pretrained("WinKawaks/vit-tiny-patch16-224")
        model.head = nn.Sequential()
    elif 318 == number_of_keys:
        print("Loading ResNet50 from torchvision-library")
        model = resnet50(weights=None)
        model = nn.Sequential(*list(model.children())[:-1])
        if use_ssl_library:
            model = Wrapper(model=model)
    else:
        raise Exception(f"No architecture with '{number_of_keys}' keys implemented yet")

    model_keys = list(model.state_dict().keys())
    assert number_of_keys == len(model_keys), f"Checkpoint does not match architecture!"

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

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    return model


# load linewise json objects like JSON Lines (.jsonl) or newline-delimited JSON (.ndjson)
def load_pd_from_jsonl(jsonl_file_path):
    df = pd.read_json(jsonl_file_path, lines=True)
    json_file_name = os.path.basename(jsonl_file_path)
    print(f"Read {len(df)} entries from {json_file_name}")
    return df


# load concatenated json objects as array similar to relaxed JSON (.rjson) without square brackets
def load_pd_from_json(metric_file_path):
    metric_file = open(metric_file_path, "r")
    content = metric_file.read().replace("\n", "").replace("}{", "},{")
    entries = json.loads("[" + content + "]")
    metric_file_name = os.path.basename(metric_file_path)
    print(f"Read {len(entries)} entries from {metric_file_name}")
    return pd.DataFrame.from_records(entries)


def select_and_sort_dataframe(df, selection_config):
    for selection_key in selection_config:
        selection_value = selection_config[selection_key]
        assert (
            selection_key in df.columns.values
        ), f"No column found with name {selection_key}"
        if selection_value is None:
            continue
        elif isinstance(selection_value, str):
            assert (
                selection_value in df[selection_key].unique()
            ), f"No rows matching the given criteria {selection_key}={selection_value}"
            df = df[df[selection_key] == selection_value]
        elif hasattr(selection_value, "__iter__"):
            unique_values = df[selection_key].unique()
            match_dict = {
                unique_value: substring
                for substring in selection_value
                for unique_value in unique_values
                if substring in unique_value
            }
            df = df[df[selection_key].isin(match_dict.keys())]
            df[selection_key] = pd.Categorical(df[selection_key], match_dict.keys())
        else:
            print(
                f"No implementation for selection {selection_key} with type {type(selection_value)}"
            )
    return df.sort_values(by=list(selection_config.keys()))


def load_values_from_previous_epochs(run_path):
    loss_file_path = os.path.join(run_path, "loss.txt")
    latest_epoch = -1
    best_loss = None
    if os.path.exists(loss_file_path):
        df_metrics = load_pd_from_json(loss_file_path)
        if 0 < len(df_metrics):
            latest_epoch = df_metrics["epoch"].max()
            print(f"Latest epoch: {latest_epoch}")

            df_losses = (
                df_metrics[df_metrics["set"] == "valid"]
                .groupby(["epoch"])["loss"]
                .mean()
            )
            best_epoch = df_losses.argmin()
            best_loss = df_losses.min()
            print(f"Best epoch: {best_epoch} with {best_loss}")
    return (loss_file_path, latest_epoch, best_loss)
