import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

# model
# precision/recall scatterplot
# activisions
# spiderplot
# confusion matrix
# line plots with standard errors
# line plots with min/max


def plot_example_images(values):
    for i, (filepath, title) in enumerate(values):
        img = Image.open(filepath)
        img = img.convert("RGB")
        img = np.array(img)
        plt.subplot(1, len(values), i + 1)
        plt.axis("off")
        plt.imshow(img)
        plt.title(title)


def plot_distribution_barh(
    df_plot,
    stacked_column_name="set",
    bar_column_name="target_code",
    portion_names=["Training", "Validation", "Test"],
    show_values=False,
    stack_legend=None,
    bar_legend=None,
    title=None,
    # colors?
):
    ascending = False
    if portion_names is None:
        portion_names = df_plot[stacked_column_name].unique()
    if stack_legend is None:
        stack_legend = portion_names
    if bar_legend is None:
        bar_legend = (
            df_plot.groupby(bar_column_name)
            .size()
            .sort_index(ascending=ascending)
            .index
        )

    coordinates = None
    for portion_name in portion_names:
        portion_distribution = (
            df_plot[df_plot[stacked_column_name] == portion_name]
            .groupby(bar_column_name)
            .size()
            .sort_index(ascending=ascending)
        )
        values = portion_distribution.values
        patches = plt.barh(bar_legend, values, left=coordinates)

        if show_values:
            plt.bar_label(patches)

        if coordinates is None:
            coordinates = values
        else:
            coordinates = [a + b for a, b in zip(coordinates, values)]
    if title:
        plt.title(title)
    plt.legend(stack_legend)


def plot_tsne_scatter(features_reduced, color_list, labels=None):
    plt.ylabel("tSNE dimension 1")
    plt.xlabel("tSNE dimension 2")
    scatter = plt.scatter(
        features_reduced[:, 0],
        features_reduced[:, 1],
        c=color_list,
        # s=None,
        cmap="jet",
    )
    handles, labels_numbers = scatter.legend_elements(num=len(set(color_list)) - 1)
    ncol = 1
    if labels is None:
        ncol = 2
        labels = labels_numbers

    plt.legend(
        handles=handles,
        labels=labels,
        title="Label",
        ncol=ncol,
        loc="right",
        bbox_to_anchor=(1.4, 0.5),
    )
    plt.xticks([])
    plt.yticks([])
