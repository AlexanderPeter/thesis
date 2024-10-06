import matplotlib.pyplot as plt


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
