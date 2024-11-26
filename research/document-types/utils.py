import matplotlib.pyplot as plt
import mlflow


def plot_and_log(plot_function, mlflow_file_name: str, **plot_kwargs):
    fig, ax = plt.subplots()
    plot_function(ax=ax, **plot_kwargs)
    mlflow.log_figure(fig, mlflow_file_name)
    plt.close(fig)
    return fig
