import matplotlib.pyplot as plt
import numpy as np


def plot_training_metrics(metrics):
    """
    Plots 3 training metrics: loss, sum accuracy, and digit accuracy.
    """
    epochs = range(1, len(metrics["loss"]) + 1)

    fig, axs = plt.subplots(1, 3, figsize=(18, 4))

    for name in ["train", "test"]:
        if name == "train":
            # --- Plot 1: Losses ---
            axs[0].plot(epochs, metrics["loss"], label="train loss")
            # --- Plot 2: Sum Accuracies ---
            axs[1].plot(epochs, metrics["accuracy_sum"], label="train acc")
        else:
            axs[0].plot(epochs, metrics["test_loss"], label="test loss")
            axs[1].plot(epochs, metrics["test_accuracy_sum"], label="test acc")
            # --- Plot 3: Digit Accuracy ---
            axs[2].plot(epochs, metrics["test_accuracy_digit"], label="test digit acc")

    for ax in axs:
        ax.legend()
    plt.tight_layout()
    plt.show()


def plot_training_metrics_logical(metrics):
    """
    Plots 4 training metrics: loss, sat, sum accuracy, and digit accuracy.
    """
    epochs = range(1, len(metrics["loss"]) + 1)

    fig, axs = plt.subplots(1, 4, figsize=(18, 4))

    for name in ["train", "test"]:
        if name == "train":
            # --- Plot 1: Losses ---
            axs[0].plot(epochs, metrics["loss"], label="train loss")
            axs[0].set_title("Loss")
            # --- Plot 2: Satisfiability ---
            axs[1].plot(epochs, metrics["sat"], label="train sat")
            axs[1].set_title("Satisfiability")
            # --- Plot 3: Sum Accuracies ---
            axs[2].plot(epochs, metrics["accuracy_sum"], label="train acc")
            axs[2].set_title("Sum Accuracy")
        else:
            axs[0].plot(epochs, metrics["test_loss"], label="test loss")
            axs[1].plot(epochs, metrics["test_sat"], label="test sat")
            axs[2].plot(epochs, metrics["test_accuracy_sum"], label="test acc")
            # --- Plot 4: Digit Accuracy ---
            axs[3].plot(epochs, metrics["test_accuracy_digit"], label="test digit acc")
            axs[3].set_title("Digit Accuracy")

    for ax in axs:
        ax.legend()
    plt.tight_layout()
    plt.show()


def compare_training_metrics(metrics_cnn, metrics_prl):
    epochs = range(1, len(metrics_cnn["accuracy_sum"]) + 1)

    orange = "#FFA500"
    blue = "#1F77B4"

    fig, axs = plt.subplots(1, 2, figsize=(9, 4))

    # --- Plot 1: Sum Accuracies ---
    # Ideally plot the test and train sum accuracies for the baseline and LTN models
    axs[0].set_title("Sum Accuracy")
    # Plot CNN baseline train and test accuracies
    axs[0].plot(
        epochs, metrics_cnn["accuracy_sum"] * 100, c=orange, label="CNN (train)"
    )
    axs[0].plot(
        epochs,
        metrics_cnn["test_accuracy_sum"] * 100,
        c=orange,
        linestyle="dashed",
        label="CNN (test)",
    )
    # Plot LTN train and test accuracies
    axs[0].plot(epochs, metrics_prl["accuracy_sum"] * 100, c=blue, label="LTN (train)")
    axs[0].plot(
        epochs,
        metrics_prl["test_accuracy_sum"] * 100,
        c=blue,
        linestyle="dashed",
        label="LTN (test)",
    )

    # Set y-axis & x-axis
    axs[0].set_ylabel("Percentage Accuracy")
    axs[0].set_xlabel("Epoch")

    # --- Plot 2: Digit Accuracy ---
    # Ideally plot the test digit accuracies for the baseline and LTN models
    axs[1].set_title("Digit Accuracy")
    # Plot CNN baseline
    axs[1].plot(
        epochs,
        metrics_cnn["test_accuracy_digit"] * 100,
        c=orange,
        linestyle="dashed",
        label="CNN (test)",
    )
    axs[1].plot(
        epochs,
        metrics_prl["test_accuracy_digit"] * 100,
        c=blue,
        linestyle="dashed",
        label="LTN (test)",
    )

    # Set y-axis & x-axis
    axs[1].set_ylabel("Percentage Accuracy")
    axs[1].set_xlabel("Epoch")

    for ax in axs:
        ax.legend()

    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import numpy as np


def plot_training_progression(
    pretrain_metrics, logic_metrics, title="Training Progression", figsize=(15, 12)
):
    """
    Plot training metrics showing both pretraining and logic training phases.

    Args:
        pretrain_metrics (dict): Metrics from train_simple
        logic_metrics (dict): Metrics from train_logic
        title (str): Overall title for the plot
        figsize (tuple): Figure size
    """
    # Get the number of epochs for each phase
    pretrain_epochs = len(pretrain_metrics["loss"])
    logic_epochs = len(logic_metrics["loss"])
    total_epochs = pretrain_epochs + logic_epochs
    
    print(pretrain_epochs)
    print(logic_epochs)
    print(total_epochs)

    # Create epoch arrays
    pretrain_x = np.arange(pretrain_epochs)
    logic_x = np.arange(pretrain_epochs, total_epochs)

    # Common metrics between both phases
    common_metrics = [
        "loss",
        "test_loss",
        "accuracy_sum",
        "test_accuracy_sum",
        "test_accuracy_digit",
    ]

    # Logic-only metrics
    logic_only_metrics = ["sat", "test_sat"]

    # Create subplots
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # Plot common metrics
    for i, metric in enumerate(common_metrics):
        row = i // 3
        col = i % 3
        ax = axes[row, col]

        # Plot pretraining phase
        pretrain_values = pretrain_metrics[metric]
        pretrain_mask = ~np.isnan(pretrain_values)
        ax.plot(
            pretrain_x[pretrain_mask],
            pretrain_values[pretrain_mask],
            "b-",
            linewidth=2,
            label="Pretraining",
            marker="o",
            markersize=3,
        )

        # Plot logic training phase
        logic_values = logic_metrics[metric]
        logic_mask = ~np.isnan(logic_values)
        ax.plot(
            logic_x[logic_mask],
            logic_values[logic_mask],
            "r-",
            linewidth=2,
            label="Logic Training",
            marker="s",
            markersize=3,
        )

        # Add vertical line to show regime change
        ax.axvline(
            x=pretrain_epochs - 0.5,
            color="green",
            linestyle="--",
            linewidth=2,
            label="Regime Change",
        )

        ax.set_title(f'{metric.replace("_", " ").title()}', fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Format y-axis for accuracy metrics
        if "accuracy" in metric:
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda y, _: "{:.1%}".format(y))
            )

    # Plot logic-only metrics
    for i, metric in enumerate(logic_only_metrics):
        row = (len(common_metrics) + i) // 3
        col = (len(common_metrics) + i) % 3
        ax = axes[row, col]

        # Plot empty pretraining phase (no satisfaction for simple training)
        ax.plot(
            pretrain_x,
            np.full(pretrain_epochs, np.nan),
            "b-",
            linewidth=2,
            label="Pretraining (N/A)",
            alpha=0.5,
        )

        # Plot logic training phase
        logic_values = logic_metrics[metric]
        logic_mask = ~np.isnan(logic_values)
        ax.plot(
            logic_x[logic_mask],
            logic_values[logic_mask],
            "r-",
            linewidth=2,
            label="Logic Training",
            marker="s",
            markersize=3,
        )

        # Add vertical line to show regime change
        ax.axvline(
            x=pretrain_epochs - 0.5,
            color="green",
            linestyle="--",
            linewidth=2,
            label="Regime Change",
        )

        ax.set_title(f'{metric.replace("_", " ").title()}', fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Satisfaction Rate")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Remove unused subplot
    if len(common_metrics) + len(logic_only_metrics) < 9:
        axes[2, 2].remove()

    plt.tight_layout()
    plt.show()


def plot_final_results_histogram(
    pretrain_metrics, logic_metrics, title="Final Training Results", figsize=(15, 10)
):
    """
    Plot histograms comparing final results from pretraining vs logic training.

    Args:
        pretrain_metrics (dict): Metrics from train_simple
        logic_metrics (dict): Metrics from train_logic
        title (str): Overall title for the plot
        figsize (tuple): Figure size
    """
    # Common metrics between both phases
    common_metrics = [
        "loss",
        "test_loss",
        "accuracy_sum",
        "test_accuracy_sum",
        "test_accuracy_digit",
    ]

    # Logic-only metrics
    logic_only_metrics = ["sat", "test_sat"]

    # Create subplots
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # Plot common metrics
    for i, metric in enumerate(common_metrics):
        row = i // 3
        col = i % 3
        ax = axes[row, col]

        # Get final values (last non-NaN value)
        pretrain_final = pretrain_metrics[metric][~np.isnan(pretrain_metrics[metric])][
            -1
        ]
        logic_final = logic_metrics[metric][~np.isnan(logic_metrics[metric])][-1]

        # Create bar chart
        methods = ["Pretraining\nFinal", "Logic Training\nFinal"]
        values = [pretrain_final, logic_final]
        colors = ["lightblue", "lightcoral"]

        bars = ax.bar(
            methods, values, color=colors, alpha=0.7, edgecolor="black", linewidth=1
        )

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            if "accuracy" in metric:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + height * 0.01,
                    f"{value:.1%}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + height * 0.01,
                    f"{value:.4f}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

        ax.set_title(f'{metric.replace("_", " ").title()}', fontweight="bold")
        ax.set_ylabel("Final Value")

        # Format y-axis for accuracy metrics
        if "accuracy" in metric:
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda y, _: "{:.1%}".format(y))
            )
            ax.set_ylim(0, min(1, max(values) * 1.1))
        else:
            ax.set_ylim(0, max(values) * 1.1)

        ax.grid(True, alpha=0.3, axis="y")

    # Plot logic-only metrics
    for i, metric in enumerate(logic_only_metrics):
        row = (len(common_metrics) + i) // 3
        col = (len(common_metrics) + i) % 3
        ax = axes[row, col]

        # Get final value for logic training only
        logic_final = logic_metrics[metric][~np.isnan(logic_metrics[metric])][-1]

        # Create single bar
        bar = ax.bar(
            ["Logic Training\nFinal"],
            [logic_final],
            color="lightcoral",
            alpha=0.7,
            edgecolor="black",
            linewidth=1,
        )

        # Add value label
        ax.text(
            bar[0].get_x() + bar[0].get_width() / 2.0,
            logic_final + logic_final * 0.01,
            f"{logic_final:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

        ax.set_title(f'{metric.replace("_", " ").title()}', fontweight="bold")
        ax.set_ylabel("Final Satisfaction Rate")
        ax.set_ylim(0, min(1, logic_final * 1.1))
        ax.grid(True, alpha=0.3, axis="y")

        # Add note about pretraining N/A
        ax.text(
            0,
            logic_final * 0.5,
            "Pretraining:\nN/A",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.5),
        )

    # Remove unused subplot
    if len(common_metrics) + len(logic_only_metrics) < 9:
        axes[2, 2].remove()

    plt.tight_layout()
    plt.show()
