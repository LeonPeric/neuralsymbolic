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
