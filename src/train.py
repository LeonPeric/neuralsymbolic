import ltn
import numpy as np
import torch
import torch.nn.functional as F
from ltn.fuzzy_ops import ConnectiveOperator
from models import LogitsToPredicate, SingleDigitClassifier
import os


def train_simple(
    model,
    optimizer,
    train_loader,
    test_loader,
    n_epochs=10,
    verbose=False,
    return_model=False,
    save_model=False,
):
    """
    Trains a neural model on digit addition using weak supervision from sum labels.

    This function uses soft expected value (via softmax) to infer digits and minimize MSE
    between predicted and true digit sums. Tracks multiple evaluation metrics per epoch.

    Args:
        model (nn.Module): Neural model predicting digit logits.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        train_loader (DataLoader): DataLoader yielding (images, sum, digits).
        test_loader (DataLoader): Same format for evaluation.
        n_epochs (int): Number of training epochs.
        verbose (bool): Whether to print metrics each epoch.

    Returns:
        dict: Dictionary of tracked metrics per epoch.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
    if verbose:
        print(f"Using device: {device}")
        
    metrics = {
        "loss": np.full([n_epochs], np.nan),
        "test_loss": np.full([n_epochs], np.nan),
        "accuracy_sum": np.full([n_epochs], np.nan),
        "test_accuracy_sum": np.full([n_epochs], np.nan),
        "test_accuracy_digit": np.full([n_epochs], np.nan),
    }

    mse_loss = torch.nn.MSELoss()
    digit_range = torch.arange(10).float().to(device)  # for expected value

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        correct_sum_train = 0
        total_train = 0

        # === TRAINING ===
        for operand_images, sum_label, _ in train_loader:
            img1 = operand_images[:, 0].to(device)
            img2 = operand_images[:, 1].to(device)
            sum_label = sum_label.to(device)

            optimizer.zero_grad()
            out1 = model(img1)
            out2 = model(img2)

            probs1 = F.softmax(out1, dim=1)
            probs2 = F.softmax(out2, dim=1)

            expected1 = probs1 @ digit_range  # weighted sum
            expected2 = probs2 @ digit_range
            pred_sum = expected1 + expected2

            loss = mse_loss(
                pred_sum.float(), sum_label.float()
            )  # MSE loss between predicted and true sum
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Track sum accuracy
            predicted_digit1 = probs1.argmax(dim=-1)
            predicted_digit2 = probs2.argmax(dim=-1)
            predicted_sum = predicted_digit1 + predicted_digit2

            correct_sum_train += len(
                [True for pred, label in zip(predicted_sum, sum_label) if pred == label]
            )
            total_train += sum_label.size(0)

        metrics["loss"][epoch] = total_loss / len(train_loader)
        metrics["accuracy_sum"][epoch] = correct_sum_train / total_train

        # === EVALUATION ===
        model.eval()
        correct_sum = 0
        correct_digits = 0
        total_test = 0
        test_loss = 0

        with torch.no_grad():
            for operand_images, sum_label, digit_labels in test_loader:
                img1 = operand_images[:, 0].to(device)
                img2 = operand_images[:, 1].to(device)
                # >> img shape: (32, 1, 28, 28)
                sum_label = sum_label.to(device)
                # >> sum_label shape: (32)
                digit_labels = digit_labels.to(device)

                # your code here
                out1 = model(img1)
                out2 = model(img2)
                # >> model output shape: torch.Size([32, 10])

                probs1 = F.softmax(out1, dim=1)
                probs2 = F.softmax(out2, dim=1)
                # >> probs shape: torch.Size([32, 10])

                expected1 = probs1 @ digit_range  # weighted sum
                expected2 = probs2 @ digit_range
                # >> expected shape: (32)
                # This is the expected value of the predicted digit?

                pred_sum = expected1 + expected2

                loss = mse_loss(pred_sum.float(), sum_label.float())

                test_loss += loss.item()

                predicted_digit1 = probs1.argmax(dim=-1)
                predicted_digit2 = probs2.argmax(dim=-1)
                pred_sum_class = predicted_digit1 + predicted_digit2

                # Count the number of times predicted sum is correct
                correct_sum += len(
                    [
                        True
                        for pred, label in zip(pred_sum_class, sum_label)
                        if pred == label
                    ]
                )
                # count the number of times predicted_digit1 matches the label
                correct_digits += (
                    len(
                        [
                            True
                            for pred, label in zip(predicted_digit1, digit_labels[:, 0])
                            if pred == label
                        ]
                    )
                    / digit_labels.shape[0]
                )
                # count the number of times predicted_digit2 matches the label
                correct_digits += (
                    len(
                        [
                            True
                            for pred, label in zip(predicted_digit2, digit_labels[:, 1])
                            if pred == label
                        ]
                    )
                    / digit_labels.shape[0]
                )
                total_test += sum_label.size(0)
                # Changed from total

        metrics["test_loss"][epoch] = test_loss / len(test_loader)
        metrics["test_accuracy_sum"][epoch] = correct_sum / total_test
        metrics["test_accuracy_digit"][epoch] = correct_digits / (2 * len(test_loader))

        if verbose and epoch == n_epochs - 1:
            print(
                f"Epoch {epoch+1} | "
                f"Train Loss: {total_loss:.4f}, "
                f"Train Sum Acc: {metrics['accuracy_sum'][epoch]*100:.2f}%, "
                f"Test Loss: {test_loss:.4f}, "
                f"Test Sum Acc: {metrics['test_accuracy_sum'][epoch]*100:.2f}%, "
                f"Test Digit Acc: {metrics['test_accuracy_digit'][epoch]*100:.2f}%"
            )

    # save model
    if save_model:
        print(f"Saving model to models/{model.__class__.__name__}.pth")
        # check if models folder exists, if not create it
        if not os.path.exists("models"):
            os.makedirs("models")
        torch.save(model.state_dict(), f"models/{model.__class__.__name__}.pth")

    if return_model:
        return metrics, model

    return metrics


def train_logic(
    model,
    optimizer,
    train_loader,
    test_loader,
    And,
    Exists,
    Forall,
    n_epochs=30,
    verbose=False,
    save_model=False,
    start_epoch=0,
):
    """
    Trains a model using Logic Tensor Networks (LTN) with symbolic constraints.

    The logic enforces that predicted digits should sum to the provided label using
    LTN's `Forall` and `Exists` quantifiers. Tracks sat rate, loss, and accuracy.

    Args:
        model (LTN Predicate): A predicate object returning logits.
        optimizer (torch.optim.Optimizer): Optimizer instance.
        train_loader (DataLoader): Loader yielding image pairs and sum.
        test_loader (DataLoader): Loader with ground truth for evaluation.
        n_epochs (int): Training duration.
        verbose (bool): Print epoch-level metrics.

    Returns:
        dict: Metrics dictionary including accuracy and satisfaction rates.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        print(f"Using device: {device}")
    metrics = {
        "loss": np.full([n_epochs], np.nan),
        "test_loss": np.full([n_epochs], np.nan),
        "accuracy_sum": np.full([n_epochs], np.nan),
        "test_accuracy_sum": np.full([n_epochs], np.nan),
        "test_accuracy_digit": np.full([n_epochs], np.nan),
        "sat": np.full([n_epochs], np.nan),
        "test_sat": np.full([n_epochs], np.nan),
    }

    d_1 = ltn.Variable("d_1", torch.tensor(range(10)))
    d_2 = ltn.Variable("d_2", torch.tensor(range(10)))

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        total_sat = 0
        correct_sum_train = 0
        total_train = 0

        for operand_images, sum_label, _ in train_loader:
            operand_images = operand_images.to(device)
            sum_label = sum_label.to(device)

            images_x = ltn.Variable("x", operand_images[:, 0])
            images_y = ltn.Variable("y", operand_images[:, 1])
            labels_z = ltn.Variable("z", sum_label)

            optimizer.zero_grad()

            sat_agg = Forall(
                ltn.diag(images_x, images_y, labels_z),
                Exists(
                    vars=[d_1, d_2],
                    formula=And(model(images_x, d_1), model(images_y, d_2)),
                    cond_vars=[d_1, d_2, labels_z],
                    cond_fn=lambda d1, d2, z: torch.eq(d1.value + d2.value, z.value),
                ),
            ).value

            loss = 1.0 - sat_agg
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_sat += sat_agg.item()

            # Predict digits for accuracy tracking
            out1 = torch.argmax(model.model.logits_model(operand_images[:, 0]), dim=1)
            out2 = torch.argmax(model.model.logits_model(operand_images[:, 1]), dim=1)

            predictions = out1 + out2
            correct_sum_train += (
                torch.count_nonzero(torch.eq(sum_label.to(ltn.device), predictions))
                / predictions.shape[0]
            )

            total_train += sum_label.size(0)

        metrics["loss"][epoch] = total_loss / len(train_loader)
        metrics["sat"][epoch] = total_sat / len(train_loader)
        metrics["accuracy_sum"][epoch] = correct_sum_train / len(train_loader)

        # === EVALUATION ===
        model.eval()
        correct_sum = 0
        correct_digits = 0
        total = 0
        test_loss = 0
        test_sat = 0

        with torch.no_grad():
            for operand_images, sum_label, digit_labels in test_loader:

                operand_images = operand_images.to(device)
                sum_label = sum_label.to(device)
                digit_labels = digit_labels.to(device)

                """ Your code here:
                    Create x,y,z ltn variables
                    Compute the SatAgg for the test set
                    Compute the loss for the test set
                """

                # Create x,y,z ltn variables
                images_x = ltn.Variable("x", operand_images[:, 0])
                images_y = ltn.Variable("y", operand_images[:, 1])
                labels_z = ltn.Variable("z", sum_label)

                # Compute the SatAgg for the test set
                sat_agg = Forall(
                    ltn.diag(images_x, images_y, labels_z),
                    Exists(
                        vars=[d_1, d_2],
                        formula=And(model(images_x, d_1), model(images_y, d_2)),
                        cond_vars=[d_1, d_2, labels_z],
                        cond_fn=lambda d1, d2, z: torch.eq(
                            d1.value + d2.value, z.value
                        ),
                    ),
                ).value

                # Compute the loss for the test set
                loss = 1.0 - sat_agg
                test_loss += loss.item()
                test_sat += sat_agg.item()

                # Predict digits for accuracy tracking
                out1 = torch.argmax(
                    model.model.logits_model(operand_images[:, 0]), dim=1
                )
                out2 = torch.argmax(
                    model.model.logits_model(operand_images[:, 1]), dim=1
                )

                predictions = out1 + out2
                correct_sum += (
                    torch.count_nonzero(torch.eq(sum_label.to(ltn.device), predictions))
                    / predictions.shape[0]
                )

                correct_digits += torch.count_nonzero(
                    torch.eq(digit_labels[:, 0].to(ltn.device), out1)
                )
                correct_digits += torch.count_nonzero(
                    torch.eq(digit_labels[:, 1].to(ltn.device), out2)
                )
                total += sum_label.shape[0]

        metrics["test_loss"][epoch] = test_loss / len(test_loader)
        metrics["test_sat"][epoch] = test_sat / len(test_loader)
        metrics["test_accuracy_sum"][epoch] = correct_sum / len(test_loader)
        metrics["test_accuracy_digit"][epoch] = correct_digits / (2 * total)

        if verbose and epoch == n_epochs - 1:
            print(
                f"Epoch {start_epoch + epoch+1:02d} | "
                f"Train Sat: {metrics['sat'][epoch]:.3f} | "
                f"Train Loss: {metrics['loss'][epoch]:.4f} | "
                f"Train Sum Acc: {metrics['accuracy_sum'][epoch]*100:.2f}% | "
                f"Test Sat: {metrics['test_sat'][epoch]:.3f} | "
                f"Test Loss: {metrics['test_loss'][epoch]:.4f} | "
                f"Test Sum Acc: {metrics['test_accuracy_sum'][epoch]*100:.2f}% | "
                f"Test Digit Acc: {metrics['test_accuracy_digit'][epoch]*100:.2f}%"
            )

    if save_model:
        torch.save(model.state_dict(), "trained_model.pth")

    return metrics
