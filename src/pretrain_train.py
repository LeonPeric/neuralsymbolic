import numpy as np
import torch
import ltn

from dataset import DataLoader, get_mnist_dataset_for_digits_addition
from logic import Stable_AND
from models import LogitsToPredicate, SingleDigitClassifier
from train import train_logic, train_simple

from utils import plot_training_progression

for i in range(10):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(i)
        torch.cuda.manual_seed_all(i)

    torch.manual_seed(i)
    np.random.seed(i)

    # Additionally, some operations on a GPU are implemented stochastic for efficiency
    # We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_set, test_set = get_mnist_dataset_for_digits_addition()

    # create train and test loader
    train_loader = DataLoader(train_set, 32, shuffle=True)
    test_loader = DataLoader(test_set, 32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    And = ltn.Connective(Stable_AND())
    # we use relaxed aggregators: see paper for details
    Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=2), quantifier="e")
    Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")

    cnn_s_d = SingleDigitClassifier().to(device)
    optimizer = torch.optim.Adam(cnn_s_d.parameters(), lr=0.001)

    N_PRETRAIN_EPOCHS = 1
    pretrain_metrics, pretrained_model = train_simple(
        cnn_s_d,
        optimizer,
        train_loader,
        test_loader,
        n_epochs=N_PRETRAIN_EPOCHS,
        verbose=True,
        return_model=True,
        save_model=False,
    )

    optimizer = torch.optim.Adam(pretrained_model.parameters(), lr=0.001)
    Digit_s_d = ltn.Predicate(LogitsToPredicate(pretrained_model)).to(device)
    
    print("Pretraining complete. Starting logic training.")

    logic_metrics = train_logic(
        Digit_s_d,
        optimizer,
        train_loader,
        test_loader,
        And,
        Exists,
        Forall,
        n_epochs=1,
        verbose=True,
        save_model=False,
        start_epoch=N_PRETRAIN_EPOCHS,
    )


    plot_training_progression(pretrain_metrics, logic_metrics)

