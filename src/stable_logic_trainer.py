import ltn
import numpy as np
import torch
from dataset import DataLoader, get_mnist_dataset_for_digits_addition
from logic import Stable_AND
from models import LogitsToPredicate, SingleDigitClassifier
from train import train_logic, train_simple

for i in range(200):
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
    Digit_s_d = ltn.Predicate(LogitsToPredicate(cnn_s_d)).to(device)
    optimizer = torch.optim.Adam(Digit_s_d.parameters(), lr=0.001)
    metrics_prl = train_logic(
        Digit_s_d,
        optimizer,
        train_loader,
        test_loader,
        And,
        Exists,
        Forall,
        n_epochs=1,
        verbose=True,
    )
