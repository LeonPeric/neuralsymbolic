import ltn
import torch
from dataset import DataLoader, get_mnist_dataset_for_digits_addition
from logic import (
    AggregMean,
    AggregMeanError,
    Lukasiewicz_AND,
    Lukasiewicz_NOT,
    Lukasiewicz_OR,
)
from models import LogitsToPredicate, SingleDigitClassifier
from train import train_logic, train_simple

for i in range(50):
    train_set, test_set = get_mnist_dataset_for_digits_addition()

    # create train and test loader
    train_loader = DataLoader(train_set, 32, shuffle=True)
    test_loader = DataLoader(test_set, 32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    And = ltn.Connective(Lukasiewicz_AND())
    Exists = ltn.Quantifier(AggregMean(), quantifier="e")
    Forall = ltn.Quantifier(AggregMeanError(), quantifier="f")

    cnn_s_d = SingleDigitClassifier().to(device)
    Digit_s_d = ltn.Predicate(LogitsToPredicate(cnn_s_d)).to(device)
    optimizer = torch.optim.Adam(Digit_s_d.parameters(), lr=0.001)

    metrics_fuzzy = train_logic(
        Digit_s_d,
        optimizer,
        train_loader,
        test_loader,
        And,
        Exists,
        Forall,
        n_epochs=10,
        verbose=True,
    )
