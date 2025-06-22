import torch
from dataset import DataLoader, get_mnist_dataset_for_digits_addition
from models import SingleDigitClassifier
from train import train_simple

for i in range(3):
    train_set, test_set = get_mnist_dataset_for_digits_addition()

    # create train and test loader
    train_loader = DataLoader(train_set, 32, shuffle=True)
    test_loader = DataLoader(test_set, 32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SingleDigitClassifier().to(device)  # should output shape (batch, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # use Adam with lr=0.001
    metrics_cnn = train_simple(
        model, optimizer, train_loader, test_loader, n_epochs=100, verbose=True
    )
