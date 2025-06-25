"""
Compute the gradient norms over the first epoch of training for 1000 models. Save the results and later plot them.
"""

###################################################################

import os
# Needed for reproducibility:
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import random
import numpy as np
import torch
import ltn

from dataset import get_mnist_dataset_for_digits_addition, DataLoader
from models import SingleDigitClassifier, LogitsToPredicate
from logic import Stable_AND
from train import train_logic_one_epoch

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # For reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set this after os.environ['CUBLAS_WORKSPACE_CONFIG']
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as e:
        print(f"Warning: torch.use_deterministic_algorithms(True) failed: {e}")

seed = 132
set_seed(seed)

###################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create train and test loader
train_set, test_set = get_mnist_dataset_for_digits_addition()
train_loader = DataLoader(train_set, 32, shuffle=True)
test_loader = DataLoader(test_set, 32, shuffle=False)

# Initialize storage for all the gradient norms and metrics
results = dict()
results['grad_norms'] = []
results['metrics'] = []

# Create a seed list for reproducibility
seed_list = [seed for seed in range(1337, 2337)]

for seed in seed_list:
    # Set seed
    set_seed(seed)

    # Initialize the model and logic components
    And = ltn.Connective(Stable_AND())
    Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=2), quantifier="e")
    Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
    cnn_s_d = SingleDigitClassifier().to(device)
    Digit_s_d = ltn.Predicate(LogitsToPredicate(cnn_s_d)).to(device)
    optimizer = torch.optim.Adam(Digit_s_d.parameters(), lr=0.001)

    metrics, grad_norms = train_logic_one_epoch(Digit_s_d, optimizer, train_loader, And, Exists, Forall, device, verbose=True)

    # Store the results
    results['grad_norms'].append(grad_norms)
    results['metrics'].append(metrics)

# Save the results to a file
output_file = 'grad_norms_exp_results.pt'
torch.save(results, output_file)
print(f"Results saved to {output_file}")