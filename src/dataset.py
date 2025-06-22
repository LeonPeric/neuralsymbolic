import numpy as np
import torch
import torchvision


def get_mnist_dataset_for_digits_addition():
    """
    Prepares the dataset for MNIST single digit addition. Adapted from https://github.com/tommasocarraro/LTNtorch

    :return: a tuple of two elements. The first element is the training set, while the second element is the test set.
    Both training set and test set are lists that contain the following information:
        1. a list [left_operands, right_operands], where left_operands is a list of MNIST images that are used as the
        left operand of the addition, while right_operands is a list of MNIST images that are used as the right operand
        of the addition;
        2. a list containing the summation of the labels of the images contained in the list at point 1. The label of
        the left operand is added to the label of the right operand, and the target label is generated. This represents
        the target of the digits addition task.
    Note that this is the output of the process for the single digit case. In the multi digits case the list at point
    1 will have 4 elements since in the multi digits case four digits are involved in each addition (two digits
    represent the first operand and two digits the second operand).
    """
    n_train_examples = 30000
    n_test_examples = 5000
    n_operands = 2

    mnist_train = torchvision.datasets.MNIST(
        "./datasets/",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    mnist_test = torchvision.datasets.MNIST(
        "./datasets/",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    train_imgs, train_labels, test_imgs, test_labels = (
        mnist_train.data,
        mnist_train.targets,
        mnist_test.data,
        mnist_test.targets,
    )

    train_imgs, test_imgs = train_imgs / 255.0, test_imgs / 255.0

    train_imgs, test_imgs = torch.unsqueeze(train_imgs, 1), torch.unsqueeze(
        test_imgs, 1
    )

    imgs_operand_train = [
        train_imgs[i * n_train_examples : i * n_train_examples + n_train_examples]
        for i in range(n_operands)
    ]
    labels_operand_train = [
        train_labels[i * n_train_examples : i * n_train_examples + n_train_examples]
        for i in range(n_operands)
    ]

    imgs_operand_test = [
        test_imgs[i * n_test_examples : i * n_test_examples + n_test_examples]
        for i in range(n_operands)
    ]
    labels_operand_test = [
        test_labels[i * n_test_examples : i * n_test_examples + n_test_examples]
        for i in range(n_operands)
    ]

    label_addition_train = labels_operand_train[0] + labels_operand_train[1]
    label_addition_test = labels_operand_test[0] + labels_operand_test[1]

    label_digit_train = torch.stack(labels_operand_train, dim=1)
    label_digit_test = torch.stack(labels_operand_test, dim=1)

    train_set = [
        torch.stack(imgs_operand_train, dim=1),
        label_addition_train,
        label_digit_train,
    ]
    test_set = [
        torch.stack(imgs_operand_test, dim=1),
        label_addition_test,
        label_digit_test,
    ]

    return train_set, test_set


# this is a standard PyTorch DataLoader to load the dataset for the training and testing of the model
class DataLoader(object):
    """
    Custom data loader for handling digit-pair datasets.

    Loads batches of digit image pairs and their associated sum labels.
    Designed for both supervised and semi-supervised learning setups.
    """

    def __init__(self, fold, batch_size=1, shuffle=True):
        self.fold = fold
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(self.fold[0].shape[0] / self.batch_size))

    def __iter__(self):
        n = self.fold[0].shape[0]
        idxlist = list(range(n))
        if self.shuffle:
            np.random.shuffle(idxlist)

        for _, start_idx in enumerate(range(0, n, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, n)
            digits = self.fold[0][idxlist[start_idx:end_idx]]
            addition_labels = self.fold[1][idxlist[start_idx:end_idx]]
            digit_labels = self.fold[2][idxlist[start_idx:end_idx]]

            yield digits, addition_labels, digit_labels
