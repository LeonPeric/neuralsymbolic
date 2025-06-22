import ltn
import torch
from ltn.fuzzy_ops import ConnectiveOperator

eps = 1e-4


class Lukasiewicz_AND(ConnectiveOperator):
    def __call__(self, x, y):
        val = x + y - 1
        # Return the element-wise maximum
        return torch.maximum(torch.zeros_like(val), val)


class Lukasiewicz_OR(ConnectiveOperator):
    def __call__(self, x, y):
        val = x + y
        # Return the element-wise minimum
        return torch.minimum(torch.ones_like(val), val)


class Lukasiewicz_NOT(ConnectiveOperator):
    def __call__(self, x):
        return 1.0 - x


class AggregMean(ltn.fuzzy_ops.AggregationOperator):
    def __call__(self, xs, dim=None, keepdim=False, mask=None):
        if mask is not None:
            numerator = torch.sum(torch.where(~mask, torch.zeros_like(xs), xs))
            # we count the number of 1 in the mask
            denominator = torch.sum(mask)
            return torch.div(numerator, denominator)
        else:
            return torch.mean(xs, dim=dim, keepdim=keepdim)


class AggregMeanError(ltn.fuzzy_ops.AggregationOperator):
    def __call__(self, xs, dim=None, keepdim=False, mask=None):
        if mask is not None:
            # here, we put 1 where the mask is not satisfied, since 1 is the maximum value for a truth value.
            # this is a way to exclude values from the minimum computation
            xs = torch.where(~mask, 1.0, xs.double())
        out = torch.amin(xs, dim=dim, keepdim=keepdim)
        return out


def pi_0(x):
    return (1 - eps) * x + eps


def pi_1(x):
    return (1 - eps) * x


class Stable_AND(ConnectiveOperator):
    def __call__(self, x, y):
        x, y = pi_0(x), pi_0(y)
        return x * y


class Stable_OR(ConnectiveOperator):
    def __call__(self, x, y):
        x, y = pi_1(x), pi_1(y)
        return x + y - x * y


class Stable_NOT(ConnectiveOperator):
    def __call__(self, x):
        return 1.0 - x
