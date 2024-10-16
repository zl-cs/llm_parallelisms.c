import torch
from torch.nn import Linear
import torch.nn.functional as F
import numpy as np


def load_tensor(fname: str, *shape: int):
    return torch.tensor(np.fromfile(fname, dtype=np.float32)).view(*shape)


def main():
    batch_size = 5

    inputs = load_tensor("dump/input", batch_size, 30)
    fc_1 = Linear(30, 50)
    fc_1.weight.data = load_tensor("dump/fc_1.w", 30, 50).T
    fc_1.bias.data = load_tensor("dump/fc_1.b", 50)
    fc_1_out = fc_1(inputs)
    fc_1_relu = F.relu(fc_1_out)

    torch.testing.assert_close(fc_1_out, load_tensor("dump/fc_1.out", batch_size, 50))
    torch.testing.assert_close(fc_1_relu, load_tensor("dump/fc_1.relu", batch_size, 50))


if __name__ == "__main__":
    main()
