import torch
from torch.nn import Linear
import torch.nn.functional as F
import numpy as np


def load_tensor(fname: str, *shape: int, dtype=np.float32):
    return torch.tensor(np.fromfile(fname, dtype=dtype)).view(*shape)


def main():
    batch_size = 5
    emb_size = 30
    hidden_size = 500
    vocab_size = 10

    inputs = load_tensor("dump/input", batch_size, emb_size)
    target = load_tensor("dump/target", batch_size, dtype=np.int32).to(torch.int64)

    fc_1 = Linear(emb_size, hidden_size)
    fc_1.weight.data = load_tensor("dump/fc_1.w", emb_size, hidden_size).T
    fc_1.bias.data = load_tensor("dump/fc_1.b", hidden_size)
    fc_1_out = fc_1(inputs)
    fc_1_relu = F.relu(fc_1_out)

    fc_2 = Linear(hidden_size, vocab_size)
    fc_2.weight.data = load_tensor("dump/fc_2.w", hidden_size, vocab_size).T
    fc_2.bias.data = load_tensor("dump/fc_2.b", vocab_size)
    fc_2_out = fc_2(fc_1_relu)
    fc_2_softmax = F.softmax(fc_2_out, dim=-1)

    torch.testing.assert_close(fc_1_out, load_tensor("dump/fc_1.out", batch_size, hidden_size))
    torch.testing.assert_close(fc_1_relu, load_tensor("dump/fc_1.relu", batch_size, hidden_size))
    torch.testing.assert_close(fc_2_out, load_tensor("dump/fc_2.out", batch_size, vocab_size))
    torch.testing.assert_close(fc_2_softmax, load_tensor("dump/fc_2.softmax", batch_size, vocab_size))

    loss = F.cross_entropy(fc_2_out, target)
    print(loss.item())

    loss.backward()
    torch.testing.assert_close(fc_2.weight.grad, load_tensor("dump/fc_2.d_w", hidden_size, vocab_size).T)
    torch.testing.assert_close(fc_2.bias.grad, load_tensor("dump/fc_2.d_b", vocab_size))
    torch.testing.assert_close(fc_1.weight.grad, load_tensor("dump/fc_1.d_w", emb_size, hidden_size).T)
    torch.testing.assert_close(fc_1.bias.grad, load_tensor("dump/fc_1.d_b", hidden_size))


if __name__ == "__main__":
    main()
