import torch
from torch.nn import Linear, Embedding
import torch.nn.functional as F
from torch.optim.sgd import SGD
import numpy as np


def load_tensor(fname: str, *shape: int, dtype=np.float32):
    return torch.tensor(np.fromfile(fname, dtype=dtype)).view(*shape)


# TODO(eugen): remove.
def overfit():
    batch_size = 32
    seq_len = 16
    emb_size = 16
    hidden_size = 4 * emb_size
    vocab_size = 27

    Xs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 9, 18, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 1, 18, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 13, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 18, 25, 19, 9, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 12, 5, 11, 19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 19, 21, 11, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 15, 13, 1, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 5, 25, 12, 1, 14, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26, 25, 18, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 5, 8, 12, 1, 14, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 22, 5, 14, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 5, 25, 18, 5, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 1, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 12, 1, 14, 3, 8, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 19, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 9, 14, 19, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 1, 1, 26, 9, 5, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 12, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 1, 22, 9, 21, 19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 9, 3, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 9, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 20, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Ys = [14, 9, 1, 11, 0, 1, 0, 21, 22, 0, 1, 2, 25, 18, 5, 22, 1, 0, 0, 25, 0, 12, 0, 2, 14, 0, 22, 0, 12, 8, 13, 7]

    Xs = torch.tensor(Xs)
    Ys = torch.tensor(Ys)
    wte = Embedding(vocab_size, emb_size)
    wte.weight.data = load_tensor("dump/wte", vocab_size, emb_size)
    fc_1 = Linear(seq_len * emb_size, hidden_size)
    fc_1.weight.data = load_tensor("dump/fc_1.w", seq_len * emb_size, hidden_size).T
    fc_1.bias.data = load_tensor("dump/fc_1.b", hidden_size)
    fc_2 = Linear(hidden_size, vocab_size)
    fc_2.weight.data = load_tensor("dump/fc_2.w", hidden_size, vocab_size).T
    fc_2.bias.data = load_tensor("dump/fc_2.b", vocab_size)

    lr = 1e-3
    params = [wte.weight, fc_1.weight, fc_1.bias, fc_2.weight, fc_2.bias]
    for step in range(10):
        h = wte(Xs)
        h = fc_1(h.view(batch_size, -1))
        h = F.relu(h)
        h = fc_2(h)
        loss = F.cross_entropy(h, Ys)

        for param in params:
            param.grad = None
        loss.backward()
        for param in params:
            param.data -= lr * param.grad
        loss = loss.item()
        print(f"{step=}, {loss=}")


def main():
    batch_size = 32
    seq_len = 16
    emb_size = 16
    hidden_size = 4 * emb_size
    vocab_size = 27

    Xs = load_tensor("dump/Xs", batch_size * seq_len, dtype=np.int32).to(torch.int64)
    Ys = load_tensor("dump/Ys", batch_size, dtype=np.int32).to(torch.int64)

    wte = Embedding(vocab_size, emb_size)
    wte.weight.data = load_tensor("dump/wte", vocab_size, emb_size)
    wte_out = wte(Xs)

    fc_1 = Linear(seq_len * emb_size, hidden_size)
    fc_1.weight.data = load_tensor("dump/fc_1.w", seq_len * emb_size, hidden_size).T
    fc_1.bias.data = load_tensor("dump/fc_1.b", hidden_size)
    fc_1_out = fc_1(wte_out.view(batch_size, seq_len * emb_size))
    fc_1_relu = F.relu(fc_1_out)

    fc_2 = Linear(hidden_size, vocab_size)
    fc_2.weight.data = load_tensor("dump/fc_2.w", hidden_size, vocab_size).T
    fc_2.bias.data = load_tensor("dump/fc_2.b", vocab_size)
    fc_2_out = fc_2(fc_1_relu)
    fc_2_softmax = F.softmax(fc_2_out, dim=-1)

    torch.testing.assert_close(wte_out, load_tensor("dump/wte.out", batch_size * seq_len, emb_size))
    torch.testing.assert_close(fc_1_out, load_tensor("dump/fc_1.out", batch_size, hidden_size))
    torch.testing.assert_close(fc_1_relu, load_tensor("dump/fc_1.relu", batch_size, hidden_size))
    torch.testing.assert_close(fc_2_out, load_tensor("dump/fc_2.out", batch_size, vocab_size))
    torch.testing.assert_close(fc_2_softmax, load_tensor("dump/fc_2.softmax", batch_size, vocab_size))

    loss = F.cross_entropy(fc_2_out, Ys)
    print(loss.item())

    loss.backward()
    torch.testing.assert_close(fc_2.weight.grad, load_tensor("dump/fc_2.d_w", hidden_size, vocab_size).T)
    torch.testing.assert_close(fc_2.bias.grad, load_tensor("dump/fc_2.d_b", vocab_size))
    torch.testing.assert_close(fc_1.weight.grad, load_tensor("dump/fc_1.d_w", seq_len * emb_size, hidden_size).T)
    torch.testing.assert_close(fc_1.bias.grad, load_tensor("dump/fc_1.d_b", hidden_size))
    torch.testing.assert_close(wte.weight.grad, load_tensor("dump/d_wte", vocab_size, emb_size))


if __name__ == "__main__":
    main()
