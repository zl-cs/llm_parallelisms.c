# Spelled out implementation of LLM parallelisms in Pure C

You like [makemore](https://github.com/karpathy/makemore/tree/master)? You like [llm.c](https://github.com/karpathy/llm.c)? You love `parallelisms`!

`parallelisms` implements a character-level, autoregressive language model with an MLP backbone.
The goal of this library is to demistify how to train LLMs in parallel by building everything from
scratch in C (including gradient computations). While this is not a production library, the ideas
presented here are exactly how SOTA LLMs (like [Llama 3](https://arxiv.org/abs/2407.21783)) are trained.

## :rocket: Why this library? 

- **Pure C**: Only dependency is MPI, everything else is manually implemented for maximal clarity.
- **Spelled out implementation of LLM parallelisms**:
  - Data parallel [[code](https://github.com/EugenHotaj/ml.c/blob/main/parallelisms/train_dp.c)]
  - Fully-sharded data parallel (FSDP) [[code](https://github.com/EugenHotaj/ml.c/blob/main/parallelisms/train_fsdp.c)]
  - Tensor parallel [[code](https://github.com/EugenHotaj/ml.c/blob/main/parallelisms/train_tp.c)]
  - Pipeline parallel [[code](https://github.com/EugenHotaj/ml.c/blob/main/parallelisms/train_pp.c)]
- **Advanced 3D Parallelism**: combining FSDP, tensor, and pipeline parallelism [[code](https://github.com/EugenHotaj/ml.c/blob/main/parallelisms/train_3d.c)]
- **No dynamic memory allocations**.

## :hammer_and_wrench: Technical Details

The most interesting files to look at are the `train_*.c` files. `train.c` contains the reference implementation for single-threaded training
while the rest implement the individual parallelism methods. Finally, `train_3d.c` brings everything together to implement 3d parallel training. 
The individual parallelism implementations are modular enough that 3d parallelism "falls out". `distributed.c` contains communication utilities
and certain functions that are used across multiple files.

The rest of the code implements an MLP with hand-crafted forward, backward passes and a data loader. Some notes on desing:

* The forward and backward functions have the exact same signature. This made it very convenient to know how to correctly call the backwards
functions and overall reduced the chance of bugs slipping into the code.
* C has no concept of classes, only `structs` and `functions`. We implement "methods" on structs by prefixing functions by the struct name and
always passing the struct as the first input argument named `self`. E.g. `Model_forward(Model* self, ...)` is a "method" of the `Model` struct.
* The code is optimized for understanding rather than raw performance. In some cases bad performacne decisions are made for the sake of clarity.
E.g. we always construct the full model then shard to ensure that the single-threaded and parallel parameters are initialize in exactly the same
way.
