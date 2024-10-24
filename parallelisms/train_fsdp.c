// Fully-sharded data parallel (i.e. Deepspeed ZeRO [1]) training loop. 
//
// Supports:
//     * Gradient sharding (i.e. ZeRO stage 2)
//     * Model parameter sharding (i.e. ZeRO stage 3)
// Optimizer parameter sharding is not (currently) supported because we use SGD.
//
// [1]: https://arxiv.org/abs/1910.02054


#include <mpi.h>
#include <stdlib.h>
#include "data.c"
#include "distributed.c"
#include "model.c"


// TODO(eugen): Consider sharding the bias as well, but usually not large enough to matter.
Model* Model_create_rank_shard(
    int batch_size, int seq_len, int vocab_size, int emb_size, int hidden_size, int rank, int world_size
) {
    // Hack! We first construct the full model then shard the parameters. This is just to 
    // ensure that the model parameters are initialized in the exact same way as the single-threaded
    // training loop for easy comparision. In practice, this approach would OOM for large models.
    Model* self = Model_create(batch_size, seq_len, vocab_size, emb_size, hidden_size);

    // Pad vocab size to be divisible by world_size.
    int vocab_size_padded = vocab_size + (world_size - (vocab_size % world_size));
    rank0_printf(rank, "Padded vocab size: %d\n", vocab_size_padded);
    // Hack! We manually construct the padded embedding instead of using vocab_size_padded in
    // Model_create above. This ensures that the RNG state matches the single-threaded training
    // loop for easy comparison.
    float* wte_padded = calloc(sizeof(float), vocab_size_padded * emb_size);
    memcpy(wte_padded, self->wte->embedding, sizeof(float) * Embedding_numel(self->wte));
    float* wte_d_padded = calloc(sizeof(float), vocab_size_padded * emb_size);
    free(self->wte->embedding);
    free(self->wte->d_embedding);
    self->wte->embedding = wte_padded;
    self->wte->d_embedding = wte_d_padded;
    self->wte->vocab_size = vocab_size_padded;

    // Shard wte.
    int wte_shard_size = Embedding_numel(self->wte) / world_size;
    float* wte_shard = malloc(sizeof(float) * wte_shard_size);
    float* wte_d_shard = calloc(sizeof(float), wte_shard_size);
    memcpy(wte_shard, self->wte->embedding + (rank * wte_shard_size), sizeof(float) * wte_shard_size);
    free(self->wte->embedding);
    self->wte->embedding = wte_shard;
    self->wte->d_embedding = wte_d_shard;
    self->wte->vocab_size = self->wte->vocab_size / world_size;

    // Shard fc_1.
    int fc_1_shard_size = Linear_weight_numel(self->fc_1) / world_size;
    float* fc_1_shard = malloc(sizeof(float) * fc_1_shard_size);
    float* fc_1_d_shard = calloc(sizeof(float), fc_1_shard_size);
    memcpy(fc_1_shard, self->fc_1->weight + (rank * fc_1_shard_size), sizeof(float) * fc_1_shard_size);
    free(self->fc_1->weight);
    free(self->fc_1->d_weight);
    self->fc_1->weight = fc_1_shard;
    self->fc_1->d_weight = fc_1_d_shard;
    self->fc_1->in_features = self->fc_1->in_features / world_size;

    // Shard fc_2.
    int fc_2_shard_size = Linear_weight_numel(self->fc_2) / world_size;
    float* fc_2_shard = malloc(sizeof(float) * fc_2_shard_size);
    float* fc_2_d_shard = calloc(sizeof(float), fc_2_shard_size);
    memcpy(fc_2_shard, self->fc_2->weight + (rank * fc_2_shard_size), sizeof(float) * fc_2_shard_size);
    free(self->fc_2->weight);
    free(self->fc_2->d_weight);
    self->fc_2->weight = fc_2_shard;
    self->fc_2->d_weight = fc_2_d_shard;
    self->fc_2->in_features = self->fc_2->in_features / world_size;

    return self;
}


// Forward for the fully sharded MLP. For each layer:
//   1. Materialize the full parameters by allgathering them.
//   2. Temporarily update pointers to point to the full parameters.
//   3. Compute the forward pass.
//   4. Revert pointers to point to the sharded parameters.
float Model_forward_fsdp(Model* self, int* Xs, int* Ys, float* flat_buffer, int world_size) {
    // wte forward.
    int wte_shard_size = Embedding_numel(self->wte);
    allgather(self->wte->embedding, wte_shard_size, flat_buffer);
    float* wte_shard = self->wte->embedding;
    int wte_shard_vocab_size = self->wte->vocab_size;
    self->wte->embedding = flat_buffer;
    self->wte->vocab_size = wte_shard_vocab_size * world_size;
    Embedding_forward(self->wte, Xs, self->wte_out);
    self->wte->embedding = wte_shard;
    self->wte->vocab_size = wte_shard_vocab_size; 

    // fc_1 forward.
    int fc_1_shard_size = Linear_weight_numel(self->fc_1);
    allgather(self->fc_1->weight, fc_1_shard_size, flat_buffer);
    float* fc_1_shard = self->fc_1->weight;
    int fc_1_shard_in_features = self->fc_1->in_features;
    self->fc_1->weight = flat_buffer;
    self->fc_1->in_features = fc_1_shard_in_features * world_size;
    Linear_forward(self->fc_1, self->wte_out_flat, self->fc_1_out);
    self->fc_1->weight = fc_1_shard;
    self->fc_1->in_features = fc_1_shard_in_features;

    relu(self->fc_1_out, self->relu_out);

    // fc_2 forward.
    int fc_2_shard_size = Linear_weight_numel(self->fc_2);
    allgather(self->fc_2->weight, fc_2_shard_size, flat_buffer);
    float* fc_2_shard = self->fc_2->weight;
    int fc_2_shard_in_features = self->fc_2->in_features;
    self->fc_2->weight = flat_buffer;
    self->fc_2->in_features = fc_2_shard_in_features * world_size;
    Linear_forward(self->fc_2, self->relu_out, self->fc_2_out);
    self->fc_2->weight = fc_2_shard;
    self->fc_2->in_features = fc_2_shard_in_features;

    softmax(self->fc_2_out, self->softmax_out);
    return cross_entropy_loss(self->softmax_out, Ys);
}


// Backward for the fully sharded MLP. For each layer:
//   1. Materialize the full parameters by allgathering them.
//   2. Temporarily update pointers to point to the full parameters / gradients.
//   3. Compute the backward pass.
//   4. Reduce scatter the gradients. 
//   5. Revert pointers to point to the sharded parameters / gradients.
void Model_backward_fsdp(Model* self, int* Xs, int* Ys, float* flat_buffer, int world_size) {
    Model_zerograd(self);
    cross_entropy_softmax_backward(self->fc_2_out, self->softmax_out, Ys);

    // fc_2 backward.
    int fc_2_shard_size = Linear_weight_numel(self->fc_2);
    int fc_2_size = fc_2_shard_size * world_size;
    memset(flat_buffer, 0, sizeof(float) * 2 * fc_2_size);
    allgather(self->fc_2->weight, fc_2_shard_size, flat_buffer);
    float* fc_2_shard = self->fc_2->weight;
    float* fc_2_d_shard = self->fc_2->d_weight;
    int fc_2_shard_in_features = self->fc_2->in_features;
    self->fc_2->weight = flat_buffer;
    self->fc_2->d_weight = flat_buffer + fc_2_size;
    self->fc_2->in_features = fc_2_shard_in_features * world_size;
    Linear_backward(self->fc_2, self->relu_out, self->fc_2_out);
    reducescatter_mean(flat_buffer + fc_2_size, fc_2_d_shard, fc_2_shard_size, world_size);
    self->fc_2->weight = fc_2_shard;
    self->fc_2->d_weight = fc_2_d_shard;
    self->fc_2->in_features = fc_2_shard_in_features;

    relu_backward(self->fc_1_out, self->relu_out);

    // fc_1 backward.
    int fc_1_shard_size = Linear_weight_numel(self->fc_1);
    int fc_1_size = fc_1_shard_size * world_size;
    memset(flat_buffer, 0, sizeof(float) * 2 * fc_1_size);
    allgather(self->fc_1->weight, fc_1_shard_size, flat_buffer);
    float* fc_1_shard = self->fc_1->weight;
    float* fc_1_d_shard = self->fc_1->d_weight;
    int fc_1_shard_in_features = self->fc_1->in_features;
    self->fc_1->weight = flat_buffer;
    self->fc_1->d_weight = flat_buffer + fc_1_size;
    self->fc_1->in_features = fc_1_shard_in_features * world_size;
    Linear_backward(self->fc_1, self->wte_out_flat, self->fc_1_out);
    reducescatter_mean(flat_buffer + fc_1_size, fc_1_d_shard, fc_1_shard_size, world_size);
    self->fc_1->weight = fc_1_shard;
    self->fc_1->d_weight = fc_1_d_shard;
    self->fc_1->in_features = fc_1_shard_in_features;

    // wte backward.
    int wte_shard_size = Embedding_numel(self->wte);
    int wte_size = wte_shard_size * world_size;
    memset(flat_buffer, 0, sizeof(float) * 2 * wte_size);
    allgather(self->wte->embedding, wte_shard_size, flat_buffer);
    float* wte_shard = self->wte->embedding;
    float* wte_d_shard = self->wte->d_embedding;
    int wte_shard_vocab_size = self->wte->vocab_size;
    self->wte->embedding = flat_buffer;
    self->wte->d_embedding = flat_buffer + wte_size;
    self->wte->vocab_size = wte_shard_vocab_size * world_size;
    Embedding_backward(self->wte, Xs, self->wte_out);
    reducescatter_mean(flat_buffer + wte_size, wte_d_shard, wte_shard_size, world_size);
    self->wte->embedding = wte_shard;
    self->wte->d_embedding = wte_d_shard;
    self->wte->vocab_size = wte_shard_vocab_size; 
}


void Model_sample_fsdp(Model* self, int* Xs, int* Ys, float* flat_buffer, int world_size, int seq_len) {
    bool done = false;
    while (!done) {
        Model_forward_fsdp(self, Xs, Ys, flat_buffer, world_size);
        int tok = Model_sample_token(self);
        // TODO(eugen): In theory, the model output and the RNG state should be 
        // identical across all ranks and hence "tok" should also be identical. However, 
        // this is not always the case in practice (possibly due to rank communication order) 
        // which can lead to MPI hangs if some ranks sample <BOS> before others. To overcome
        // this issue, we broadcast the sampled token from rank 0 to all other ranks.
        MPI_Bcast(&tok, 1, MPI_INT, 0, MPI_COMM_WORLD);
        done = Model_sample_update_input(Xs, Ys, tok, seq_len);
    }
}


int max(int a, int b) {
    return a >= b ? a : b;
}


int main(int argc, char** argv) {
    int global_batch_size = 32;
    int seq_len = 16;  // seq_len is computed offline and is equal to the longest word.
    int vocab_size = 27;
    int emb_size = 16;
    int hidden_size = 4 * emb_size;

    // Initialize environment. 
    srand(42);
    MPI_Init(&argc, &argv);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (emb_size % world_size != 0) {
        rank0_printf(rank, "Embedding size must be divisible by world size!\n");
        exit(1);
    }
    if (hidden_size % world_size != 0) {
        rank0_printf(rank, "Hidden dimension size must be divisible by world size!\n");
        exit(1);
    }

    // Compute per-rank batch size from the global batch size.
    if (global_batch_size % world_size != 0) {
        rank0_printf(rank, "Global batch size must be divisible by world size!\n");
        exit(1);
    }
    int batch_size = global_batch_size / world_size;
    rank0_printf(rank, "Micro batch_size: %d\n", batch_size); 

    // Create dataset.
    Dataset* dataset = Dataset_create_from_file("data/names.txt", seq_len);
    Dataset train_split, test_split;
    Dataset_train_test_split(dataset, &train_split, &test_split, /* train_percent */ 0.9);
    int* global_Xs = malloc(sizeof(int) * global_batch_size * seq_len);
    int* global_Ys = malloc(sizeof(int) * global_batch_size);
    int* Xs = malloc(sizeof(int) * batch_size * seq_len);
    int* Ys = malloc(sizeof(int) * batch_size);

    // Create model shard and temporary buffer to store allgathered params/grads of individual layers.
    Model* model = Model_create_rank_shard(
        batch_size, seq_len, vocab_size, emb_size, hidden_size, rank, world_size
    );
    int max_layer_size = 0;
    max_layer_size = max(Embedding_numel(model->wte) * world_size, max_layer_size);
    max_layer_size = max(Linear_weight_numel(model->fc_1) * world_size, max_layer_size);
    max_layer_size = max(Linear_weight_numel(model->fc_2) * world_size, max_layer_size);
    rank0_printf(rank, "Maximum layer size: %d\n", max_layer_size);
    float* flat_buffer = malloc(sizeof(float) * 2 * max_layer_size);  // Account for gradients.

    // Train.
    float lr = 0.1f;
    int steps = 25000;
    for (int step = 0; step < steps; step++) {
        Dataset_get_rank_batch(&train_split, global_Xs, global_Ys, Xs, Ys, global_batch_size, rank, world_size);
        float loss = Model_forward_fsdp(model, Xs, Ys, flat_buffer, world_size);
        MPI_Allreduce(MPI_IN_PLACE, &loss, /* count */ 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        rank0_printf(rank, "step: %d, loss %f\n", step, loss / world_size);
        Model_backward_fsdp(model, Xs, Ys, flat_buffer, world_size);
        Model_step(model, lr);
    }

    // Validate.
    float loss = 0.0f;
    int n_valid_batches = 100;
    for (int i = 0; i < n_valid_batches; i ++) {
        Dataset_get_rank_batch(&test_split, global_Xs, global_Ys, Xs, Ys, global_batch_size, rank, world_size);
        loss += Model_forward_fsdp(model, Xs, Ys, flat_buffer, world_size);
    }
    MPI_Allreduce(MPI_IN_PLACE, &loss, /* count */ 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    rank0_printf(rank, "Final validation loss: %f\n", loss / n_valid_batches / world_size);

    // Sample.
    int sample_batch_size = 1;
    int* sample_Xs = calloc(sizeof(float), batch_size * seq_len);
    int* dummy_Ys = calloc(sizeof(float), batch_size);
    for (int i = 0; i < 10 ; i++)  {
        Model_sample_fsdp(model, sample_Xs, dummy_Ys, flat_buffer, world_size, seq_len);
        if (rank == 0) {
            Dataset_print_batch(sample_Xs, dummy_Ys, sample_batch_size, seq_len);
        }
        memset(sample_Xs, 0, sizeof(float) * batch_size * seq_len);
    }

    MPI_Finalize();
    return 0;
}