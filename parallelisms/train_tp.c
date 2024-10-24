// Tensor parallel (i.e. Megatron-LM [1]) training loop.
//
// [1]: https://arxiv.org/abs/1909.08053


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

    // Shard fc_1 to be column parallel across ranks.
    int fc_1_shard_cols = self->fc_1->out_features / world_size;
    float* fc_1_weight_shard = malloc(sizeof(float) * self->fc_1->in_features * fc_1_shard_cols);
    for (int row = 0; row < self->fc_1->in_features; row++) {
        int shard_offset = row * fc_1_shard_cols;
        int weight_offset = row * self->fc_1->out_features + rank * fc_1_shard_cols;
        memcpy(fc_1_weight_shard + shard_offset, self->fc_1->weight + weight_offset, fc_1_shard_cols);
    }
    float* fc_1_d_weight_shard = calloc(sizeof(float), self->fc_1->in_features * fc_1_shard_cols);
    float* fc_1_bias_shard = malloc(sizeof(float) * fc_1_shard_cols);
    memcpy(fc_1_bias_shard, self->fc_1->bias + rank * fc_1_shard_cols, fc_1_shard_cols);
    float* fc_1_d_bias_shard = calloc(sizeof(float), fc_1_shard_cols);
    free(self->fc_1->weight); self->fc_1->weight = fc_1_weight_shard;
    free(self->fc_1->d_weight); self->fc_1->d_weight = fc_1_d_weight_shard;
    free(self->fc_1->bias); self->fc_1->bias = fc_1_bias_shard;
    free(self->fc_1->d_bias); self->fc_1->d_bias = fc_1_d_bias_shard;
    self->fc_1->out_features = fc_1_shard_cols;

    // Shard fc_2 to be row parallel.
    int fc_2_shard_rows = self->fc_2->in_features / world_size;
    int fc_2_weight_shard_size = Linear_weight_numel(self->fc_2) / world_size;
    float* fc_2_weight_shard = malloc(sizeof(float) * fc_2_weight_shard_size);
    memcpy(fc_2_weight_shard, self->fc_2->weight + rank * fc_2_weight_shard_size, fc_2_weight_shard_size);
    float* fc_2_d_weight_shard = calloc(sizeof(float), fc_2_weight_shard_size);
    free(self->fc_2->weight); self->fc_2->weight = fc_2_weight_shard;
    free(self->fc_2->d_weight); self->fc_2->d_weight = fc_2_d_weight_shard;
    self->fc_2->in_features = fc_2_shard_rows;

    // Update activation shapes to match.
    Activation_destory(self->fc_1_out);
    Activation_destory(self->relu_out);
    self->fc_1_out = Activation_create(batch_size, fc_1_shard_cols);
    self->relu_out= Activation_create(batch_size, fc_1_shard_cols);

    return self;
}


// TODO(eugen): This is almost identical to the single-threaded model forward. It might
// be possible to merge some of this code.
float Model_forward_tp(Model* self, int* Xs, int* Ys, int world_size) {
    Embedding_forward(self->wte, Xs, self->wte_out);
    Linear_forward(self->fc_1, self->wte_out_flat, self->fc_1_out);
    relu(self->fc_1_out, self->relu_out);
    Linear_forward(self->fc_2, self->relu_out, self->fc_2_out);
    allreduce_mean(self->fc_2_out->value, Activation_numel(self->fc_2_out), world_size);
    softmax(self->fc_2_out, self->softmax_out);
    return cross_entropy_loss(self->softmax_out, Ys);
}


// TODO(eugen): This is almost identical to the single-threaded model backward. It might
// be possible to merge some of this code.
void Model_backward_tp(Model* self, int* Xs, int* Ys, int world_size) {
    Model_zerograd(self);
    cross_entropy_softmax_backward(self->fc_2_out, self->softmax_out, Ys);
    Linear_backward(self->fc_2, self->relu_out, self->fc_2_out);
    relu_backward(self->fc_1_out, self->relu_out);
    Linear_backward(self->fc_1, self->wte_out_flat, self->fc_1_out);
    allreduce_mean(self->wte_out_flat->d_value, Activation_numel(self->wte_out_flat), world_size);
    Embedding_backward(self->wte, Xs, self->wte_out);
}


int main(int argc, char** argv) {
    int batch_size = 32;
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

    // Create dataset.
    Dataset* dataset = Dataset_create_from_file("data/names.txt", seq_len);
    Dataset train_split, test_split;
    Dataset_train_test_split(dataset, &train_split, &test_split, /* train_percent */ 0.9);
    int* Xs = malloc(sizeof(int) * batch_size * seq_len);
    int* Ys = malloc(sizeof(int) * batch_size);

    // Create model.
    Model* model = Model_create_rank_shard(batch_size, seq_len, vocab_size, emb_size, hidden_size, rank, world_size);

    Dataset_get_batch(&train_split, Xs, Ys, batch_size);
    float loss = Model_forward_tp(model, Xs, Ys, world_size);
    rank0_printf(rank, "step: %d, loss %f\n", 0, loss);

    MPI_Finalize();
    return 0;
}