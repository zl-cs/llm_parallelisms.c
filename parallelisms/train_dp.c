// Data parallel training loop. 

#include <mpi.h>
#include <stdlib.h>
#include "data.c"
#include "model.c"


#define rank0_printf(rank, ...) if (rank == 0) { printf(__VA_ARGS__); }


// TODO(eugen): Consider adding this functionality directly into Dataset_get_batch so 
// we can get the rank batch directly without first getting the global batch.
void Dataset_get_rank_batch(
    Dataset* self,
    int* global_Xs, 
    int* global_Ys, 
    int* Xs, 
    int* Ys, 
    int global_batch_size, 
    int rank,
    int world_size
) {
    Dataset_get_batch(self, global_Xs, global_Ys, global_batch_size);
    int local_b = 0;
    for (int b = 0; b < global_batch_size; b++) {
        if (b % world_size != rank) {
            continue;
        }

        for (int i = 0; i < self->seq_len; i ++) {
            int local_idx = local_b * self->seq_len + i;
            int global_idx = b * self->seq_len + i;
            Xs[local_idx] = global_Xs[global_idx];
        }
        Ys[local_b] = global_Ys[b];
        local_b += 1;
    }
}


void allreduce_grad(float* grad, int size, int world_size) {
    MPI_Allreduce(MPI_IN_PLACE, grad, size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    for (int i = 0; i < size; i++) {
        grad[i] = grad[i] / world_size;
    }
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

    // Create model.
    Model* model = Model_create(batch_size, seq_len, vocab_size, emb_size, hidden_size);

    // Train.
    float lr = 0.1;
    int steps = 25000;
    for (int step = 0; step < steps; step++) {
        Dataset_get_rank_batch(&train_split, global_Xs, global_Ys, Xs, Ys, global_batch_size, rank, world_size);
        float loss_acc = Model_forward(model, Xs, Ys);
        MPI_Allreduce(MPI_IN_PLACE, &loss_acc, /* count */ 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        rank0_printf(rank, "step: %d, loss %f\n", step, loss_acc / world_size);

        Model_backward(model, Xs, Ys);
        allreduce_grad(model->wte->d_embedding, Embedding_numel(model->wte), world_size);
        allreduce_grad(model->fc_1->d_weight, Linear_weight_numel(model->fc_1), world_size);
        allreduce_grad(model->fc_1->d_bias, model->fc_1->out_features, world_size);
        allreduce_grad(model->fc_2->d_weight, Linear_weight_numel(model->fc_2), world_size);
        allreduce_grad(model->fc_2->d_bias, model->fc_2->out_features, world_size);

        Model_step(model, lr);
    }

    // Validate.
    float loss_acc = 0.0f;
    int n_valid_batches = 100;
    for (int i = 0; i < n_valid_batches; i ++) {
        Dataset_get_rank_batch(&test_split, global_Xs, global_Ys, Xs, Ys, global_batch_size, rank, world_size);
        loss_acc += Model_forward(model, Xs, Ys);
    }
    MPI_Allreduce(MPI_IN_PLACE, &loss_acc, /* count */ 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    rank0_printf(rank, "Final validation loss: %f\n", loss_acc / n_valid_batches / world_size);

    // Sample.
    if (rank == 0) {
        int sample_batch_size = 1;
        int* sample_Xs = calloc(sizeof(float), batch_size * seq_len);
        int* dummy_Ys = calloc(sizeof(float), batch_size);
        for (int i = 0; i < 10; i++)  {
            Model_sample(model, sample_Xs, dummy_Ys, seq_len);
            Dataset_print_batch(sample_Xs, dummy_Ys, sample_batch_size, seq_len);
            memset(sample_Xs, 0, sizeof(float) * batch_size * seq_len);
        }
    }

    MPI_Finalize();
    return 0;
}