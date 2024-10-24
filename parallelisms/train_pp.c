// Pipeline parallel training loop.

#include <mpi.h>
#include <stdlib.h>
#include "data.c"
#include "distributed.c"
#include "model.c"


void send(float* input, int input_size, int to_rank) {
    MPI_Send(input, input_size, MPI_FLOAT, to_rank, 0, MPI_COMM_WORLD);
}


void recv(float* output, int output_size, int from_rank) {
    MPI_Status status;
    MPI_Recv(output, output_size, MPI_FLOAT, from_rank, 0, MPI_COMM_WORLD, &status);
}


Model* Model_create_rank_stage(
    int batch_size, int seq_len, int vocab_size, int emb_size, int hidden_size, int rank
) {
    // Hack! We first construct the full model then shard the parameters to stages. This is just to 
    // ensure that the model parameters are initialized in the exact same way as the single-threaded
    // training loop for easy comparision. In practice, this approach would OOM for large models.
    Model* self = Model_create(batch_size, seq_len, vocab_size, emb_size, hidden_size);

    if (rank == 0) {
        Linear_destroy(self->fc_1); self->fc_1 = NULL;
        Linear_destroy(self->fc_2); self->fc_2 = NULL;
        Activation_destory(self->fc_1_out); self->fc_1_out = NULL;
        Activation_destory(self->relu_out); self->relu_out = NULL;
        Activation_destory(self->fc_2_out); self->fc_2_out = NULL;
        Activation_destory(self->softmax_out); self->softmax_out = NULL;
    } else if (rank == 1) {
        Embedding_destory(self->wte); self->wte = NULL;
        Linear_destroy(self->fc_2); self->fc_2 = NULL;
        Activation_destory(self->fc_2_out); self->fc_2_out = NULL;
        Activation_destory(self->softmax_out); self->softmax_out = NULL;
    } else if (rank == 2) {
        Embedding_destory(self->wte); self->wte = NULL;
        Linear_destroy(self->fc_1); self->fc_1 = NULL;
        Activation_destory(self->wte_out); self->wte_out = NULL; self->wte_out_flat = NULL;
        Activation_destory(self->fc_1_out); self->fc_1_out = NULL;
    } else {
        printf("Unknown rank: %d\n", rank);
        MPI_Finalize();
        exit(1);
    }

    return self;
}


float Model_forward_pp(Model* self, int* Xs, int* Ys, int rank) {
    float loss;
    if (rank == 0) {
        Embedding_forward(self->wte, Xs, self->wte_out);
        send(self->wte_out->value, Activation_numel(self->wte_out), /* to_rank */ 1);
    } else if (rank == 1) {
        recv(self->wte_out_flat->value, Activation_numel(self->wte_out_flat), /* from_rank */ 0);
        Linear_forward(self->fc_1, self->wte_out_flat, self->fc_1_out);
        relu(self->fc_1_out, self->relu_out);
        send(self->relu_out->value, Activation_numel(self->relu_out), /* to_rank */ 2);
    } else if (rank == 2) {
        recv(self->relu_out->value, Activation_numel(self->relu_out), /* from_rank */ 1);
        Linear_forward(self->fc_2, self->relu_out, self->fc_2_out);
        softmax(self->fc_2_out, self->softmax_out);
        loss = cross_entropy_loss(self->softmax_out, Ys);
    } else {
        printf("Unknown rank: %d\n", rank);
        MPI_Finalize();
        exit(1);
    }
    // We don't technically need to broadcast here, but it's nicer if all the ranks have the
    // same loss value at the end.
    MPI_Bcast(&loss, /* count */ 1, MPI_FLOAT, /* root */ 2, MPI_COMM_WORLD);
    return loss;
}


void Model_backward_pp(Model* self, int* Xs, int* Ys, int rank) {
    if (rank == 2) {
        // Zero grad.
        memset(self->fc_2->d_weight, 0, sizeof(float) * Linear_weight_numel(self->fc_2));
        memset(self->fc_2->d_bias, 0, sizeof(float) * self->fc_2->out_features);
        memset(self->relu_out->d_value, 0, sizeof(float) * Activation_numel(self->relu_out));
        memset(self->fc_2_out->d_value, 0, sizeof(float) * Activation_numel(self->fc_2_out));
        memset(self->softmax_out->d_value, 0, sizeof(float) * Activation_numel(self->softmax_out));
        // Backward.
        cross_entropy_softmax_backward(self->fc_2_out, self->softmax_out, Ys);
        Linear_backward(self->fc_2, self->relu_out, self->fc_2_out);
        send(self->relu_out->d_value, Activation_numel(self->relu_out), /* to_rank */ 1);
    } else if (rank == 1) {
        // Zero grad.
        memset(self->fc_1->d_weight, 0, sizeof(float) * Linear_weight_numel(self->fc_1));
        memset(self->fc_1->d_bias, 0, sizeof(float) * self->fc_1->out_features);
        memset(self->wte_out_flat->d_value, 0, sizeof(float) * Activation_numel(self->wte_out));
        memset(self->fc_1_out->d_value, 0, sizeof(float) * Activation_numel(self->fc_1_out));
        memset(self->relu_out->d_value, 0, sizeof(float) * Activation_numel(self->relu_out));
        // Backward.
        recv(self->relu_out->d_value, Activation_numel(self->relu_out), /* from_rank */ 2);
        relu_backward(self->fc_1_out, self->relu_out);
        Linear_backward(self->fc_1, self->wte_out_flat, self->fc_1_out);
        send(self->wte_out_flat->d_value, Activation_numel(self->wte_out_flat), /* to_rank */ 0);
    } else if (rank == 0) {
        // Zero grad.
        memset(self->wte->d_embedding, 0, sizeof(float) * Embedding_numel(self->wte));
        memset(self->wte_out->d_value, 0, sizeof(float) * Activation_numel(self->wte_out));
        // Backward.
        recv(self->wte_out->d_value, Activation_numel(self->wte_out), /* from_rank */ 1);
        Embedding_backward(self->wte, Xs, self->wte_out);
    } else {
        printf("Unknown rank: %d\n", rank);
        MPI_Finalize();
        exit(1);
    }
}


void Model_step_pp(Model* self, float lr, int rank) {
    if (rank == 0) {
        sgd_step(self->wte->embedding, self->wte->d_embedding, Embedding_numel(self->wte), lr);
    } else if (rank == 1) {
        sgd_step(self->fc_1->weight, self->fc_1->d_weight, Linear_weight_numel(self->fc_1), lr);
        sgd_step(self->fc_1->bias, self->fc_1->d_bias, self->fc_1->out_features, lr);
    } else if (rank == 2) {
        sgd_step(self->fc_2->weight, self->fc_2->d_weight, Linear_weight_numel(self->fc_2), lr);
        sgd_step(self->fc_2->bias, self->fc_2->d_bias, self->fc_2->out_features, lr);
    } else {
        printf("Unknown rank: %d\n", rank);
        MPI_Finalize();
        exit(1);
    }
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
    if (world_size != 3) {
        rank0_printf(rank, "Pipeline parallelism requires world_size = 3!\n");
        MPI_Finalize();
        exit(1);
    }

    // Create dataset.
    Dataset* dataset = Dataset_create_from_file("data/names.txt", seq_len);
    Dataset train_split, test_split;
    Dataset_train_test_split(dataset, &train_split, &test_split, /* train_percent */ 0.9);
    int* Xs = malloc(sizeof(int) * batch_size * seq_len);
    int* Ys = malloc(sizeof(int) * batch_size);

    // Create model.
    Model* model = Model_create_rank_stage(batch_size, seq_len, vocab_size, emb_size, hidden_size, rank);

    // Train.
    float lr = 0.1;
    int steps = 25000;
    for (int step = 0; step < steps; step++) {
        Dataset_get_batch(&train_split, Xs, Ys, batch_size);
        float loss = Model_forward_pp(model, Xs, Ys, rank);
        rank0_printf(rank, "step: %d, loss %f\n", step, loss);
        Model_backward_pp(model, Xs, Ys, rank);
        Model_step_pp(model, lr, rank);
    }

    // Validate.
    float loss = 0.0f;
    int n_valid_batches = 100;
    for (int i = 0; i < n_valid_batches; i ++) {
        Dataset_get_batch(&test_split, Xs, Ys, batch_size);
        loss += Model_forward_pp(model, Xs, Ys, rank);
    }
    rank0_printf(rank, "Final validation loss: %f\n", loss / n_valid_batches);


    MPI_Finalize();
    return 0;
} 