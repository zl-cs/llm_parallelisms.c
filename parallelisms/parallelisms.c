#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "ops.c"


typedef struct {
    int n_rows; 
    int seq_len;
    int* dataset;
} Dataset;


Dataset* Dataset_create_from_file(const char* filepath, int n_file_rows, int seq_len) {
    FILE* file = fopen(filepath, "r");
    if (file == NULL) {
        perror("Error opening file!\n");
        exit(1);
    }

    // The maximum possible memory we could use is n_rows * block_size * block_size 
    // if we split up one row into block_size rows. In practice we will use much less.
    int block_size = seq_len + 1;
    int* dataset = calloc(sizeof(int), n_file_rows * block_size * block_size);
    int n_rows = 0;
    char buffer[block_size];
    while(fgets(buffer, block_size, file)) {
        int n_tokens = strlen(buffer);
        while (n_tokens > 0) {
            for (int i = n_tokens - 1; i >= 0; i--) {
                // Subtract 96 so that tokens are between 1 and 27. Token 0 is used as <BOS>.
                char tok = buffer[i] == '\n' || buffer[i] == '\0' ? 0 : buffer[i] - 96;
                int offset = n_rows * block_size + block_size - n_tokens;
                dataset[offset + i] = tok;
            }
            n_tokens--;
            n_rows++;
        }
    }
    fclose(file);

    Dataset* self = malloc(sizeof(Dataset));
    self->n_rows = n_rows;
    self->seq_len = seq_len;
    self->dataset = dataset;
    return self;
}


void Dataset_get_batch(Dataset* self, int* Xs, int* Ys, int batch_size) {
    int block_size = self->seq_len + 1;
    for (int b = 0; b < batch_size; b++) {
        int idx = rand() % self->n_rows;
        // X = row[:-1]
        for (int i = 0; i < self->seq_len; i++) { 
            int data_idx = idx * block_size + i; 
            int Xs_idx = b * self->seq_len + i;
            Xs[Xs_idx] = self->dataset[data_idx];
        }
        // Y = row[-1]
        Ys[b] = self->dataset[idx * block_size + self->seq_len];
    }
}


void sgd_step(float* param, float* d_param, int size, float lr) {
    for (int i = 0; i < size; i++) {
        param[i] -= lr * d_param[i];
    }
}


void zero_grad(float* d_param, int size) {
    for (int i = 0; i < size; i++) {
        d_param[i] = 0.0f;
    }
}


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    printf("Hello from rank %d of %d\n", world_rank, world_size);
    MPI_Finalize();

    int batch_size = 32;
    int seq_len = 16;  // seq_len is computed offline and is equal to the longest word.
    int vocab_size = 27; 
    int emb_size = 16;
    int hidden_size = 4 * emb_size;

    // Create dataset.
    Dataset* dataset = Dataset_create_from_file(
        "data/names.txt", /* n_file_rows */ 32033, seq_len
    );
    int* Xs = malloc(sizeof(int) * batch_size * seq_len);
    int* Ys = malloc(sizeof(int) * batch_size);

    // Create model.
    Embedding* wte = Embedding_create(vocab_size, emb_size);
    Linear* fc_1 = Linear_create(seq_len * emb_size, hidden_size);
    Linear* fc_2 = Linear_create(hidden_size, vocab_size);
    
    // Create activations.
    Activation* wte_out = Activation_create(batch_size * seq_len, emb_size);
    Activation* wte_out_flat = Activation_view(wte_out, batch_size, seq_len * emb_size);
    Activation* fc_1_out = Activation_create(batch_size, hidden_size);
    Activation* relu_out = Activation_create(batch_size, hidden_size);
    Activation* fc_2_out = Activation_create(batch_size, vocab_size);
    Activation* softmax_out = Activation_create(batch_size, vocab_size);

    float lr = 0.01;
    int steps = 10000;
    Dataset_get_batch(dataset, Xs, Ys, batch_size);
    // print_batch(Xs, Ys, batch_size, seq_len);
    for (int step = 0; step < steps; step++) {
        // Forward pass.
        Embedding_forward(wte, Xs, wte_out);
        Linear_forward(fc_1, wte_out_flat, fc_1_out);
        relu(fc_1_out, relu_out);
        Linear_forward(fc_2, relu_out, fc_2_out);
        softmax(fc_2_out, softmax_out);
        float loss = cross_entropy_loss(softmax_out, Ys);
        printf("step: %d, loss %f\n", step, loss);

        // Zero grad.
        zero_grad(wte->d_embedding, Embedding_numel(wte));
        zero_grad(fc_1->d_weight, Linear_weight_numel(fc_1));
        zero_grad(fc_1->d_bias, fc_1->out_features);
        zero_grad(fc_2->d_weight, Linear_weight_numel(fc_2));
        zero_grad(fc_2->d_bias, fc_2->out_features);
        zero_grad(wte_out->d_value, Activation_numel(wte_out));
        zero_grad(fc_1_out->d_value, Activation_numel(fc_1_out));
        zero_grad(relu_out->d_value, Activation_numel(relu_out));
        zero_grad(fc_2_out->d_value, Activation_numel(fc_2_out));
        zero_grad(softmax_out->d_value, Activation_numel(softmax_out));

        // Backward pass.
        cross_entropy_softmax_backward(fc_2_out, softmax_out, Ys);
        Linear_backward(fc_2, relu_out, fc_2_out);
        relu_backward(fc_1_out, relu_out);
        Linear_backward(fc_1, wte_out_flat, fc_1_out);
        Embedding_backward(wte, Xs, wte_out);

        // Gradient step.
        sgd_step(wte->embedding, wte->d_embedding, Embedding_numel(wte), lr);
        sgd_step(fc_1->weight, fc_1->d_weight, Linear_weight_numel(fc_1), lr);
        sgd_step(fc_1->bias, fc_1->d_bias, fc_1->out_features, lr);
        sgd_step(fc_2->weight, fc_2->d_weight, Linear_weight_numel(fc_2), lr);
        sgd_step(fc_2->bias, fc_2->d_bias, fc_2->out_features, lr);

    }
}
