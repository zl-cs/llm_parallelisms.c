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


void Dataset_train_test_split(Dataset* self, Dataset* train_split, Dataset* test_split, float train_percent) {
    int n_train_rows = (int)(self->n_rows * train_percent);
    int n_test_rows = self->n_rows - n_train_rows;

    train_split->n_rows = n_train_rows;
    train_split->seq_len = self->seq_len;
    train_split->dataset = self->dataset;

    test_split->n_rows = n_test_rows;
    test_split->seq_len = self->seq_len;
    test_split->dataset = self->dataset + n_train_rows * (self->seq_len + 1);
}


void Dataset_get_batch(Dataset* self, int* Xs, int* Ys, int batch_size) {
    int block_size = self->seq_len + 1;
    for (int b = 0; b < batch_size; b++) {
        int idx = rand() % self->n_rows;
        // Xs = row[:-1]
        for (int i = 0; i < self->seq_len; i++) { 
            int data_idx = idx * block_size + i; 
            int Xs_idx = b * self->seq_len + i;
            Xs[Xs_idx] = self->dataset[data_idx];
        }
        // Ys = row[-1]
        Ys[b] = self->dataset[idx * block_size + self->seq_len];
    }
}


void Dataset_print_batch(int* Xs, int* Ys, int batch_size, int seq_len) {
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++)  {
            int idx = b * seq_len + s;
            char tok = Xs[idx] != 0 ? Xs[idx] + 96 : '.';
            printf("%c ", tok);
        }
        printf(" --> %c\n", Ys[b] != 0 ? Ys[b] + 96 : '.');
    }
}


void sgd_step(float* param, float* d_param, int size, float lr) {
    for (int i = 0; i < size; i++) {
        param[i] += -lr * d_param[i];
    }
}


typedef struct {
    Embedding* wte;
    Linear* fc_1;
    Linear* fc_2;

    Activation* wte_out;
    Activation* wte_out_flat;
    Activation* fc_1_out;
    Activation* relu_out;
    Activation* fc_2_out;
    Activation* softmax_out;
} Model;


Model* Model_create(int batch_size, int seq_len, int vocab_size, int emb_size, int hidden_size) {
    // Create parameters.
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

    Model* self = malloc(sizeof(Model));
    self->wte = wte;
    self->fc_1 = fc_1;
    self->fc_2 = fc_2;
    self->wte_out = wte_out;
    self->wte_out_flat = wte_out_flat;
    self->fc_1_out = fc_1_out;
    self->relu_out = relu_out;
    self->fc_2_out = fc_2_out;
    self->softmax_out = softmax_out;
    return self;
}


float Model_forward(Model* self, int* Xs, int* Ys) {
    Embedding_forward(self->wte, Xs, self->wte_out);
    Linear_forward(self->fc_1, self->wte_out_flat, self->fc_1_out);
    relu(self->fc_1_out, self->relu_out);
    Linear_forward(self->fc_2, self->relu_out, self->fc_2_out);
    softmax(self->fc_2_out, self->softmax_out);
    return cross_entropy_loss(self->softmax_out, Ys);
}


void Model_backward(Model* self, int* Xs, int* Ys) {
        // Zero grad.
        memset(self->wte->d_embedding, 0, sizeof(float) * Embedding_numel(self->wte));
        memset(self->fc_1->d_weight, 0, sizeof(float) * Linear_weight_numel(self->fc_1));
        memset(self->fc_1->d_bias, 0, sizeof(float) * self->fc_1->out_features);
        memset(self->fc_2->d_weight, 0, sizeof(float) * Linear_weight_numel(self->fc_2));
        memset(self->fc_2->d_bias, 0, sizeof(float) * self->fc_2->out_features);
        memset(self->wte_out->d_value, 0, sizeof(float) * Activation_numel(self->wte_out));
        memset(self->fc_1_out->d_value, 0, sizeof(float) * Activation_numel(self->fc_1_out));
        memset(self->relu_out->d_value, 0, sizeof(float) * Activation_numel(self->relu_out));
        memset(self->fc_2_out->d_value, 0, sizeof(float) * Activation_numel(self->fc_2_out));
        memset(self->softmax_out->d_value, 0, sizeof(float) * Activation_numel(self->softmax_out));

        // Backward pass.
        cross_entropy_softmax_backward(self->fc_2_out, self->softmax_out, Ys);
        Linear_backward(self->fc_2, self->relu_out, self->fc_2_out);
        relu_backward(self->fc_1_out, self->relu_out);
        Linear_backward(self->fc_1, self->wte_out_flat, self->fc_1_out);
        Embedding_backward(self->wte, Xs, self->wte_out);
}


void Model_step(Model* self, float lr) {
    sgd_step(self->wte->embedding, self->wte->d_embedding, Embedding_numel(self->wte), lr);
    sgd_step(self->fc_1->weight, self->fc_1->d_weight, Linear_weight_numel(self->fc_1), lr);
    sgd_step(self->fc_1->bias, self->fc_1->d_bias, self->fc_1->out_features, lr);
    sgd_step(self->fc_2->weight, self->fc_2->d_weight, Linear_weight_numel(self->fc_2), lr);
    sgd_step(self->fc_2->bias, self->fc_2->d_bias, self->fc_2->out_features, lr);
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
    Dataset train_split, test_split;
    Dataset_train_test_split(dataset, &train_split, &test_split, /* train_percent */ 0.9);
    int* Xs = malloc(sizeof(int) * batch_size * seq_len);
    int* Ys = malloc(sizeof(int) * batch_size);

    // Train.
    float lr = 0.1;
    int steps = 25000;
    Model* model = Model_create(batch_size, seq_len, vocab_size, emb_size, hidden_size);
    for (int step = 0; step < steps; step++) {
        Dataset_get_batch(&train_split, Xs, Ys, batch_size);
        float loss = Model_forward(model, Xs, Ys);
        printf("step: %d, loss %f\n", step, loss);
        Model_backward(model, Xs, Ys);
        Model_step(model, lr);
   }

   // Validate.
   float loss_acc = 0.0f;
   int n_valid_batches = 500;
   for (int i = 0; i < n_valid_batches; i ++) {
        Dataset_get_batch(&test_split, Xs, Ys, batch_size);
        loss_acc += Model_forward(model, Xs, Ys);
   }
   printf("Final validation loss: %f\n", loss_acc / n_valid_batches);
}
