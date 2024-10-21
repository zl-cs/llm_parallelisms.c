#include <stdio.h>
#include <string.h>


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

