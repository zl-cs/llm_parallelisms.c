#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>


typedef struct {
    int in_features;
    int out_features;
    float* weight;
    float* d_weight;
    float* bias;
    float* d_bias;
} Linear;


float he_init(float k) {
    float uniform = (float)rand() / RAND_MAX;	
    return 2 * k * uniform - k;
}


Linear* Linear_create(int in_features, int out_features) {
    float* weight = malloc(sizeof(float) * in_features * out_features);
    float* d_weight = malloc(sizeof(float) * in_features * out_features);
    float* bias = malloc(sizeof(float) * out_features);
    float* d_bias = malloc(sizeof(float) * out_features);

    // Initalize weights, biases, and gradients.
    float k = 1.0f / (float)in_features;
    for (int i = 0; i < in_features * out_features; i++) {
        weight[i] = he_init(k);
        d_weight[i] = 0.0f;
    }
    for (int i = 0; i < out_features; i ++){
        bias[i] = he_init(k);
        d_bias[i] = 0.0f;
    }
 

    Linear* self = malloc(sizeof(Linear));
    self->in_features = in_features;
    self->out_features = out_features;
    self->weight = weight;
    self->d_weight = d_weight;
    self->bias = bias;
    self->d_bias = d_bias;

   return self;
}


void Linear_forward(Linear* self, float* input, float* output, int batch_size) {
    // Y = X @ W + B
    for (int b = 0; b < batch_size; b++) {
        for (int o = 0; o < self->out_features; o++) {
            float sum = 0.0f;
            for (int i = 0; i < self->in_features; i++) {
                int input_idx = b * self->in_features + i;
                int weight_idx = i * self->out_features + o;
                sum += input[input_idx] * self->weight[weight_idx];
            }
            output[b * self->out_features + o] = sum + self->bias[o];
        }
    }
}


void Linear_backward(Linear* self, float* input, float* d_input, float* d_output, int batch_size) {
    // dL/dX = dL/dY @ W.T 
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < self->in_features; i++) {
            float sum = 0.0f;
            for (int o = 0; o < self->out_features; o++) {
                int d_output_idx = b * self->out_features + o;
                int weight_idx = i * self->out_features + o;
                sum += d_output[d_output_idx] * self->weight[weight_idx];
            }
            d_input[b * self->in_features + i] = sum;
        }
    }

    // dL/dW = X.T @ dL/dY
    for (int i = 0; i < self->in_features; i++) {
        for (int o = 0; o < self->out_features; o++) {
            float sum = 0.0f;
            for (int b = 0; b < batch_size; b++) {
                int input_idx = b * self->in_features + i;
                int d_output_idx = b * self->out_features + o;
                sum += input[input_idx] * d_output[d_output_idx];
            }
            self->d_weight[i * self->out_features + o] = sum;
        }
    }

    // TODO(eugen): This might be more efficient to above with the dL/dW calculation
    // since the memory is already loaded.
    // dL/db = 1 @ dL/dY
    for (int o = 0; o < self->out_features; o++) {
        float sum = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            sum += d_output[b * self->out_features + o];
        }
        self->d_bias[o] = sum;
    }
}


void relu(float* input, float* output, int size) {
    for (int i = 0; i < size; i++) {
        output[i] = input[i] > 0 ? input[i] : 0.0f;
    }
}


void relu_backward(float* input, float* d_input, float* d_output, int size) {
    for (int i = 0; i < size; i++) {
        d_input[i] = input[i] > 0 ? d_output[i] * 1.0f : 0.0f;
    }
}


void softmax(float* logits, float* probs, int batch_size, int vocab_size) {
    for (int b = 0; b < batch_size; b++) {
        // Find the max value of the row.
        float max_value = -INFINITY;
        float Z = 0;
        for (int v = 0; v < vocab_size; v++) {
            int idx = b * vocab_size + v;
            float prev_max = max_value;
            max_value = logits[idx] > max_value ? logits[idx] : max_value;
            Z *= exp(prev_max - max_value);
            Z += exp(logits[idx] - max_value);
        }
        // Compute stable softmax.
        for (int v = 0; v < vocab_size; v++) {
            int idx = b * vocab_size + v;
            probs[idx] = exp(logits[idx] - max_value) / Z;
        }
    }
}


float cross_entropy_loss(float* probs, int* target, int batch_size, int vocab_size) {
    float loss = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        int idx = b * vocab_size + target[b];
        loss += log(probs[idx]);
    }
    return -(loss / batch_size);
}


void cross_entropy_softmax_backward(
    float* probs, int* target, float* d_logits, int batch_size, int vocab_size
) {
    float d_loss = 1.0f / batch_size;
    for (int b = 0; b < batch_size; b++) {
        for (int v = 0; v < vocab_size; v++) {
            int idx = b * vocab_size + v;
            int indicator = v == target[b] ? 1.0f : 0.0f;
            d_logits[idx] += d_loss * (probs[idx] - indicator);
        }
    }
}


void dump_float_tensor(const char* filename, float* tensor, int size) {
    FILE* file = fopen(filename, "wb");
    if (file == NULL) {
        printf("Error opening file!\n");
        return;
    }
    fwrite(tensor, sizeof(float), size, file);
}


void dump_int_tensor(const char* filename, int* tensor, int size) {
    FILE* file = fopen(filename, "wb");
    if (file == NULL) {
        printf("Error opening file!\n");
        return;
    }
    fwrite(tensor, sizeof(int), size, file);
}


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    printf("Hello from rank %d of %d\n", world_rank, world_size);

    MPI_Finalize();

    if (world_rank == 0) {
        int batch_size = 5;
        int emb_size = 30;
        int hidden_size = 500;
        int vocab_size = 10;

        // Create input.
        float* input = malloc(sizeof(float) * batch_size * emb_size);
        for (int i = 0; i < batch_size * emb_size; i++) {
            input[i] = he_init(1.0);
        }
        dump_float_tensor("dump/input", input, batch_size * emb_size);

        // Create output.
        int* target = malloc(sizeof(int) * batch_size);
        for (int i = 0; i < batch_size; i++) {
            target[i] = rand() % vocab_size;
        }
        dump_int_tensor("dump/target", target, batch_size);

        // Create network.
        Linear* fc_1 = Linear_create(emb_size, hidden_size);
        Linear* fc_2 = Linear_create(hidden_size, vocab_size);
        dump_float_tensor("dump/fc_1.w", fc_1->weight, fc_1->in_features * fc_1->out_features);
        dump_float_tensor("dump/fc_1.b", fc_1->bias, fc_1->out_features);
        dump_float_tensor("dump/fc_2.w", fc_2->weight, fc_2->in_features * fc_2->out_features);
        dump_float_tensor("dump/fc_2.b", fc_2->bias, fc_2->out_features);

        // ========= Forward Pass =========
        float* fc_1_out = malloc(sizeof(float) * batch_size * hidden_size);
        Linear_forward(fc_1, input, fc_1_out, batch_size);
        dump_float_tensor("dump/fc_1.out", fc_1_out, batch_size * hidden_size);

        float* relu_out = malloc(sizeof(float) * batch_size * hidden_size);
        relu(fc_1_out, relu_out, batch_size * hidden_size);
        dump_float_tensor("dump/fc_1.relu", relu_out, batch_size * hidden_size);

        float* fc_2_out = malloc(sizeof(float) * batch_size * vocab_size);
        Linear_forward(fc_2, relu_out, fc_2_out, batch_size);
        dump_float_tensor("dump/fc_2.out", fc_2_out, batch_size * vocab_size);

        float* softmax_out = malloc(sizeof(float) * batch_size * vocab_size);
        softmax(fc_2_out, softmax_out, batch_size, vocab_size);
        dump_float_tensor("dump/fc_2.softmax", softmax_out, batch_size * vocab_size);

        float loss = cross_entropy_loss(softmax_out, target, batch_size, vocab_size);
        printf("Loss: %f\n", loss);

        // ========= Backward Pass =========
        float* d_fc_2_out = calloc(sizeof(float), batch_size * vocab_size);
        cross_entropy_softmax_backward(softmax_out, target, d_fc_2_out, batch_size, vocab_size);
        float* d_relu_out = calloc(sizeof(float), batch_size * hidden_size);
        Linear_backward(fc_2, relu_out, d_relu_out, d_fc_2_out, batch_size);
        dump_float_tensor("dump/fc_2.d_w", fc_2->d_weight, fc_2->in_features * fc_2->out_features);
        dump_float_tensor("dump/fc_2.d_b", fc_2->d_bias, fc_2->out_features);

        float* d_fc_1_out = calloc(sizeof(float), batch_size * hidden_size);
        relu_backward(fc_1_out, d_fc_1_out, d_relu_out, batch_size * hidden_size);
        float* d_input = calloc(sizeof(float), batch_size * emb_size);
        Linear_backward(fc_1, input, d_input, d_fc_1_out, batch_size);
        dump_float_tensor("dump/fc_1.d_w", fc_1->d_weight, fc_1->in_features * fc_1->out_features);
        dump_float_tensor("dump/fc_1.d_b", fc_1->d_bias, fc_1->out_features);
    }

    return 0;
}
