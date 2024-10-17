#include <stdio.h>
#include "ops.c"


void dump_float_tensor(const char* filename, float* tensor, int size) {
    FILE* file = fopen(filename, "wb");
    if (file == NULL) {
        printf("Error opening file!\n");
        exit(1);
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
    int batch_size = 5;
    int emb_size = 30;
    int hidden_size = 500;
    int vocab_size = 10;

    // Create input.
    Activation* input = Activation_create(batch_size, emb_size);
    for (int i = 0; i < batch_size * emb_size; i++) {
        input->value[i] = he_init(1.0);
    }

    // Create output.
    int* target = malloc(sizeof(int) * batch_size);
    for (int i = 0; i < batch_size; i++) {
        target[i] = rand() % vocab_size;
    }

    // Create network.
    Linear* fc_1 = Linear_create(emb_size, hidden_size);
    Linear* fc_2 = Linear_create(hidden_size, vocab_size);

    // Create activations.
    Activation* fc_1_out = Activation_create(batch_size, hidden_size);
    Activation* relu_out = Activation_create(batch_size, hidden_size);
    Activation* fc_2_out = Activation_create(batch_size, vocab_size);
    Activation* softmax_out = Activation_create(batch_size, vocab_size);

    // ========= Forward Pass =========
    Linear_forward(fc_1, input, fc_1_out);
    relu(fc_1_out, relu_out);
    Linear_forward(fc_2, relu_out, fc_2_out);
    softmax(fc_2_out, softmax_out);
    float loss = cross_entropy_loss(softmax_out, target);
    printf("Loss: %f\n", loss);

    // ========= Backward Pass =========
    cross_entropy_softmax_backward(softmax_out, fc_2_out, target);
    Linear_backward(fc_2, relu_out, fc_2_out);
    relu_backward(fc_1_out, relu_out);
    Linear_backward(fc_1, input, fc_1_out);

    // Dump weights, gradients, activations for PyTorch check.
    dump_float_tensor("dump/input", input->value, batch_size * emb_size);
    dump_int_tensor("dump/target", target, batch_size);
    dump_float_tensor("dump/fc_1.w", fc_1->weight, fc_1->in_features * fc_1->out_features);
    dump_float_tensor("dump/fc_1.b", fc_1->bias, fc_1->out_features);
    dump_float_tensor("dump/fc_2.w", fc_2->weight, fc_2->in_features * fc_2->out_features);
    dump_float_tensor("dump/fc_2.b", fc_2->bias, fc_2->out_features);
    dump_float_tensor("dump/fc_2.d_w", fc_2->d_weight, fc_2->in_features * fc_2->out_features);
    dump_float_tensor("dump/fc_2.d_b", fc_2->d_bias, fc_2->out_features);
    dump_float_tensor("dump/fc_1.d_w", fc_1->d_weight, fc_1->in_features * fc_1->out_features);
    dump_float_tensor("dump/fc_1.d_b", fc_1->d_bias, fc_1->out_features);
    dump_float_tensor("dump/fc_1.out", fc_1_out->value, Activation_numel(fc_1_out));
    dump_float_tensor("dump/fc_1.relu", relu_out->value, Activation_numel(relu_out));
    dump_float_tensor("dump/fc_2.out", fc_2_out->value, Activation_numel(fc_2_out));
    dump_float_tensor("dump/fc_2.softmax", softmax_out->value, Activation_numel(softmax_out));
}
