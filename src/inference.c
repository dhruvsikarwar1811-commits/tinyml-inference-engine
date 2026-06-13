#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define INPUT_SIZE 784
#define HIDDEN1_SIZE 128
#define HIDDEN2_SIZE 64
#define OUTPUT_SIZE 10

typedef struct {
    float fc1_w[HIDDEN1_SIZE][INPUT_SIZE];
    float fc1_b[HIDDEN1_SIZE];
    float fc2_w[HIDDEN2_SIZE][HIDDEN1_SIZE];
    float fc2_b[HIDDEN2_SIZE];
    float fc3_w[OUTPUT_SIZE][HIDDEN2_SIZE];
    float fc3_b[OUTPUT_SIZE];
} TinyMLP;

void relu(float* x, int size) {
    for (int i = 0; i < size; i++)
        if (x[i] < 0) x[i] = 0;
}

void softmax(float* x, int size) {
    float max = x[0];
    for (int i = 1; i < size; i++) if (x[i] > max) max = x[i];
    float sum = 0;
    for (int i = 0; i < size; i++) { x[i] = expf(x[i] - max); sum += x[i]; }
    for (int i = 0; i < size; i++) x[i] /= sum;
}

void matmul(float* out, float* in, float w[][INPUT_SIZE], float* b, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        out[i] = b[i];
        for (int j = 0; j < cols; j++)
            out[i] += w[i][j] * in[j];
    }
}

void matmul2(float* out, float* in, float* w, float* b, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        out[i] = b[i];
        for (int j = 0; j < cols; j++)
            out[i] += w[i * cols + j] * in[j];
    }
}

int forward(TinyMLP* model, float* input) {
    float h1[HIDDEN1_SIZE], h2[HIDDEN2_SIZE], out[OUTPUT_SIZE];

    // Layer 1
    matmul(h1, input, model->fc1_w, model->fc1_b, HIDDEN1_SIZE, INPUT_SIZE);
    relu(h1, HIDDEN1_SIZE);

    // Layer 2
    matmul2(h2, h1, (float*)model->fc2_w, model->fc2_b, HIDDEN2_SIZE, HIDDEN1_SIZE);
    relu(h2, HIDDEN2_SIZE);

    // Layer 3
    matmul2(out, h2, (float*)model->fc3_w, model->fc3_b, OUTPUT_SIZE, HIDDEN2_SIZE);
    softmax(out, OUTPUT_SIZE);

    int pred = 0;
    for (int i = 1; i < OUTPUT_SIZE; i++)
        if (out[i] > out[pred]) pred = i;

    printf("Prediction: %d (confidence: %.2f%%)\n", pred, out[pred]*100);
    return pred;
}

int main() {
    printf("TinyML Inference Engine v1.0\n");
    printf("Pure C Neural Network - No ML Framework!\n\n");

    TinyMLP model;
    memset(&model, 0, sizeof(TinyMLP));

    // Demo with random input
    float input[INPUT_SIZE];
    for (int i = 0; i < INPUT_SIZE; i++)
        input[i] = (float)rand() / RAND_MAX;

    printf("Running inference on test input...\n");
    forward(&model, input);
    printf("\nInference complete!\n");
    return 0;
}
