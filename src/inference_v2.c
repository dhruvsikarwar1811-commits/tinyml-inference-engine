#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "weights.h"

#define INPUT_SIZE 784
#define HIDDEN1_SIZE 128
#define HIDDEN2_SIZE 64
#define OUTPUT_SIZE 10

void relu(float* x, int size) {
    for (int i = 0; i < size; i++)
        if (x[i] < 0) x[i] = 0;
}

void softmax(float* x, int size) {
    float max = x[0];
    for (int i = 1; i < size; i++) if (x[i] > max) max = x[i];
    float sum = 0;
    for (int i = 0; i < size; i++) { x[i] = expf(x[i]-max); sum += x[i]; }
    for (int i = 0; i < size; i++) x[i] /= sum;
}

void linear(float* out, float* in, const float* w, const float* b, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        out[i] = b[i];
        for (int j = 0; j < cols; j++)
            out[i] += w[i*cols+j] * in[j];
    }
}

int predict(float* input) {
    float h1[HIDDEN1_SIZE], h2[HIDDEN2_SIZE], out[OUTPUT_SIZE];
    linear(h1, input, fc1_weight, fc1_bias, HIDDEN1_SIZE, INPUT_SIZE);
    relu(h1, HIDDEN1_SIZE);
    linear(h2, h1, fc2_weight, fc2_bias, HIDDEN2_SIZE, HIDDEN1_SIZE);
    relu(h2, HIDDEN2_SIZE);
    linear(out, h2, fc3_weight, fc3_bias, OUTPUT_SIZE, HIDDEN2_SIZE);
    softmax(out, OUTPUT_SIZE);
    int pred = 0;
    for (int i = 1; i < OUTPUT_SIZE; i++)
        if (out[i] > out[pred]) pred = i;
    printf("Scores: ");
    for (int i = 0; i < OUTPUT_SIZE; i++) printf("%.2f ", out[i]);
    printf("\nPrediction: %d (%.1f%% confidence)\n", pred, out[pred]*100);
    return pred;
}

int main() {
    printf("=================================\n");
    printf(" TinyML Inference Engine v2.0\n");
    printf(" Pure C | Trained MNIST Weights\n");
    printf("=================================\n\n");

    clock_t start, end;
    float input[INPUT_SIZE];

    // Test 5 random inputs
    srand(42);
    for (int t = 0; t < 5; t++) {
        for (int i = 0; i < INPUT_SIZE; i++)
            input[i] = (float)rand()/RAND_MAX;
        start = clock();
        printf("Test %d: ", t+1);
        predict(input);
        end = clock();
        printf("Inference time: %.3f ms\n\n",
               (double)(end-start)/CLOCKS_PER_SEC*1000);
    }
    return 0;
}
