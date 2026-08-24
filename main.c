#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* ============================================================
   Neural Network From Scratch in C — Step 9: Train XOR
   Architecture: 2 inputs -> 2 hidden neurons -> 1 output
   ============================================================ */

#define NUM_SAMPLES   4
#define NUM_INPUTS    2
#define NUM_HIDDEN    2
#define NUM_OUTPUTS   1
#define LEARNING_RATE 0.5
#define EPOCHS        10000

/* Sigmoid activation and its derivative (in terms of sigmoid output) */
float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float sigmoid_derivative(float sigmoid_output) {
    return sigmoid_output * (1.0f - sigmoid_output);
}

/* Random weight in [-1, 1] */
float random_weight() {
    return ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}

int main() {

    srand(42); /* fixed seed so runs are reproducible */

    /* ---- Training data: XOR truth table ---- */
    float inputs[NUM_SAMPLES][NUM_INPUTS] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };
    float targets[NUM_SAMPLES] = {0, 1, 1, 0};

    /* ---- Weights & biases ---- */
    /* input -> hidden */
    float w_hidden[NUM_INPUTS][NUM_HIDDEN];
    float b_hidden[NUM_HIDDEN];

    /* hidden -> output */
    float w_output[NUM_HIDDEN][NUM_OUTPUTS];
    float b_output[NUM_OUTPUTS];

    /* Initialize weights/biases randomly */
    for (int i = 0; i < NUM_INPUTS; i++)
        for (int h = 0; h < NUM_HIDDEN; h++)
            w_hidden[i][h] = random_weight();

    for (int h = 0; h < NUM_HIDDEN; h++)
        b_hidden[h] = random_weight();

    for (int h = 0; h < NUM_HIDDEN; h++)
        for (int o = 0; o < NUM_OUTPUTS; o++)
            w_output[h][o] = random_weight();

    for (int o = 0; o < NUM_OUTPUTS; o++)
        b_output[o] = random_weight();

    /* ---- Training loop ---- */
    for (int epoch = 0; epoch <= EPOCHS; epoch++) {

        float total_loss = 0.0f;

        for (int s = 0; s < NUM_SAMPLES; s++) {

            /* ===== FORWARD PROPAGATION ===== */
            float hidden_out[NUM_HIDDEN];
            for (int h = 0; h < NUM_HIDDEN; h++) {
                float z = b_hidden[h];
                for (int i = 0; i < NUM_INPUTS; i++)
                    z += inputs[s][i] * w_hidden[i][h];
                hidden_out[h] = sigmoid(z);
            }

            float output[NUM_OUTPUTS];
            for (int o = 0; o < NUM_OUTPUTS; o++) {
                float z = b_output[o];
                for (int h = 0; h < NUM_HIDDEN; h++)
                    z += hidden_out[h] * w_output[h][o];
                output[o] = sigmoid(z);
            }

            /* ===== LOSS (Mean Squared Error) ===== */
            float error = targets[s] - output[0];
            total_loss += error * error;

            /* ===== BACKPROPAGATION ===== */

            /* Output layer delta */
            float delta_output[NUM_OUTPUTS];
            for (int o = 0; o < NUM_OUTPUTS; o++) {
                float err = targets[s] - output[o];
                delta_output[o] = err * sigmoid_derivative(output[o]);
            }

            /* Hidden layer delta */
            float delta_hidden[NUM_HIDDEN];
            for (int h = 0; h < NUM_HIDDEN; h++) {
                float err = 0.0f;
                for (int o = 0; o < NUM_OUTPUTS; o++)
                    err += delta_output[o] * w_output[h][o];
                delta_hidden[h] = err * sigmoid_derivative(hidden_out[h]);
            }

            /* ===== GRADIENT DESCENT: update weights & biases ===== */

            /* hidden -> output weights */
            for (int h = 0; h < NUM_HIDDEN; h++)
                for (int o = 0; o < NUM_OUTPUTS; o++)
                    w_output[h][o] += LEARNING_RATE * delta_output[o] * hidden_out[h];

            for (int o = 0; o < NUM_OUTPUTS; o++)
                b_output[o] += LEARNING_RATE * delta_output[o];

            /* input -> hidden weights */
            for (int i = 0; i < NUM_INPUTS; i++)
                for (int h = 0; h < NUM_HIDDEN; h++)
                    w_hidden[i][h] += LEARNING_RATE * delta_hidden[h] * inputs[s][i];

            for (int h = 0; h < NUM_HIDDEN; h++)
                b_hidden[h] += LEARNING_RATE * delta_hidden[h];
        }

        if (epoch % 1000 == 0)
            printf("Epoch %5d | Loss: %f\n", epoch, total_loss / NUM_SAMPLES);
    }

    /* ---- Final predictions ---- */
    printf("\n--- Final predictions after training ---\n");
    for (int s = 0; s < NUM_SAMPLES; s++) {

        float hidden_out[NUM_HIDDEN];
        for (int h = 0; h < NUM_HIDDEN; h++) {
            float z = b_hidden[h];
            for (int i = 0; i < NUM_INPUTS; i++)
                z += inputs[s][i] * w_hidden[i][h];
            hidden_out[h] = sigmoid(z);
        }

        float output = b_output[0];
        for (int h = 0; h < NUM_HIDDEN; h++)
            output += hidden_out[h] * w_output[h][0];
        output = sigmoid(output);

        printf("Input: (%.0f, %.0f) -> Predicted: %f | Target: %.0f\n",
               inputs[s][0], inputs[s][1], output, targets[s]);
    }

    return 0;
}
