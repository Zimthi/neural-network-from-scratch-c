# Neural Network From Scratch in C

A tiny feedforward neural network, written in pure C with no libraries beyond
`math.h`, trained to solve the XOR problem.

## Architecture

```
Input (2) -> Hidden (2, sigmoid) -> Output (1, sigmoid)
```

XOR isn't linearly separable, so it needs at least one hidden layer —
that's why this step exists in the roadmap.

## Build & run

```bash
make
make run
```

or manually:

```bash
gcc src/main.c -o neural_network -lm
./neural_network
```

## What it does

1. **Forward propagation** — inputs flow through the hidden layer, then the
   output layer, each neuron applying `sigmoid(weighted sum + bias)`.
2. **Loss** — mean squared error between prediction and target.
3. **Backpropagation** — computes how much each weight contributed to the
   error, working backward from the output layer to the hidden layer.
4. **Gradient descent** — nudges every weight and bias in the direction
   that reduces the error, scaled by `LEARNING_RATE`.
5. Repeats for `EPOCHS` passes over all 4 XOR examples.

## Expected output

Loss should fall from ~0.26 to under 0.001, and the final predictions
should land close to the XOR truth table:

```
(0, 0) -> ~0.02   (target 0)
(0, 1) -> ~0.98   (target 1)
(1, 0) -> ~0.98   (target 1)
(1, 1) -> ~0.02   (target 0)
```

## Next steps (per the original roadmap)

- Save/load trained weights to a file
- Generalize to arbitrary layer sizes (currently hardcoded for 2-2-1)
- Swap in a dataset like MNIST for digit recognition
