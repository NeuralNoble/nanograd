from nanograd.engine import Value
from nanograd.nn import MLP,Layer,Neuron
from nanograd.graph import draw_dot,save_graph

# Tiny dataset
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]

ys = [1.0, -1.0, -1.0, 1.0]  # Desired targets

# Initialize a single neuron with 3 inputs (size of xs)
n = Neuron(3)

# Training loop for 30 iterations
for k in range(30):
    # Forward pass: pass each input through the neuron
    ypred = [n(x) for x in xs]

    # Calculate loss (Mean Squared Error)
    loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

    # Backpropagation: clear the gradients
    for p in n.params():
        p.grad = 0.0
    loss.backward()

    # Update parameters using gradient descent
    for p in n.params():
        p.data += -0.1 * p.grad

    # Print the loss at each step
    print(f"Epoch {k + 1}, Loss: {loss.data}")

save_graph(ypred[0],"example")


