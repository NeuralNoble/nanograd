from nanograd.engine import Value
from nanograd.nn import MLP,Layer,Neuron
from nanograd.graph import draw_dot,save_graph
from nanograd.optim import SGD

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]

ys = [1.0, -1.0, -1.0, 1.0]

model = MLP(3,[4,4,1])

optim = SGD(model.params(),lr=0.01)

epochs = 30
for epoch in range(epochs):
    total_loss = 0.0
    for i in range(len(xs)):  # Loop through each training example
        x_sample = xs[i]
        y_sample = ys[i]

        y_pred = model(x_sample)  # Forward pass
        loss = (y_pred - y_sample) ** 2  # Loss function (MSE)

        optim.zero_grad()  # Reset gradients
        loss.backward()  # Backpropagation
        optim.step()  # Update parameters

        total_loss += loss.data  # Accumulate loss for logging

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(xs)}")
