from nn import MLP
from graphviz import Digraph


x = [2.0, 3.0, -1.0]
n = MLP(3, [4, 4, 1])

# simple dataset
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]

y_true = [1.0, -1.0, -1.0, 1.0]

learning_rate = 0.01
epochs = 100


for i in range(epochs):
    y_pred = [n(x) for x in xs]
    loss = sum((yout - ygt) ** 2 for yout, ygt in zip(y_pred, y_true)) # MSE
    loss.backward()
    
    # update params of network based on gradient descent
    for p in n.parameters():
        p.data += -learning_rate * p.grad

    print(f"Loss at epoch {i} is {loss}")