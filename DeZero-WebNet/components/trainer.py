import numpy as np
from dezero import Variable
import dezero.functions as F
import dezero.optimizers as O

def train(model, epochs=10, lr=0.01, input_dim=10, batch_size=16, return_log=False):
    x = np.random.randn(batch_size, input_dim)
    y = np.ones((batch_size, 1))  # target is all 1s

    x = Variable(x)
    y = Variable(y)

    optimizer = O.SGD(lr).setup(model)

    losses = []
    accs = []
    logs = []

    for epoch in range(epochs):
        y_pred = model(x)
        loss = F.mean_squared_error(y_pred, y)

        model.cleargrads()
        loss.backward()
        optimizer.update()

        pred_binary = (np.array(y_pred.data) > 0.5).astype(np.float32)
        acc = np.mean(pred_binary == y.data)

        losses.append(float(np.array(loss.data).flatten()[0]))
        accs.append(float(acc))
        logs.append(f"[Epoch {epoch+1}/{epochs}] Loss: {float(np.array(loss.data).flatten()[0]):.4f} | Accuracy: {acc:.4f}")

    if return_log:
        return losses, accs, logs
    return losses