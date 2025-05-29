from dezero import Model
import dezero.functions as F
import dezero.layers as L

class SimpleMLP(Model):
    def __init__(self, layer_list):
        super().__init__()
        self.layers = []
        for spec in layer_list:
            if spec["type"] == "Linear":
                out_size = spec["output_size"]
                layer = L.Linear(out_size)
            elif spec["type"] == "ReLU":
                layer = lambda x: F.relu(x)
            elif spec["type"] == "Sigmoid":
                layer = lambda x: F.sigmoid(x)
            elif spec["type"] == "Tanh":
                layer = lambda x: F.tanh(x)
            elif spec["type"] == "Dropout":
                rate = spec.get("rate", 0.5)
                layer = lambda x, rate=rate: F.dropout(x, dropout_ratio=rate)
            else:
                raise ValueError(f"Unsupported layer type: {spec}")
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __call__(self, x):
        return self.forward(x)

def build_model(layer_list):
    return SimpleMLP(layer_list)