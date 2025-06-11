# STEP61: Bayesian Dropout을 활용한 예측 불확실성 추정 실험
# 기존 DeZero는 예측값만 제공하지만, 이 스텝에서는 추론 시에도 Dropout을 활성화하여
# Monte Carlo 추론으로 예측의 신뢰구간(평균 ± 표준편차)을 계산한다.

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import dezero.functions as F
import matplotlib.pyplot as plt
from dezero import Model
from dezero import Layer
from dezero.layers import Linear

# Bayesian Dropout 정의
class BayesianDropout(Layer):
    def __init__(self, dropout_ratio=0.5):
        super().__init__()
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train=True):
        if train:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            scale = 1.0 / (1.0 - self.dropout_ratio)
            return x * self.mask * scale
        else:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            scale = 1.0 / (1.0 - self.dropout_ratio)
            return x * self.mask * scale

    def __call__(self, x, train=True):
        return self.forward(x, train=train)

# MLP 모델 정의
class MLPBayesian(Model):
    def __init__(self, hidden_size=100, out_size=1):
        super().__init__()
        self.l1 = Linear(hidden_size)
        self.dropout1 = BayesianDropout(0.05)
        self.l2 = Linear(hidden_size)
        self.dropout2 = BayesianDropout(0.05)
        self.l3 = Linear(out_size)

    def forward(self, x, train=True):
        y = self.l1(x)
        y = F.relu(y)
        y = self.dropout1(y, train=train)
        y = self.l2(y)
        y = F.relu(y)
        y = self.dropout2(y, train=train)
        y = self.l3(y)
        return y

    def __call__(self, x, train=True):
        return self.forward(x, train=train)

# 데이터 생성 (sin 곡선)
def generate_sin_data(seq_len=1000):
    x = np.linspace(0, 8 * np.pi, seq_len)
    y = np.sin(x)
    return x.reshape(-1, 1), y.reshape(-1, 1)

x_data, t_data = generate_sin_data()
x_data = (x_data - x_data.mean()) / x_data.std()

# 학습 설정
x_train, t_train = x_data[:800], t_data[:800]
x_val, t_val = x_data[800:], t_data[800:]

model = MLPBayesian(hidden_size=128, out_size=1)
lr = 0.001
iters = 10000

for i in range(iters):
    batch_index = np.random.choice(len(x_train), 16)
    x_batch = x_train[batch_index]
    t_batch = t_train[batch_index]

    y = model(x_batch, train=True)
    loss = F.mean((y - t_batch) ** 2)

    model.cleargrads()
    loss.backward()
    for p in model.params():
        p.data -= lr * p.grad.data

    if i % 300 == 0:
        print(f'Iter {i}, Loss: {loss.data:.4f}')

# Monte Carlo 추론 (불확실성 추정)
mc_predictions = []
for _ in range(50):
    y_pred = model(x_val, train=False)
    mc_predictions.append(y_pred.data)

mc_predictions = np.stack(mc_predictions)
pred_mean = mc_predictions.mean(axis=0)
pred_std = mc_predictions.std(axis=0)

# 시각화
plt.figure(figsize=(10, 4))
plt.plot(pred_mean, 'r-', label='Predicted Mean')
plt.fill_between(np.arange(len(pred_mean)),
                 pred_mean.flatten() - pred_std.flatten(),
                 pred_mean.flatten() + pred_std.flatten(),
                 color='red', alpha=0.3, label='±1 std')
plt.plot(t_val, 'b--', label='True')
plt.title('Bayesian Dropout 예측 결과 (불확실성 포함)')
plt.legend()
plt.tight_layout()
plt.show()
