import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import time

# 가상의 간단한 Layer 정의
class Linear:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.zeros((1, output_size))

    def forward(self, x):
        return x @ self.weights + self.bias

class ReLU:
    def forward(self, x):
        return np.maximum(0, x)

class DummyModel:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

def plot_metrics(loss_history, acc_history):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(loss_history)
    ax[0].set_title("Loss")
    ax[1].plot(acc_history)
    ax[1].set_title("Accuracy")
    st.pyplot(fig)

# Streamlit UI
st.title("신경망 시각화 및 학습 플랫폼")

st.sidebar.header("Layer 구성")
layer_list = []
num_layers = st.sidebar.number_input("Layer 수", min_value=1, max_value=10, value=2)

input_size = st.sidebar.number_input("Input size", min_value=1, value=4)
last_output_size = input_size

for i in range(num_layers):
    layer_type = st.sidebar.selectbox(f"Layer {i+1} 유형", ["Linear", "ReLU"], key=f"layer_{i}")
    if layer_type == "Linear":
        output_size = st.sidebar.number_input(f"Linear Layer {i+1} output size", min_value=1, value=4, key=f"out_{i}")
        layer_list.append(Linear(last_output_size, output_size))
        last_output_size = output_size
    else:
        layer_list.append(ReLU())

model = DummyModel(layer_list)

st.header("학습 파라미터 설정")
epochs = st.number_input("Epoch 수", min_value=1, value=10)
batch_size = st.number_input("Batch size", min_value=1, value=8)

if st.button("학습 시작"):
    st.write("학습 중...")
    loss_history = deque()
    acc_history = deque()
    graph_placeholder = st.empty()

    for epoch in range(epochs):
        # 더미 입력 및 타겟
        x = np.random.randn(batch_size, input_size)
        target = np.ones((batch_size, last_output_size))
        output = model.forward(x)

        # 더미 손실 및 정확도 계산
        loss = np.mean((output - target)**2)
        acc = np.mean((np.round(output) == target))

        loss_history.append(loss)
        acc_history.append(acc)

        with graph_placeholder:
            plot_metrics(loss_history, acc_history)
        time.sleep(0.5)

    st.success("학습 완료!")

    st.markdown("""
    ### 📘 그래프 해석 가이드

    - **Loss**는 모델이 얼마나 오답을 내고 있는지를 나타냅니다. 낮을수록 좋습니다.
    - **Accuracy**는 정답률을 나타냅니다. 1에 가까울수록 좋습니다.
    - 학습이 잘 되고 있다면, Loss는 점점 감소하고, Accuracy는 증가해야 합니다.
    - 그래프가 **너무 변동이 심하거나, Loss가 줄지 않는다면** 학습률이나 Layer 구성을 다시 확인해보세요.
    """)

    st.markdown('<div id="scroll-anchor"></div>', unsafe_allow_html=True)
    st.markdown("""<script>
document.getElementById('scroll-anchor').scrollIntoView({behavior: 'smooth'});
</script>""", unsafe_allow_html=True)

if st.button("초기화"):
    st.experimental_rerun()