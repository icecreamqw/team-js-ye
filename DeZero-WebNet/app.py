import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, "./")

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import time

from dezero import Variable
from components.model_builder import build_model
from components.trainer import train
from dz_utils import graph_visualizer
plot_model_graph = graph_visualizer.plot_model_graph

# Streamlit page setup
st.set_page_config(page_title="DeZero Demo", layout="wide")
st.title("🧠 DeZero 기반 신경망 시각화 데모")

# Session state initialization
if "layers" not in st.session_state:
    st.session_state.layers = []

# Sidebar layout
st.sidebar.title("⚙️ 설정 메뉴")
st.sidebar.header("🧱 Layer 구성")

num_layers = st.sidebar.number_input("Layer 수", min_value=1, max_value=10, value=max(len(st.session_state.layers), 1))

# Adjust session layer list
if len(st.session_state.layers) != num_layers:
    st.session_state.layers = st.session_state.layers[:num_layers]
    while len(st.session_state.layers) < num_layers:
        st.session_state.layers.append({"type": "Linear", "output_size": 4})

input_size = st.sidebar.number_input("Input size", min_value=1, value=4)
last_output_size = input_size

for i in range(num_layers):
    layer_type = st.sidebar.selectbox(f"Layer {i+1} 유형", ["Linear", "ReLU", "Sigmoid", "Tanh", "Dropout"], key=f"layer_{i}_type")
    st.session_state.layers[i]["type"] = layer_type

    if layer_type == "Linear":
        output_size = st.sidebar.number_input(f"Linear Layer {i+1} output size", min_value=1, value=st.session_state.layers[i].get("output_size", 4), key=f"out_{i}")
        st.session_state.layers[i]["output_size"] = output_size
        last_output_size = output_size
    elif layer_type == "Dropout":
        dropout_ratio = st.sidebar.slider(f"Dropout Layer {i+1} ratio", min_value=0.0, max_value=1.0, value=st.session_state.layers[i].get("dropout_ratio", 0.5), step=0.05, key=f"dropout_{i}")
        st.session_state.layers[i]["dropout_ratio"] = dropout_ratio

st.sidebar.markdown("---")
st.sidebar.header("📚 학습 파라미터")
lr = st.sidebar.number_input("Learning Rate", min_value=0.0001, max_value=1.0, value=0.01, step=0.0001, format="%.4f")
epochs = st.sidebar.number_input("Epoch 수", min_value=1, max_value=500, value=10, step=1)

# Layer display and reordering
st.subheader("🧱 현재 구성된 네트워크:")
for i, layer in enumerate(st.session_state.layers):
    cols = st.columns([6, 1, 1, 1])
    with cols[0]:
        if layer["type"] == "Linear":
            st.write(f"{i+1}. Linear({layer.get('output_size', 4)})")
        elif layer["type"] == "Dropout":
            st.write(f"{i+1}. Dropout({layer.get('dropout_ratio', 0.5)})")
        else:
            st.write(f"{i+1}. {layer['type']}")
    with cols[1]:
        if i > 0 and cols[1].button("🔼", key=f"up_{i}"):
            st.session_state.layers[i - 1], st.session_state.layers[i] = st.session_state.layers[i], st.session_state.layers[i - 1]
            st.rerun()
    with cols[2]:
        if i < len(st.session_state.layers) - 1 and cols[2].button("🔽", key=f"down_{i}"):
            st.session_state.layers[i], st.session_state.layers[i + 1] = st.session_state.layers[i + 1], st.session_state.layers[i]
            st.rerun()
    with cols[3]:
        if cols[3].button("❌", key=f"remove_{i}"):
            st.session_state.layers.pop(i)
            st.rerun()

# Training block
if st.button("🚀 학습 시작"):
    st.info("학습을 시작합니다...")

    model = build_model(st.session_state.layers)

    loss_history = deque()
    acc_history = deque()
    logs = []
    graph_placeholder = st.empty()

    for epoch in range(epochs):
        losses, accs, log = train(model, epochs=1, lr=lr, return_log=True)
        loss_history.extend(losses)
        acc_history.extend(accs)
        logs.extend(log)

        with graph_placeholder:
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            ax[0].plot(loss_history, label="Loss", color="tomato")
            ax[0].set_xlabel("Epoch")
            ax[0].legend()
            ax[1].plot(acc_history, label="Accuracy", color="royalblue")
            ax[1].set_xlabel("Epoch")
            ax[1].legend()
            st.pyplot(fig)
        time.sleep(0.3)

    st.markdown("""
    ### 📘 그래프 해석 가이드

    - **Loss**는 모델이 얼마나 오답을 내고 있는지를 나타냅니다. 낮을수록 좋습니다.
    - **Accuracy**는 정답률을 나타냅니다. 1에 가까울수록 좋습니다.
    - 학습이 잘 되고 있다면, Loss는 점점 감소하고, Accuracy는 증가해야 합니다.
    - 그래프가 **너무 변동이 심하거나, Loss가 줄지 않는다면** 학습률이나 Layer 구성을 다시 확인해보세요.
    """)

    st.subheader("📜 학습 로그")
    st.text_area("Log", value="\n".join(logs), height=200)

    st.subheader("🧠 계산 그래프")
    dummy_x = Variable(np.random.randn(1, input_size))
    graph_path = plot_model_graph(model, dummy_x)
    if graph_path and os.path.exists(graph_path):
        st.image(graph_path, caption="계산 그래프")
    else:
        st.warning("계산 그래프 이미지를 불러올 수 없습니다. Graphviz가 설치되어 있는지 확인하세요.")

    st.markdown('<div id="scroll-anchor"></div>', unsafe_allow_html=True)
    st.markdown("""
    <script>
    document.getElementById('scroll-anchor').scrollIntoView({behavior: 'smooth'});
    </script>
    """, unsafe_allow_html=True)

    if st.button("🔄 초기화"):
        st.session_state.layers = []
        st.rerun()