import os

def plot_model_graph(model, x, filename="model_graph.png"):
    """
    모델의 계산 그래프를 dot 파일로 출력한 뒤 PNG로 변환합니다.
    Requires: Graphviz 설치 필요
    """
    dot_filename = "model_graph.dot"
    model.plot(x, to_file=dot_filename)

    # dot → png 변환 (Graphviz 설치되어 있어야 함)
    exit_code = os.system(f"dot -Tpng {dot_filename} -o {filename}")
    if exit_code != 0 or not os.path.exists(filename):
        return None
    return filename
