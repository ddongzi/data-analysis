import numpy as np
import streamlit as st
import plotly.graph_objects as go

# =========================
# 基本参数
# =========================
G = 1.0
m = np.array([1.0, 1.0, 1.0])

# =========================
# 三体微分方程
# =========================
def acceleration(r):
    a = np.zeros_like(r)
    for i in range(3):
        for j in range(3):
            if i != j:
                diff = r[j] - r[i]
                dist = np.linalg.norm(diff)
                a[i] += G * m[j] * diff / (dist**3 + 1e-9)
    return a

def integrate(r0, v0, dt, steps):
    r = r0.copy()
    v = v0.copy()

    traj = np.zeros((steps, 3, 3))
    for i in range(steps):
        traj[i] = r
        a = acceleration(r)
        v += a * dt
        r += v * dt
    return traj

# =========================
# Streamlit UI
# =========================
st.set_page_config(layout="wide")
st.title("Three-Body Problem · Streamlit + Plotly")

st.sidebar.header("Simulation Parameters")

dt = st.sidebar.slider("dt", 0.0005, 0.01, 0.002, step=0.0005)
steps = st.sidebar.slider("Steps", 2000, 20000, 8000, step=1000)
z_eps = st.sidebar.slider("z 扰动", 0.0, 0.1, 0.02)

run = st.sidebar.button("Run Simulation")

# =========================
# 初始条件（Figure-8 + 3D扰动）
# =========================
r0 = np.array([
    [-0.97000436,  0.24308753, 0.0],
    [ 0.97000436, -0.24308753, 0.0],
    [ 0.0,         0.0,        z_eps],
])

v0 = np.array([
    [ 0.4662036850,  0.4323657300, 0.0],
    [ 0.4662036850,  0.4323657300, 0.0],
    [-0.9324073700, -0.8647314600, 0.02],
])

# =========================
# 计算 & 绘制
# =========================
if run:
    traj = integrate(r0, v0, dt, steps)

    fig = go.Figure()
    colors = ["red", "green", "blue"]

    for i in range(3):
        fig.add_trace(go.Scatter3d(
            x=traj[:, i, 0],
            y=traj[:, i, 1],
            z=traj[:, i, 2],
            mode="lines",
            line=dict(width=4),
            name=f"Body {i+1}"
        ))

        fig.add_trace(go.Scatter3d(
            x=[traj[-1, i, 0]],
            y=[traj[-1, i, 1]],
            z=[traj[-1, i, 2]],
            mode="markers",
            marker=dict(size=6),
            showlegend=False
        ))

    fig.update_layout(
        height=700,
        scene=dict(
            aspectmode="data",
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("点击左侧 Run Simulation 开始计算")
