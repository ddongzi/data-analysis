import numpy as np
import streamlit as st
import plotly.graph_objects as go

# =========================
# 基本参数
# =========================
G = 1.0

# =========================
# 三体微分方程 (保持不变)
# =========================
def acceleration(r, m):
    a = np.zeros_like(r)
    for i in range(3):
        for j in range(3):
            if i != j:
                diff = r[j] - r[i]
                dist = np.linalg.norm(diff)
                a[i] += G * m[j] * diff / (dist**3 + 1e-9)
    return a

def integrate(r0, v0, m, dt, steps):
    r = r0.copy()
    v = v0.copy()
    traj = np.zeros((steps, 3, 3))
    for i in range(steps):
        traj[i] = r
        a = acceleration(r, m)
        v += a * dt
        r += v * dt
    return traj

# =========================
# Streamlit UI
# =========================
st.set_page_config(layout="wide", page_title="3-Body Simulator")
st.title("三体问题数值模拟")

# 侧边栏配置
st.sidebar.header("1. 模拟环境参数")
dt = st.sidebar.slider("时间步长 (dt)", 0.0005, 0.02, 0.002, step=0.0005)
steps = st.sidebar.slider("总步数", 1000, 50000, 8000, step=1000)

st.sidebar.header("2. 质量与初始状态")

# 定义默认值（Figure-8 经典解）
def_r = [[-0.970, 0.243, 0.0], [0.970, -0.243, 0.0], [0.0, 0.0, 0.0]]
def_v = [[0.466, 0.432, 0.0], [0.466, 0.432, 0.0], [-0.932, -0.864, 0.0]]

m = np.ones(3)
r0 = np.zeros((3, 3))
v0 = np.zeros((3, 3))

# 使用 Expander 整理输入界面，避免侧边栏过长
for i in range(3):
    with st.sidebar.expander(f"天体 {i+1} 参数", expanded=(i==0)):
        m[i] = st.number_input(f"质量 m{i+1}", value=1.0, step=0.1)
        
        st.markdown("**初始位置 (r)**")
        col_r = st.columns(3)
        r0[i, 0] = col_r[0].number_input(f"x{i+1}", value=def_r[i][0], format="%.3f")
        r0[i, 1] = col_r[1].number_input(f"y{i+1}", value=def_r[i][1], format="%.3f")
        r0[i, 2] = col_r[2].number_input(f"z{i+1}", value=def_r[i][2], format="%.3f")
        
        st.markdown("**初始速度 (v)**")
        col_v = st.columns(3)
        v0[i, 0] = col_v[0].number_input(f"vx{i+1}", value=def_v[i][0], format="%.3f")
        v0[i, 1] = col_v[1].number_input(f"vy{i+1}", value=def_v[i][1], format="%.3f")
        v0[i, 2] = col_v[2].number_input(f"vz{i+1}", value=def_v[i][2], format="%.3f")

run = st.sidebar.button("🚀 开始运行模拟", use_container_width=True)

# =========================
# 计算 & 绘制
# =========================
if run:
    with st.spinner("计算中..."):
        traj = integrate(r0, v0, m, dt, steps)

    fig = go.Figure()
    colors = ["#FF4B4B", "#1C83E1", "#00C04B"]

    for i in range(3):
        # 轨迹线
        fig.add_trace(go.Scatter3d(
            x=traj[:, i, 0], y=traj[:, i, 1], z=traj[:, i, 2],
            mode="lines",
            line=dict(width=4, color=colors[i]),
            name=f"天体 {i+1} (m={m[i]})"
        ))
        # 终点标记
        fig.add_trace(go.Scatter3d(
            x=[traj[-1, i, 0]], y=[traj[-1, i, 1]], z=[traj[-1, i, 2]],
            mode="markers",
            marker=dict(size=6, color=colors[i]),
            showlegend=False
        ))

    fig.update_layout(
        height=800,
        scene=dict(
            aspectmode="data",
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            bgcolor="black" # 设置背景为黑色更有科幻感
        ),
        template="plotly_dark",
        margin=dict(l=0, r=0, b=0, t=40),
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("在左侧调整参数并点击 '开始运行模拟'。默认值为经典的 Figure-8 轨道。")
