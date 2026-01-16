import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.integrate import solve_ivp

# --- 1. 页面整体配置 ---
st.set_page_config(page_title="三体问题实验室", layout="wide")

# --- 2. 知识阐述模块 ---
st.title("🌌 三体问题 (Three-Body Problem) 模拟实验室")

with st.expander("📖 什么是三体问题？（点击展开）", expanded=True):
    st.markdown("""
    ### 1. 物理定义
    三体问题是指三个质量、初始位置和初始速度都是任意的可视为质点的天体，在相互之间万有引力的作用下的运动规律问题。

    ### 2. 数学表达
    每个天体受到另外两个天体的引力作用，遵循牛顿第二定律：
    """)
    # 使用 LaTeX 渲染核心公式
    st.latex(
        r"m_i \frac{d^2\mathbf{r}_i}{dt^2} = \sum_{j \neq i} \frac{Gm_im_j(\mathbf{r}_j - \mathbf{r}_i)}{|\mathbf{r}_j - \mathbf{r}_i|^3}")

    st.markdown("""
    ### 3. 为什么它如此困难？
    * **非线性与混沌：** 三体问题是高度非线性的。1887年庞加莱证明，除了极少数特殊解（如拉格朗日点），三体问题**没有通用的解析解**。
    * **蝴蝶效应：** 初始条件的极微小变化，都会导致后续轨道发生翻天覆地的变化。这就是为什么《三体》小说中，三体文明无法预测恒星纪元的原因。
    """)
with st.expander("📖 深度解析：三体问题的数学精髓", expanded=True):
    st.markdown("""
    ### 1. 向量化运动方程
    对于 $n=3$ 的系统，每个天体 $i$ 的运动受其他两个天体的万有引力叠加。其加速度向量 $\mathbf{a}_i$ 表达为：
    """)

    # 更精细的求和公式
    st.latex(
        r"\mathbf{a}_i = \frac{d^2\mathbf{r}_i}{dt^2} = \sum_{j \neq i}^{3} \frac{G m_j (\mathbf{r}_j - \mathbf{r}_i)}{|\mathbf{r}_j - \mathbf{r}_i|^3}")

    st.markdown("""
    其中：
    - $\mathbf{r}_i = (x_i, y_i, z_i)$ 是天体 $i$ 的位置向量。
    - $|\mathbf{r}_j - \mathbf{r}_i|$ 是天体 $j$ 与 $i$ 之间的欧几里得距离。
    - $G$ 是引力常数。

    ### 2. 数值求解：状态空间转化
    由于计算机无法直接求解连续的二阶导数，我们需要将上述方程转化为一阶微分方程组（状态空间表示）。
    若令 $\mathbf{v}_i = \frac{d\mathbf{r}_i}{dt}$，则有：
    """)

    st.latex(r"""
    \begin{cases}
    \frac{d\mathbf{r}_i}{dt} = \mathbf{v}_i \\
    \frac{d\mathbf{v}_i}{dt} = \mathbf{a}_i
    \end{cases}
    """)
    st.info("💡 这是一个包含 18 个一阶微分方程的方程组（3个天体 × 3维空间 × (位置+速度)）。")

# --- 3. 参数控制侧边栏 ---
st.sidebar.header("🛠️ 模拟参数设置")
G = st.sidebar.slider("引力常数 (G)", 0.1, 2.0, 1.0)
t_max = st.sidebar.slider("模拟总时长", 5, 100, 20)
dt = st.sidebar.slider("时间精度 (步长)", 0.01, 0.1, 0.05)

st.sidebar.subheader("🪐 天体初始质量")
m1 = st.sidebar.number_input("天体 A 质量", value=1.0)
m2 = st.sidebar.number_input("天体 B 质量", value=1.0)
m3 = st.sidebar.number_input("天体 C 质量", value=1.0)


# --- 4. 物理计算逻辑 ---
def get_physics_engine(t, y, m1, m2, m3, G):
    r1, r2, r3 = y[0:3], y[3:6], y[6:9]
    v1, v2, v3 = y[9:12], y[12:15], y[15:18]

    def accel(pos_a, pos_b, m_b):
        dist = np.linalg.norm(pos_b - pos_a)
        return G * m_b * (pos_b - pos_a) / (dist ** 3 + 1e-5)  # 加上 epsilon 防止除零

    a1 = accel(r1, r2, m2) + accel(r1, r3, m3)
    a2 = accel(r2, r1, m1) + accel(r2, r3, m3)
    a3 = accel(r3, r1, m1) + accel(r3, r2, m2)

    return np.concatenate([v1, v2, v3, a1, a2, a3])


# 初始位置与速度 (随机或预设一组混沌轨道)
y0 = np.array([
    -1, 0, 0, 1, 0, 0, 0, 1, 0,  # 位置 (x1,y1,z1, x2,y2,z2...)
    0.2, 0.2, 0, -0.2, 0, 0, 0, -0.2, 0  # 速度
])

# --- 5. 运行模拟与绘图 ---
if st.button("🚀 开始模拟计算并运行轨道动画"):
    t_eval = np.arange(0, t_max, dt)
    sol = solve_ivp(get_physics_engine, (0, t_max), y0, args=(m1, m2, m3, G), t_eval=t_eval)

    # 提取轨迹
    r1_track = sol.y[0:3]
    r2_track = sol.y[3:6]
    r3_track = sol.y[6:9]

    # 构建动态图表
    fig = go.Figure(
        data=[
            go.Scatter3d(x=r1_track[0][:1], y=r1_track[1][:1], z=r1_track[2][:1], name="天体 A",
                         line=dict(color='red', width=4)),
            go.Scatter3d(x=r2_track[0][:1], y=r2_track[1][:1], z=r2_track[2][:1], name="天体 B",
                         line=dict(color='blue', width=4)),
            go.Scatter3d(x=r3_track[0][:1], y=r3_track[1][:1], z=r3_track[2][:1], name="天体 C",
                         line=dict(color='green', width=4))
        ],
        layout=go.Layout(
            scene=dict(aspectmode='cube', xaxis=dict(range=[-3, 3]), yaxis=dict(range=[-3, 3]),
                       zaxis=dict(range=[-3, 3])),
            updatemenus=[dict(type="buttons", buttons=[dict(label="播放轨道", method="animate", args=[None])])]
        ),
        frames=[go.Frame(data=[
            go.Scatter3d(x=r1_track[0][:k], y=r1_track[1][:k], z=r1_track[2][:k]),
            go.Scatter3d(x=r2_track[0][:k], y=r2_track[1][:k], z=r2_track[2][:k]),
            go.Scatter3d(x=r3_track[0][:k], y=r3_track[1][:k], z=r3_track[2][:k])
        ]) for k in range(2, len(t_eval), 5)]
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption("提示：可以用鼠标拖动旋转画面，或者点击 Play 按钮查看动态演化过程。")