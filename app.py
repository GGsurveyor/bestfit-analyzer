import io
import numpy as np
import pandas as pd
import streamlit as st


# 定义 BestFit 核心计算引擎
class BestFitEngine:

  @staticmethod
  def best_fit(measured, design):
    centroid_m = np.mean(measured, axis=0)
    centroid_d = np.mean(design, axis=0)
    m_centered = measured - centroid_m
    d_centered = design - centroid_d
    H = np.dot(m_centered.T, d_centered)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
      Vt[2, :] *= -1
      R = np.dot(Vt.T, U.T)
    T = centroid_d - np.dot(R, centroid_m)
    return R, T

  @staticmethod
  def apply_transform(measured, R, T):
    return np.dot(measured, R.T) + T


# Streamlit 网页界面布局
st.set_page_config(page_title="BestFit 测量对齐工具", page_icon="📐", layout="wide")

st.title("📐 3D BestFit 自动对齐与分析系统")
st.markdown(
    "上传 **Design（设计文件）** 与 **Before Bestfit（测量前文件）**"
    "，系统将自动计算出最优旋转矩阵、平移向量及对齐后的坐标结果。"
)

# 侧边栏文件上传
st.sidebar.header("📂 上传测量数据")
uploaded_design = st.sidebar.file_uploader(
    "上传 Design CSV 文件", type=["csv"]
)
uploaded_before = st.sidebar.file_uploader(
    "上传 Before Bestfit CSV 文件", type=["csv"]
)

if uploaded_design is not None and uploaded_before is not None:
  try:
    # 读取数据
    df_design = pd.read_csv(
        uploaded_design, header=None, names=["Point", "X", "Y", "Z"]
    )
    df_before = pd.read_csv(
        uploaded_before, header=None, names=["Point", "X", "Y", "Z"]
    )

    df_design.set_index("Point", inplace=True)
    df_before.set_index("Point", inplace=True)

    # 提取共同点进行计算
    common_points = df_design.index.intersection(df_before.index)

    if len(common_points) < 3:
      st.error("错误：共同点数量少于 3 个，无法进行 3D BestFit 计算！")
    else:
      design_pts = df_design.loc[common_points, ["X", "Y", "Z"]].values
      before_pts = df_before.loc[common_points, ["X", "Y", "Z"]].values

      # 运行算法
      R, T = BestFitEngine.best_fit(before_pts, design_pts)

      # 应用到全量数据
      all_before_pts = df_before[["X", "Y", "Z"]].values
      transformed_pts = BestFitEngine.apply_transform(all_before_pts, R, T)

      df_after = pd.DataFrame(
          transformed_pts,
          index=df_before.index,
          columns=["X", "N_Y" if False else "Y", "Z"],
      )  # 修正列名
      df_after.columns = ["X", "Y", "Z"]

      # 展示计算出的矩阵
      st.markdown("---")
      st.subheader("📊 空间变换矩阵结果")
      col1, col2 = st.columns(2)
      with col1:
        st.text("旋转矩阵 R (Rotation):")
        st.write(R)
      with col2:
        st.text("平移向量 T (Translation):")
        st.write(T)

      # 展示对齐后的坐标表格
      st.markdown("---")
      st.subheader("📋 计算后的 After Bestfit 坐标预览")
      st.dataframe(df_after)

      # 提供下载按钮
      csv_data = df_after.reset_index().to_csv(index=False, header=False)
      st.download_button(
          label="📥 下载 After Bestfit 结果文件 (.CSV)",
          data=csv_data,
          file_name="LP22B_AW_calculated_after.CSV",
          mime="text/csv",
      )

  except Exception as e:
    st.error(f"处理过程中出现错误: {e}")
else:
  st.info("👈 请在左侧侧边栏同时上传 **Design** 和 **Before** 两个 CSV 文件。")
