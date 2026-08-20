import numpy as np
import pandas as pd
import streamlit as st

class BestFitEngine:
    @staticmethod
    def best_fit_3d(measured, design):
        centroid_m, centroid_d = np.mean(measured, axis=0), np.mean(design, axis=0)
        H = np.dot((measured - centroid_m).T, (design - centroid_d))
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = np.dot(Vt.T, U.T)
        T = centroid_d - np.dot(R, centroid_m)
        return R, T

    @staticmethod
    def best_fit_2d(measured, design):
        m_xy, d_xy = measured[:, :2], design[:, :2]
        centroid_m, centroid_d = np.mean(m_xy, axis=0), np.mean(d_xy, axis=0)
        H = np.dot((m_xy - centroid_m).T, (d_xy - centroid_d))
        U, S, Vt = np.linalg.svd(H)
        R_2d = np.dot(Vt.T, U.T)
        if np.linalg.det(R_2d) < 0:
            Vt[1, :] *= -1
            R_2d = np.dot(Vt.T, U.T)
        T_2d = centroid_d - np.dot(R_2d, centroid_m)
        R = np.eye(3)
        R[:2, :2] = R_2d
        T = np.zeros(3); T[:2] = T_2d; T[2] = np.mean(design[:, 2]) - np.mean(measured[:, 2])
        return R, T

st.set_page_config(page_title="Advanced Multi-Stage BestFit", layout="wide")
st.title("🏗️ Professional Pipeline: Multi-Station 2D -> Combined 3D BestFit")

# 1. Sidebar Uploads
st.sidebar.header("📁 Data Inputs")
df_design = pd.read_csv(st.sidebar.file_uploader("Design Points", type=["csv"], key="d"), header=None, names=["Point","X","Y","Z"]) if st.sidebar.file_uploader("Design Points", type=["csv"], key="d") else None
df_ctrl = pd.read_csv(st.sidebar.file_uploader("Control Points (for 2D)", type=["csv"], key="c"), header=None, names=["Point","X","Y","Z"]) if st.sidebar.file_uploader("Control Points (for 2D)", type=["csv"], key="c") else None
df_raw = pd.read_csv(st.sidebar.file_uploader("Raw Data", type=["csv"], key="r"), header=None, names=["Point","X","Y","Z"]) if st.sidebar.file_uploader("Raw Data", type=["csv"], key="r") else None

if all([df_design is not None, df_ctrl is not None, df_raw is not None]):
    df_design.set_index("Point", inplace=True)
    df_ctrl.set_index("Point", inplace=True)
    
    # Session States
    if "fitted_stations" not in st.session_state: st.session_state.fitted_stations = {}

    st.subheader("🛠️ Step 1: Split & 2D Fit Stations")
    # 假设分站逻辑已存在，此处简化展示：循环处理每个站
    stations = ["Station-1", "Station-2", "Station-3", "Station-4"]
    
    for s_name in stations:
        with st.expander(f"Setup {s_name}"):
            # 此处模拟提取对应 Station 的数据，实际开发中请绑定您的切分逻辑
            raw_data = df_raw.iloc[0:100] # 示例切片
            if st.button(f"Apply 2D BestFit to {s_name}", key=f"btn_{s_name}"):
                common = df_ctrl.index.intersection(raw_data["Point"])
                R, T = BestFitEngine.best_fit_2d(raw_data.set_index("Point").loc[common].values, df_ctrl.loc[common].values)
                fitted_pts = np.dot(raw_data.set_index("Point").values, R.T) + T
                st.session_state.fitted_stations[s_name] = pd.DataFrame(fitted_pts, index=raw_data["Point"], columns=["X","Y","Z"])
                st.success(f"{s_name} 2D Fitted successfully.")

    # 2. Final 3D BestFit
    st.subheader("🎯 Step 2: Final Combined 3D BestFit")
    if len(st.session_state.fitted_stations) == 4:
        # 合并
        combined_df = pd.concat(st.session_state.fitted_stations.values())
        
        if st.button("🚀 Perform Final 3D BestFit on Combined Data"):
            common = df_design.index.intersection(combined_df.index)
            R_3d, T_3d = BestFitEngine.best_fit_3d(combined_df.loc[common].values, df_design.loc[common].values)
            final_coords = np.dot(combined_df.values, R_3d.T) + T_3d
            final_df = pd.DataFrame(final_coords, index=combined_df.index, columns=["X","Y","Z"])
            
            st.dataframe(final_df.style.format("{:.4f}"))
            st.download_button("📥 Download Final Combined Result", final_df.to_csv(), "Final_BestFit_Result.csv")
    else:
        st.warning("Please complete 2D BestFit for all 4 stations first.")
else:
    st.info("Please upload all required files to start.")
