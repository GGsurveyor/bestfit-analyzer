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
        R = np.eye(3); R[:2, :2] = R_2d
        T = np.zeros(3); T[:2] = T_2d; T[2] = np.mean(design[:, 2]) - np.mean(measured[:, 2])
        return R, T

st.set_page_config(page_title="Advanced Pipeline BestFit", layout="wide")
st.title("🏗️ Advanced Pipeline: Multi-Station BestFit Workstation")

# Uploads
st.sidebar.header("📁 Data Inputs")
uploaded_design = st.sidebar.file_uploader("Design Points", type=["csv"], key="design_file")
uploaded_ctrl = st.sidebar.file_uploader("Control Points (for 2D)", type=["csv"], key="ctrl_file")
uploaded_raw = st.sidebar.file_uploader("Raw Data", type=["csv"], key="raw_file")

if all([uploaded_design, uploaded_ctrl, uploaded_raw]):
    df_design = pd.read_csv(uploaded_design, header=None, names=["Point","X","Y","Z"]).set_index("Point")
    df_ctrl = pd.read_csv(uploaded_ctrl, header=None, names=["Point","X","Y","Z"]).set_index("Point")
    df_raw = pd.read_csv(uploaded_raw, header=None, names=["Point","X","Y","Z"])
    
    # 1. Station Config
    num_stations = st.sidebar.number_input("Number of Stations", 1, 10, 4)
    station_configs = {f"Station-{i+1}": (i*(len(df_raw)//num_stations), (i+1)*(len(df_raw)//num_stations)) for i in range(num_stations)}
    
    # 2. Step 2: Individual Fit
    st.subheader("🎯 Step 2: Individual Station Fit")
    if "fitted_stations" not in st.session_state: st.session_state.fitted_stations = {}
    
    for s_name, (start, end) in station_configs.items():
        with st.expander(f"Configuring {s_name}"):
            stn_raw = df_raw.iloc[start:end].set_index("Point")
            common = df_ctrl.index.intersection(stn_raw.index)
            
            # 选项：选择算法 + 排除点
            col1, col2 = st.columns(2)
            method = col1.selectbox(f"Method for {s_name}", ["2D BestFit", "3D BestFit"], key=f"m_{s_name}")
            exclude = col2.multiselect(f"Exclude Points in {s_name}", common, key=f"ex_{s_name}")
            
            if st.button(f"Fit {s_name}", key=f"btn_{s_name}"):
                use_points = [p for p in common if p not in exclude]
                m_pts = stn_raw.loc[use_points].values
                d_pts = df_ctrl.loc[use_points].values
                
                R, T = BestFitEngine.best_fit_3d(m_pts, d_pts) if "3D" in method else BestFitEngine.best_fit_2d(m_pts, d_pts)
                st.session_state.fitted_stations[s_name] = pd.DataFrame(np.dot(stn_raw.values, R.T) + T, index=stn_raw.index, columns=["X","Y","Z"])
                st.success(f"{s_name} fitted using {method}")

    # 3. Step 3: Final Combined Fit
    st.subheader("🚀 Step 3: Final Combined Fit")
    if len(st.session_state.fitted_stations) == num_stations:
        combined = pd.concat(st.session_state.fitted_stations.values())
        common_d = df_design.index.intersection(combined.index)
        
        c1, c2 = st.columns(2)
        f_method = c1.selectbox("Final Method", ["3D BestFit", "2D BestFit"])
        f_exclude = c2.multiselect("Exclude Points for Final Fit", common_d)
        
        if st.button("Final Execution"):
            use_pts = [p for p in common_d if p not in f_exclude]
            R, T = BestFitEngine.best_fit_3d(combined.loc[use_pts].values, df_design.loc[use_pts].values) if "3D" in f_method else BestFitEngine.best_fit_2d(combined.loc[use_pts].values, df_design.loc[use_pts].values)
            final_df = pd.DataFrame(np.dot(combined.values, R.T) + T, index=combined.index, columns=["X","Y","Z"])
            st.dataframe(final_df.style.format("{:.4f}"))
            st.download_button("Download Final Result", final_df.to_csv(), "Final_Combined_Fit.csv")
    else:
        st.warning("Please finish all stations in Step 2 first.")
