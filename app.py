import numpy as np
import pandas as pd
import streamlit as st


class BestFitEngine:
    @staticmethod
    def best_fit_3d(measured, design):
        centroid_m = np.mean(measured, axis=0)
        centroid_d = np.mean(design, axis=0)
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
        T = np.zeros(3)
        T[:2] = T_2d
        T[2] = np.mean(design[:, 2]) - np.mean(measured[:, 2])
        return R, T

    @staticmethod
    def calculate_error(df_after, df_target):
        # 计算 After - Design/Control
        err = df_after.sub(df_target, fill_value=0)
        err.columns = ["Delta E", "Delta N", "Delta El"]
        err["Total_Error"] = np.sqrt(err["Delta E"]**2 + err["Delta N"]**2 + err["Delta El"]**2)
        return err

st.set_page_config(page_title="BestFit Analyzer Pro", layout="wide")
st.title("🏗️ Complete Pipeline: With Deviation Analysis")

# Sidebar Uploads
st.sidebar.header("📂 Data Inputs")
uploaded_design = st.sidebar.file_uploader("Design Points CSV", type=["csv"], key="design")
uploaded_ctrl = st.sidebar.file_uploader("Control Points CSV (for 2D)", type=["csv"], key="ctrl")
uploaded_raw = st.sidebar.file_uploader("Raw Data CSV", type=["csv"], key="raw")

if all([uploaded_design, uploaded_ctrl, uploaded_raw]):
    df_design = pd.read_csv(uploaded_design, header=None, names=["Point","X","Y","Z"]).set_index("Point")
    df_ctrl = pd.read_csv(uploaded_ctrl, header=None, names=["Point","X","Y","Z"]).set_index("Point")
    df_raw = pd.read_csv(uploaded_raw, header=None, names=["Point","X","Y","Z"])
    
    # --- Step 0 & 1: Raw Data & Splitting ---
    if "df_raw" not in st.session_state: st.session_state.df_raw = df_raw
    st.session_state.df_raw = st.data_editor(st.session_state.df_raw, num_rows="dynamic", key="ed")
    
    # --- Step 2: Station BestFit + Error Analysis ---
    st.subheader("🎯 Step 2: Individual Station Fit & Deviations")
    if "fitted_dfs" not in st.session_state: st.session_state.fitted_dfs = {}
    
    num_stn = st.number_input("Number of Stations", 1, 10, 4)
    for i in range(num_stn):
        s_name = f"Station-{i+1}"
        with st.expander(f"Configuring {s_name}"):
            stn_data = st.session_state.df_raw.iloc[i*(len(st.session_state.df_raw)//num_stn):(i+1)*(len(st.session_state.df_raw)//num_stn)].set_index("Point")
            common = df_ctrl.index.intersection(stn_data.index)
            
            m1, m2 = st.columns(2)
            method = m1.selectbox(f"Method {s_name}", ["2D BestFit", "3D BestFit"], key=f"m_{s_name}")
            exclude = m2.multiselect(f"Exclude {s_name}", common, key=f"ex_{s_name}")
            
            if st.button(f"Fit {s_name}", key=f"btn_{s_name}"):
                pts = [p for p in common if p not in exclude]
                R, T = BestFitEngine.best_fit_3d(stn_data.loc[pts].values, df_ctrl.loc[pts].values) if "3D" in method else BestFitEngine.best_fit_2d(stn_data.loc[pts].values, df_ctrl.loc[pts].values)
                
                fitted = pd.DataFrame(np.dot(stn_data.values, R.T) + T, index=stn_data.index, columns=["X","Y","Z"])
                st.session_state.fitted_dfs[s_name] = fitted
                
                # Show Deviations
                err = BestFitEngine.calculate_error(fitted.loc[pts], df_ctrl.loc[pts])
                st.write("Deviation Analysis (Control Points):")
                st.dataframe(err.style.format("{:.4f}"))

    # --- Step 3: Final Fit + Error Analysis ---
    st.subheader("🚀 Step 3: Combined Final Fit & Deviations")
    if len(st.session_state.fitted_dfs) == num_stn:
        combined = pd.concat(st.session_state.fitted_dfs.values())
        common_d = df_design.index.intersection(combined.index)
        
        m_final = st.selectbox("Final Method", ["3D BestFit", "2D BestFit"])
        ex_final = st.multiselect("Exclude Final", common_d)
        
        if st.button("Execute Final Fit"):
            pts = [p for p in common_d if p not in ex_final]
            R, T = BestFitEngine.best_fit_3d(combined.loc[pts].values, df_design.loc[pts].values) if "3D" in m_final else BestFitEngine.best_fit_2d(combined.loc[pts].values, df_design.loc[pts].values)
            final_df = pd.DataFrame(np.dot(combined.values, R.T) + T, index=combined.index, columns=["X","Y","Z"])
            
            # Show Deviations
            err = BestFitEngine.calculate_error(final_df.loc[pts], df_design.loc[pts])
            st.write("Final Deviation Analysis (Design Points):")
            st.dataframe(err.style.format("{:.4f}"))
            
            st.download_button("Download Final Result", final_df.to_csv(), "Final_Result.csv")
