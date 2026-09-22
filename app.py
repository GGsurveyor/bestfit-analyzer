if "df_final_result" in st.session_state:
                        st.markdown("### 📊 Active Source Preview, Deviation & Flatness Analysis")
                        
                        # --- 🌟 新增：Dimensional Control Flatness 自动分析报告 ---
                        z_vals = active_cad_df["Z"].astype(float).values
                        max_dev = np.max(z_vals)import os
import tempfile
from datetime import datetime
from fpdf import FPDF
import ezdxf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


class BestFitEngine:

    @staticmethod
    def best_fit_3d(measured, design):
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
    def best_fit_2d(measured, design):
        m_xy = measured[:, :2]
        d_xy = design[:, :2]
        centroid_m = np.mean(m_xy, axis=0)
        centroid_d = np.mean(d_xy, axis=0)
        m_centered = m_xy - centroid_m
        d_centered = d_xy - centroid_d
        H = np.dot(m_centered.T, d_centered)
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
        err = df_after.sub(df_target, fill_value=0)
        err.columns = ["Delta E", "Delta N", "Delta El"]
        err["Total_Error"] = np.sqrt(
            err["Delta E"] ** 2 + err["Delta N"] ** 2 + err["Delta El"] ** 2
        )
        if not err.index.is_unique:
            err = err.loc[~err.index.duplicated(keep="first")]
        return err


class TransformEngine:

    @staticmethod
    def translate(df, dx, dy, dz):
        res = df.copy()
        if "X/E" in res.columns:
            res["X/E"] += dx
            res["Y/N"] += dy
            res["Z/EL"] += dz
        elif "X" in res.columns:
            res["X"] += dx
            res["Y"] += dy
            res["Z"] += dz
        return res

    @staticmethod
    def rotate_plane(df, plane_type, angle_deg):
        res = df.copy()
        rad = np.radians(angle_deg)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        
        x_col = "X/E" if "X/E" in res.columns else "X"
        y_col = "Y/N" if "Y/N" in res.columns else "Y"
        z_col = "Z/EL" if "Z/EL" in res.columns else "Z"

        if plane_type == "E/N plane (X/Y)":
            x = res[x_col].values
            y = res[y_col].values
            res[x_col] = x * cos_a - y * sin_a
            res[y_col] = x * sin_a + y * cos_a
        elif plane_type == "N/EL plane (Y/Z)":
            y = res[y_col].values
            z = res[z_col].values
            res[y_col] = y * cos_a - z * sin_a
            res[z_col] = y * sin_a + z * cos_a
        elif plane_type == "E/EL plane (X/Z)":
            x = res[x_col].values
            z = res[z_col].values
            res[x_col] = x * cos_a + z * sin_a
            res[z_col] = -x * sin_a + z * cos_a
        return res

    @staticmethod
    def fit_to_horizontal_plane(df, selected_points=None, target_z=0.0):
        """
        专业级 3D 最佳拟合平面调平引擎：
        1. 严格按用户指定的点集子范围（如 1.1 到 1.52）提取用于拟合的点云。
        2. 采用 SVD（奇异值分解 / 总体最小二乘法）计算空间最佳拟合平面的法向量。
        3. 以所选点集的几何中心 (Centroid) 作为旋转中心（Pivot），进行严格的 3D 刚体旋转，
           将平面法向量旋转对齐至垂直 Z 轴 [0, 0, 1]，消除无效平移偏差。
        4. 将拟合平面的平均高程平移对齐至目标高程 (Target Z，默认 0.0)。
        """
        res = df.copy()
        x_col = "X/E" if "X/E" in res.columns else ("X" if "X" in res.columns else res.columns[1])
        y_col = "Y/N" if "Y/N" in res.columns else ("Y" if "Y" in res.columns else res.columns[2])
        z_col = "Z/EL" if "Z/EL" in res.columns else ("Z" if "Z" in res.columns else res.columns[3])
        p_col = res.columns[0]

        # 1. 严格筛选参与拟合的目标点子集
        if selected_points is not None and len(selected_points) > 0:
            cleaned_selected = [str(p).strip() for p in selected_points]
            sub = res[res[p_col].astype(str).str.strip().isin(cleaned_selected)]
            if not sub.empty:
                fit_pts = sub[[x_col, y_col, z_col]].values.astype(float)
            else:
                fit_pts = res[[x_col, y_col, z_col]].values.astype(float)
        else:
            fit_pts = res[[x_col, y_col, z_col]].values.astype(float)

        all_xyz = res[[x_col, y_col, z_col]].values.astype(float)

        # 2. 计算所选点集的几何中心 (Centroid) 作为旋转中心
        centroid = np.mean(fit_pts, axis=0)

        # 3. 使用 SVD（奇异值分解）进行正交距离平面拟合（Total Least Squares）
        centered_fit_pts = fit_pts - centroid
        U, S, Vt = np.linalg.svd(centered_fit_pts)
        normal = Vt[2, :]  # 最小奇异值对应的右奇异向量即为平面法向量
        
        # 确保法向量朝上（Z 分量为正）
        if normal[2] < 0:
            normal = -normal

        # 4. 计算将法向量旋转对齐至 [0, 0, 1] 的 3D 旋转矩阵 R
        target = np.array([0, 0, 1])
        v = np.cross(normal, target)
        s_norm = np.linalg.norm(v)
        c_dot = np.dot(normal, target)

        if s_norm < 1e-6:
            R = np.eye(3)
        else:
            vx = np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])
            R = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c_dot) / (s_norm ** 2))

        # 5. 对全部点云应用以几何中心为原点的旋转：P_new = R @ (P - Centroid) + Centroid
        centered_all = all_xyz - centroid
        rotated = np.dot(centered_all, R.T) + centroid

        # 6. 计算旋转后拟合点集的平均高程，并将其整体平移对齐至目标 Z（target_z）
        if selected_points is not None and len(selected_points) > 0 and not sub.empty:
            sub_mask = res[p_col].astype(str).str.strip().isin([str(p).strip() for p in selected_points])
            mean_z_rot = np.mean(rotated[sub_mask, 2])
        else:
            mean_z_rot = np.mean(rotated[:, 2])

        delta_z = target_z - mean_z_rot
        rotated[:, 2] += delta_z

        # 7. 写回结果 DataFrame
        res[x_col] = rotated[:, 0]
        res[y_col] = rotated[:, 1]
        res[z_col] = rotated[:, 2]
        return res


def highlight_excess_error(val):
    try:
        if abs(float(val)) > 0.002:
            return "color: #ff4b4b; font-weight: bold;"
    except (ValueError, TypeError):
        pass
    return ""


class PDFReport(FPDF):

    def header(self):
        self.set_font("helvetica", "B", 14)
        self.cell(0, 10, "2D/3D Multi-Station BestFit & CAD Conversion Report", 0, 1, "C")
        self.set_font("helvetica", "I", 9)
        self.cell(0, 5, "Generated by Ng Yit Fung - BestFit & DXF/SCR Converter Pro", 0, 1, "C")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        current_date_full = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cell(0, 10, f"Page {self.page_no()} | Report Date & Time: {current_date_full}", 0, 0, "C")


def generate_pdf_report(df_final, df_err, include_deviation=True):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font("helvetica", "", 10)

    current_date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 8, "1. Executive Summary", 0, 1, "L")
    pdf.set_font("helvetica", "", 10)
    pdf.cell(0, 6, f"Report Generation Date & Time: {current_date_str}", 0, 1, "L")
    pdf.cell(0, 6, f"Total Processed Points in Final Result: {len(df_final)}", 0, 1, "L")
    pdf.ln(4)

    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 8, "2. Final Coordinates Results", 0, 1, "L")
    pdf.set_font("helvetica", "B", 9)
    pdf.set_fill_color(220, 220, 220)
    pdf.cell(50, 7, "Point ID", 1, 0, "C", True)
    pdf.cell(45, 7, "X / E", 1, 0, "C", True)
    pdf.cell(45, 7, "Y / N", 1, 0, "C", True)
    pdf.cell(45, 7, "Z / EL", 1, 1, "C", True)

    pdf.set_font("helvetica", "", 9)
    for idx, row in df_final.iterrows():
        pdf.cell(50, 6, str(idx), 1, 0, "C")
        pdf.cell(45, 6, f"{float(row['X']):.4f}", 1, 0, "C")
        pdf.cell(45, 6, f"{float(row['Y']):.4f}", 1, 0, "C")
        pdf.cell(45, 6, f"{float(row['Z']):.4f}", 1, 1, "C")

    pdf.ln(6)

    if include_deviation and df_err is not None and not df_err.empty:
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, 8, "3. Deviation Analysis Report", 0, 1, "L")
        pdf.set_font("helvetica", "B", 9)
        pdf.cell(40, 7, "Point ID", 1, 0, "C", True)
        pdf.cell(35, 7, "Delta E", 1, 0, "C", True)
        pdf.cell(35, 7, "Delta N", 1, 0, "C", True)
        pdf.cell(35, 7, "Delta El", 1, 0, "C", True)
        pdf.cell(40, 7, "Total Error", 1, 1, "C", True)

        pdf.set_font("helvetica", "", 9)
        for idx, row in df_err.iterrows():
            pdf.cell(40, 6, str(idx), 1, 0, "C")
            pdf.cell(35, 6, f"{float(row['Delta E']):.4f}", 1, 0, "C")
            pdf.cell(35, 6, f"{float(row['Delta N']):.4f}", 1, 0, "C")
            pdf.cell(35, 6, f"{float(row['Delta El']):.4f}", 1, 0, "C")
            pdf.cell(40, 6, f"{float(row['Total_Error']):.4f}", 1, 1, "C")

    return bytes(pdf.output())


CAD_COLORS = {
    "White (Default)": ("white", 7),
    "Red": ("red", 1),
    "Yellow": ("yellow", 2),
    "Green": ("green", 3),
    "Cyan": ("cyan", 4),
    "Blue": ("blue", 5),
    "Magenta": ("magenta", 6),
    "Gray": ("gray", 8),
}

st.set_page_config(
    page_title="2D/3D Multi-Station BestFit & CAD DXF/SCR Converter Pro",
    page_icon="🏗️",
    layout="wide",
)

st.title("🏗️ 2D/3D Multi-Station BestFit & CAD DXF/SCR Converter - Made by Ng Yit Fung")
st.markdown("Complete Raw Data editing, Station splitting, Pre/Post Transforms, BestFit analysis, and export to CSV / DXF / SCR / PDF formats.")

st.sidebar.header("📂 Data Uploads")
uploaded_design = st.sidebar.file_uploader("Upload Design Points CSV (Optional)", type=["csv"], key="design_file")
uploaded_ctrl = st.sidebar.file_uploader("Upload Control Points CSV (Optional)", type=["csv"], key="ctrl_file")
uploaded_raw = st.sidebar.file_uploader("Upload Raw Data CSV (Required)", type=["csv"], key="raw_file")

if uploaded_raw is not None:
    try:
        if "df_design_edited" not in st.session_state:
            if uploaded_design is not None:
                df_d_init = pd.read_csv(uploaded_design, header=None, names=["Point", "X/E", "Y/N", "Z/EL"], dtype=str)
            else:
                df_d_init = pd.DataFrame(columns=["Point", "X/E", "Y/N", "Z/EL"])
            df_d_init.index = range(1, len(df_d_init) + 1)
            st.session_state["df_design_edited"] = df_d_init

        if "df_ctrl_edited" not in st.session_state:
            if uploaded_ctrl is not None:
                df_c_init = pd.read_csv(uploaded_ctrl, header=None, names=["Point", "X/E", "Y/N", "Z/EL"], dtype=str)
            else:
                df_c_init = pd.DataFrame(columns=["Point", "X/E", "Y/N", "Z/EL"])
            df_c_init.index = range(1, len(df_c_init) + 1)
            st.session_state["df_ctrl_edited"] = df_c_init

        if "df_raw_edited" not in st.session_state:
            df_r_init = pd.read_csv(uploaded_raw, header=None, names=["Point", "X/E", "Y/N", "Z/EL"], dtype=str)
            df_r_init.index = range(1, len(df_r_init) + 1)
            st.session_state["df_raw_edited"] = df_r_init

        unified_column_config = {
            "Point": st.column_config.TextColumn("Point", width="small"),
            "X/E": st.column_config.NumberColumn("X/E", format="%.4f", width="small"),
            "Y/N": st.column_config.NumberColumn("Y/N", format="%.4f", width="small"),
            "Z/EL": st.column_config.NumberColumn("Z/EL", format="%.4f", width="small"),
        }

        st.markdown("---")
        col_row1_1, col_row1_2 = st.columns(2)

        with col_row1_1:
            st.subheader("📐 Design Points Editor")
            current_design = st.session_state["df_design_edited"].copy()
            current_design["Point"] = current_design["Point"].astype(str)
            for col in ["X/E", "Y/N", "Z/EL"]:
                current_design[col] = pd.to_numeric(current_design[col], errors="coerce")
            current_design.index = range(1, len(current_design) + 1)
            edited_design_df = st.data_editor(
                current_design, num_rows="dynamic", use_container_width=True, hide_index=False, column_config=unified_column_config, key="design_data_editor"
            )
            st.session_state["df_design_edited"] = edited_design_df

        with col_row1_2:
            st.subheader("🎯 Control Points Editor")
            current_ctrl = st.session_state["df_ctrl_edited"].copy()
            current_ctrl["Point"] = current_ctrl["Point"].astype(str)
            for col in ["X/E", "Y/N", "Z/EL"]:
                current_ctrl[col] = pd.to_numeric(current_ctrl[col], errors="coerce")
            current_ctrl.index = range(1, len(current_ctrl) + 1)
            edited_ctrl_df = st.data_editor(
                current_ctrl, num_rows="dynamic", use_container_width=True, hide_index=False, column_config=unified_column_config, key="ctrl_data_editor"
            )
            st.session_state["df_ctrl_edited"] = edited_ctrl_df

        df_design = None
        if not st.session_state["df_design_edited"].empty:
            temp_d = st.session_state["df_design_edited"].dropna(subset=["Point"]).copy()
            if not temp_d.empty and "X/E" in temp_d.columns:
                temp_d["Point"] = temp_d["Point"].astype(str).str.strip()
                temp_d = temp_d.loc[~temp_d["Point"].duplicated(keep="first")]
                temp_d.set_index("Point", inplace=True)
                df_design = temp_d[["X/E", "Y/N", "Z/EL"]].astype(float).copy()
                df_design.columns = ["X", "Y", "Z"]

        df_ctrl = None
        if not st.session_state["df_ctrl_edited"].empty:
            temp_c = st.session_state["df_ctrl_edited"].dropna(subset=["Point"]).copy()
            if not temp_c.empty and "X/E" in temp_c.columns:
                temp_c["Point"] = temp_c["Point"].astype(str).str.strip()
                temp_c = temp_c.loc[~temp_c["Point"].duplicated(keep="first")]
                temp_c.set_index("Point", inplace=True)
                df_ctrl = temp_c[["X/E", "Y/N", "Z/EL"]].astype(float).copy()
                df_ctrl.columns = ["X", "Y", "Z"]

        st.markdown("---")
        col_row2_1, col_row2_2 = st.columns(2)

        with col_row2_1:
            st.subheader("✏️ Raw Data Editor")
            current_raw = st.session_state["df_raw_edited"].copy()
            current_raw["Point"] = current_raw["Point"].astype(str)
            for col in ["X/E", "Y/N", "Z/EL"]:
                current_raw[col] = pd.to_numeric(current_raw[col], errors="coerce")
            current_raw.index = range(1, len(current_raw) + 1)
            edited_raw_df = st.data_editor(
                current_raw, num_rows="dynamic", use_container_width=True, hide_index=False, column_config=unified_column_config, key="raw_data_editor"
            )
            st.session_state["df_raw_edited"] = edited_raw_df

        with col_row2_2:
            st.subheader("🛠️ Step 1: Split Ranges")
            df_raw_current = st.session_state["df_raw_edited"].copy()
            total_rows = len(df_raw_current)
            st.info(f"Total rows in Raw Data: **{total_rows}**.")

            num_stations = st.number_input("Number of Stations", min_value=1, max_value=10, value=1, step=1)

            default_ranges_data = []
            chunk_size = total_rows // num_stations if num_stations > 0 else total_rows
            for i in range(num_stations):
                start = i * chunk_size + 1
                end = (i + 1) * chunk_size if i < num_stations - 1 else total_rows
                default_ranges_data.append({
                    "Station Name": f"Station-{i+1}",
                    "Start Row": int(start),
                    "End Row": int(end),
                })

            edited_ranges_df = st.data_editor(
                pd.DataFrame(default_ranges_data), num_rows="fixed", use_container_width=True, hide_index=True, key="station_ranges_editor"
            )

        with st.expander("🌐 [Option A: Pre-Station] Global Raw Data Transform & Alignment (Before Splitting Stations)", expanded=False):
            st.write("### 🎛️ Global Pre-Processing Operations")
            t_action_pre = st.selectbox(
                "Select Transform Action (Pre-Station)",
                [
                    "None",
                    "Set to (0,0,0)",
                    "Custom Coord Set (Translate by delta)",
                    "Fit to horizontal plane (3D Plane Fit)",
                    "Rotate E/N plane (X/Y)",
                    "Rollback / Reset Raw Data"
                ],
                key="pre_transform_menu_action"
            )
            
            raw_point_ids = list(st.session_state["df_raw_edited"]["Point"].astype(str).str.strip())
            selected_pre_pts = []
            
            if t_action_pre == "Fit to horizontal plane (3D Plane Fit)":
                selected_pre_pts = st.multiselect("Select Target Points for 3D Plane Fit (Leave empty to use all points)", options=raw_point_ids, key="pre_tr_selected_pts")
                target_z_pre = st.number_input("Target Elevation (Z)", value=0.0, step=0.01, key="pre_tr_target_z")
            elif t_action_pre == "Custom Coord Set (Translate by delta)":
                p_dx = st.number_input("Delta X/E", value=0.0, step=1.0, key="pre_tr_dx")
                p_dy = st.number_input("Delta Y/N", value=0.0, step=1.0, key="pre_tr_dy")
                p_dz = st.number_input("Delta Z/EL", value=0.0, step=1.0, key="pre_tr_dz")
            elif "Rotate" in t_action_pre:
                p_angle = st.number_input("Rotation Angle (Degrees)", value=0.0, step=0.5, key="pre_tr_angle")
                
            if st.button("Apply Transform (Pre-Station)", key="apply_pre_transform_btn"):
                if "original_raw_df" not in st.session_state:
                    st.session_state["original_raw_df"] = st.session_state["df_raw_edited"].copy()
                cur_df = st.session_state["df_raw_edited"].copy()
                
                if t_action_pre == "Set to (0,0,0)":
                    st.session_state["df_raw_edited"] = TransformEngine.translate(cur_df, -cur_df["X/E"].mean(), -cur_df["Y/N"].mean(), -cur_df["Z/EL"].mean())
                    st.success("Pre-station: Origin set to (0,0,0).")
                elif t_action_pre == "Custom Coord Set (Translate by delta)":
                    st.session_state["df_raw_edited"] = TransformEngine.translate(cur_df, p_dx, p_dy, p_dz)
                    st.success("Pre-station translation applied.")
                elif t_action_pre == "Fit to horizontal plane (3D Plane Fit)":
                    st.session_state["df_raw_edited"] = TransformEngine.fit_to_horizontal_plane(cur_df, selected_pre_pts, target_z_pre)
                    st.success("Pre-station 3D Plane Fit applied.")
                elif "Rotate" in t_action_pre:
                    st.session_state["df_raw_edited"] = TransformEngine.rotate_plane(cur_df, "E/N plane (X/Y)", p_angle)
                    st.success("Pre-station rotation applied.")
                elif "Rollback" in t_action_pre and "original_raw_df" in st.session_state:
                    st.session_state["df_raw_edited"] = st.session_state["original_raw_df"].copy()
                    st.success("Rolled back to initial raw data.")
                st.rerun()

        station_configs = {}
        valid_ranges = True
        for _, row in edited_ranges_df.iterrows():
            s_name = str(row["Station Name"]).strip()
            try:
                r_start, r_end = int(row["Start Row"]), int(row["End Row"])
            except ValueError:
                valid_ranges = False
                break
            if r_start > r_end or r_start < 1 or r_end > total_rows:
                valid_ranges = False
            station_configs[s_name] = (r_start - 1, r_end)

        if not valid_ranges:
            st.error(f"⚠️ Please check station row ranges (must be valid integers between 1 and {total_rows}).")
        else:
            st.markdown("---")
            st.subheader("🎯 Step 2: Individual Station Fit, Post-Station Transform & Analysis")

            if "station_fitted_dfs" not in st.session_state:
                st.session_state["station_fitted_dfs"] = {}

            tabs = st.tabs(list(station_configs.keys()))

            df_raw_final_check = st.session_state["df_raw_edited"].copy()
            df_raw_final_check["Point"] = df_raw_final_check["Point"].astype(str).str.strip()

            for idx, (s_name, (s_start, s_end)) in enumerate(station_configs.items()):
                with tabs[idx]:
                    st.markdown(f"#### Managing **{s_name}** (Rows {s_start+1} to {s_end})")

                    stn_raw_df = df_raw_final_check.iloc[s_start:s_end].copy()
                    
                    stn_indexed_prep = stn_raw_df.copy()
                    stn_indexed_prep = stn_indexed_prep.loc[~stn_indexed_prep["Point"].duplicated(keep="first")]
                    stn_indexed = stn_indexed_prep.set_index("Point")
                    
                    stn_calc_indexed = stn_indexed[["X/E", "Y/N", "Z/EL"]].astype(float).copy()
                    stn_calc_indexed.columns = ["X", "Y", "Z"]

                    if df_ctrl is not None and not df_ctrl.empty:
                        common_ctrl = df_ctrl.index.intersection(stn_calc_indexed.index)
                        col_m1, col_m2 = st.columns(2)
                        fit_method_stn = col_m1.selectbox(f"Select Fit Method for {s_name}", ["2D BestFit", "3D BestFit"], key=f"method_{s_name}")
                        exclude_stn = col_m2.multiselect(f"Exclude Control Points in {s_name}", options=common_ctrl.tolist(), key=f"exclude_{s_name}")

                        active_ctrl_pts = [p for p in common_ctrl if p not in exclude_stn]

                        if len(active_ctrl_pts) >= 3:
                            if st.button(f"Execute {fit_method_stn} for {s_name}", key=f"fit_btn_{s_name}"):
                                m_pts = stn_calc_indexed.loc[active_ctrl_pts, ["X", "Y", "Z"]].values
                                d_pts = df_ctrl.loc[active_ctrl_pts, ["X", "Y", "Z"]].values

                                if "3D" in fit_method_stn:
                                    R, T = BestFitEngine.best_fit_3d(m_pts, d_pts)
                                else:
                                    R, T = BestFitEngine.best_fit_2d(m_pts, d_pts)

                                transformed_pts = np.dot(stn_calc_indexed[["X", "Y", "Z"]].values, R.T) + T
                                df_fitted = pd.DataFrame(transformed_pts, index=stn_calc_indexed.index, columns=["X", "Y", "Z"])
                                st.session_state["station_fitted_dfs"][s_name] = df_fitted

                                err_stn = BestFitEngine.calculate_error(df_fitted.loc[active_ctrl_pts], df_ctrl.loc[active_ctrl_pts])
                                st.session_state[f"err_{s_name}"] = err_stn
                                st.success(f"✅ {s_name} fitted successfully!")
                        else:
                            st.warning("⚠️ At least 3 active common control points required.")
                    else:
                        if s_name not in st.session_state["station_fitted_dfs"]:
                            st.session_state["station_fitted_dfs"][s_name] = stn_calc_indexed[["X", "Y", "Z"]]

                    with st.expander(f"🌐 [Option B: Post-Station] Transform / Adjust for {s_name}", expanded=False):
                        st.write(f"### 🎛️ Individual Post-Station Adjustments for {s_name}")
                        t_action_post = st.selectbox(
                            f"Select Transform Action for {s_name}",
                            ["None", "Translate (Delta)", "Fit to horizontal plane (3D Plane Fit)", "Rotate E/N plane (X/Y)"],
                            key=f"post_tr_action_{s_name}"
                        )
                        
                        stn_curr_pts = list(stn_calc_indexed.index)
                        selected_post_pts = []
                        if t_action_post == "Fit to horizontal plane (3D Plane Fit)":
                            selected_post_pts = st.multiselect(f"Select Points for {s_name} 3D Plane Fit (Leave empty to use all)", options=stn_curr_pts, key=f"post_tr_pts_{s_name}")
                            target_z_post = st.number_input(f"Target Z for {s_name}", value=0.0, step=0.01, key=f"post_tr_z_{s_name}")
                        elif t_action_post == "Translate (Delta)":
                            s_dx = st.number_input(f"Delta X for {s_name}", value=0.0, step=1.0, key=f"post_dx_{s_name}")
                            s_dy = st.number_input(f"Delta Y for {s_name}", value=0.0, step=1.0, key=f"post_dy_{s_name}")
                            s_dz = st.number_input(f"Delta Z for {s_name}", value=0.0, step=1.0, key=f"post_dz_{s_name}")
                        elif "Rotate" in t_action_post:
                            s_angle = st.number_input(f"Rotation Angle for {s_name}", value=0.0, step=0.5, key=f"post_angle_{s_name}")

                        if st.button(f"Apply Transform to {s_name}", key=f"apply_post_btn_{s_name}"):
                            if s_name in st.session_state["station_fitted_dfs"]:
                                target_df = st.session_state["station_fitted_dfs"][s_name]
                            else:
                                target_df = stn_calc_indexed[["X", "Y", "Z"]]
                                
                            if t_action_post == "Translate (Delta)":
                                res_tf = TransformEngine.translate(target_df, s_dx, s_dy, s_dz)
                                st.session_state["station_fitted_dfs"][s_name] = res_tf
                                st.success(f"Successfully translated {s_name}.")
                            elif t_action_post == "Fit to horizontal plane (3D Plane Fit)":
                                res_tf = TransformEngine.fit_to_horizontal_plane(target_df, selected_post_pts, target_z_post)
                                st.session_state["station_fitted_dfs"][s_name] = res_tf
                                st.success(f"Successfully applied 3D Plane Fit to {s_name}.")
                            elif "Rotate" in t_action_post:
                                res_tf = TransformEngine.rotate_plane(target_df, "E/N plane (X/Y)", s_angle)
                                st.session_state["station_fitted_dfs"][s_name] = res_tf
                                st.success(f"Successfully rotated {s_name}.")
                            st.rerun()

                    st.markdown("---")
                    col_h1, col_h2, col_h3 = st.columns(3)

                    with col_h1:
                        st.markdown(f"**📂 Raw Data ({s_name})**")
                        st.dataframe(stn_raw_df.style.format({"X/E": "{:.4f}", "Y/N": "{:.4f}", "Z/EL": "{:.4f}"}), use_container_width=True)

                    with col_h2:
                        st.markdown(f"**📈 Result Preview ({s_name}_after)**")
                        if s_name in st.session_state["station_fitted_dfs"]:
                            st.dataframe(st.session_state["station_fitted_dfs"][s_name].style.format("{:.4f}"), use_container_width=True)
                        else:
                            st.info("Pending fit execution.")

                    with col_h3:
                        st.markdown(f"**📊 Fit Deviation Analysis**")
                        if f"err_{s_name}" in st.session_state:
                            st.dataframe(
                                st.session_state[f"err_{s_name}"]
                                .style.format({"Delta E": "{:.4f}", "Delta N": "{:.4f}", "Delta El": "{:.4f}", "Total_Error": "{:.4f}"})
                                .map(highlight_excess_error, subset=["Delta E", "Delta N", "Delta El", "Total_Error"]),
                                use_container_width=True,
                            )
                        else:
                            st.info("No deviation data yet.")

                    if s_name in st.session_state["station_fitted_dfs"]:
                        stn_csv_data = st.session_state["station_fitted_dfs"][s_name].reset_index().to_csv(index=False, header=False, float_format="%.4f")
                        safe_s_name = s_name.replace(" ", "-").lower()
                        st.download_button(
                            label=f"📥 Download [{s_name}_after.CSV]",
                            data=stn_csv_data,
                            file_name=f"{safe_s_name}_after.CSV",
                            mime="text/csv",
                            key=f"dl_btn_{s_name}",
                        )

            st.markdown("---")
            st.subheader("🚀 Step 3: Merge Stations & Final BestFit with Design Points")

            if len(st.session_state["station_fitted_dfs"]) == len(station_configs):
                combined_df = pd.concat(list(st.session_state["station_fitted_dfs"].values()))
                if not combined_df.index.is_unique:
                    combined_df = combined_df.loc[~combined_df.index.duplicated(keep="first")]

                if df_design is not None and not df_design.empty:
                    common_design = df_design.index.intersection(combined_df.index)

                    col_f1, col_f2 = st.columns(2)
                    final_method = col_f1.selectbox("Select Final Fit Method", ["3D BestFit", "2D BestFit"], key="f_method")
                    final_exclude = col_f2.multiselect("Exclude Design Points for Final Fit", options=common_design.tolist(), key="f_exclude")

                    active_design_pts = [p for p in common_design if p not in final_exclude]

                    if len(active_design_pts) >= 3:
                        if st.button("✨ Execute Final Combined Fit with Design Points", type="primary", use_container_width=True):
                            m_final = combined_df.loc[active_design_pts, ["X", "Y", "Z"]].values
                            d_final = df_design.loc[active_design_pts, ["X", "Y", "Z"]].values

                            if "3D" in final_method:
                                R_final, T_final = BestFitEngine.best_fit_3d(m_final, d_final)
                            else:
                                R_final, T_final = BestFitEngine.best_fit_2d(m_final, d_final)

                            final_coords = np.dot(combined_df[["X", "Y", "Z"]].values, R_final.T) + T_final
                            df_step3_result = pd.DataFrame(final_coords, index=combined_df.index, columns=["X", "Y", "Z"])
                            st.session_state["df_step3_result"] = df_step3_result

                            err_final = BestFitEngine.calculate_error(df_step3_result.loc[active_design_pts], df_design.loc[active_design_pts])
                            st.session_state["err_step3"] = err_final
                            
                            st.success("🎉 Final Combined BestFit completed successfully!")
                    else:
                        st.warning("⚠️ At least 3 active common design points are required.")
                else:
                    st.session_state["df_step3_result"] = combined_df
                    st.info("ℹ️ Design Points not provided. Merged stations are ready as Step 3 result.")

                if "df_step3_result" in st.session_state:
                    st.markdown("---")
                    st.subheader("📋 Step 3: Final Result Preview & Deviation Analysis")
                    
                    col_s3_1, col_s3_2 = st.columns(2)
                    with col_s3_1:
                        st.markdown("#### 📋 Final Result Preview (Step 3)")
                        st.dataframe(st.session_state["df_step3_result"].style.format("{:.4f}"), use_container_width=True)

                    with col_s3_2:
                        st.markdown("#### 📊 Final Deviation Analysis (Step 3)")
                        if "err_step3" in st.session_state:
                            st.dataframe(
                                st.session_state["err_step3"]
                                .style.format({"Delta E": "{:.4f}", "Delta N": "{:.4f}", "Delta El": "{:.4f}", "Total_Error": "{:.4f}"})
                                .map(highlight_excess_error, subset=["Delta E", "Delta N", "Delta El", "Total_Error"]),
                                use_container_width=True,
                            )
                        else:
                            st.info("No deviation data available (Design points not used).")

                    step3_csv = st.session_state["df_step3_result"].reset_index().to_csv(index=False, header=False, float_format="%.4f")
                    st.download_button(
                        label="📥 Download Step 3 Final Result [station-Combine All_after BestFit.CSV]",
                        data=step3_csv,
                        file_name="station-Combine_All_after_BestFit_Result.CSV",
                        mime="text/csv",
                    )

                st.markdown("---")
                st.subheader("🎯 Step 4: Transform / Adjust Merged Stations & Post-Adjustment Visuals & CAD Export")
                
                with st.expander("🌐 [Option C: Final Stage] Transform / Adjust Merged Stations", expanded=True):
                    st.write("### 🎛️ Final Stage Adjustments on Merged Data")
                    t_action_final = st.selectbox(
                        "Select Transform Action for Merged Data",
                        ["None", "Translate (Delta)", "Fit to horizontal plane (3D Plane Fit)", "Rotate E/N plane (X/Y)"],
                        key="final_tr_action"
                    )
                    
                    base_df_for_step4 = st.session_state.get("df_step3_result", combined_df)
                    combined_pts = list(base_df_for_step4.index)
                    selected_final_pts = []
                    
                    if t_action_final == "Fit to horizontal plane (3D Plane Fit)":
                        selected_final_pts = st.multiselect("Select Points for Final 3D Plane Fit (Leave empty to use all)", options=combined_pts, key="final_tr_pts")
                        target_z_final = st.number_input("Target Z for Final Stage", value=0.0, step=0.01, key="final_tr_z")
                    elif t_action_final == "Translate (Delta)":
                        f_dx = st.number_input("Delta X for Merged Data", value=0.0, step=1.0, key="final_dx")
                        f_dy = st.number_input("Delta Y for Merged Data", value=0.0, step=1.0, key="final_dy")
                        f_dz = st.number_input("Delta Z for Merged Data", value=0.0, step=1.0, key="final_dz")
                    elif "Rotate" in t_action_final:
                        f_angle = st.number_input("Rotation Angle for Merged Data", value=0.0, step=0.5, key="final_angle")

                    if st.button("Apply Transform to Merged Data (Trigger Step 4 Output)", key="apply_final_tr_btn", type="primary"):
                        if t_action_final == "None":
                            st.session_state["df_final_result"] = base_df_for_step4.copy()
                            if "err_step3" in st.session_state:
                                st.session_state["err_final"] = st.session_state["err_step3"].copy()
                            st.success("Applied 'None' (Copied Step 3 result to Step 4).")
                        else:
                            if t_action_final == "Translate (Delta)":
                                res_tf = TransformEngine.translate(base_df_for_step4, f_dx, f_dy, f_dz)
                            elif t_action_final == "Fit to horizontal plane (3D Plane Fit)":
                                res_tf = TransformEngine.fit_to_horizontal_plane(base_df_for_step4, selected_final_pts, target_z_final)
                            elif "Rotate" in t_action_final:
                                res_tf = TransformEngine.rotate_plane(base_df_for_step4, "E/N plane (X/Y)", f_angle)
                            else:
                                res_tf = base_df_for_step4.copy()
                                
                            st.session_state["df_final_result"] = res_tf
                            if df_design is not None and not df_design.empty:
                                common_d_f = df_design.index.intersection(res_tf.index)
                                if len(common_d_f) > 0:
                                    st.session_state["err_final"] = BestFitEngine.calculate_error(res_tf.loc[common_d_f], df_design.loc[common_d_f])
                            st.success("Successfully applied Step 4 transformation and generated custom analysis/preview!")
                        st.rerun()

                if "df_step3_result" in st.session_state:
                    st.markdown("---")
                    st.subheader("📐 CAD Layout Preview & DXF/SCR Converter (Source Selection)")
                    
                    cad_source_choice = st.radio(
                        "Select Data Source for CAD Layout & Export",
                        ["Step 3 Result (Merged & BestFit)", "Step 4 Result (Custom Transformed/Adjusted)"],
                        index=0 if "df_final_result" not in st.session_state else 1,
                        horizontal=True,
                        key="cad_source_choice_radio"
                    )
                    
                    if "Step 3" in cad_source_choice:
                        active_cad_df = st.session_state["df_step3_result"]
                        active_err_df = st.session_state.get("err_step3", pd.DataFrame())
                        source_label_str = "Step3"
                    else:
                        active_cad_df = st.session_state.get("df_final_result", st.session_state["df_step3_result"])
                        active_err_df = st.session_state.get("err_final", pd.DataFrame())
                        source_label_str = "Step4"

                    if "df_final_result" in st.session_state:
                        st.markdown("### 📊 Active Source Preview & Deviation Analysis")
                        col_res1, col_res2 = st.columns(2)

                        with col_res1:
                            st.markdown(f"#### 📋 {source_label_str} Result Preview")
                            st.dataframe(active_cad_df.style.format("{:.4f}"), use_container_width=True)

                        with col_res2:
                            st.markdown(f"#### 📊 {source_label_str} Deviation Analysis")
                            if not active_err_df.empty:
                                st.dataframe(
                                    active_err_df
                                    .style.format({"Delta E": "{:.4f}", "Delta N": "{:.4f}", "Delta El": "{:.4f}", "Total_Error": "{:.4f}"})
                                    .map(highlight_excess_error, subset=["Delta E", "Delta N", "Delta El", "Total_Error"]),
                                    use_container_width=True,
                                )
                            else:
                                st.info("No deviation data available for this source.")

                        step_csv_data = active_cad_df.reset_index().to_csv(index=False, header=False, float_format="%.4f")
                        st.download_button(
                            label=f"📥 Download [{source_label_str}_Result.CSV]",
                            data=step_csv_data,
                            file_name=f"station-Combine_All_{source_label_str}_Result.CSV",
                            mime="text/csv",
                        )

                    st.markdown("---")
                    st.markdown(f"### 🖥️ Live Layout Preview & Export (Using **{source_label_str}**)")

                    dxf_df = active_cad_df.reset_index()
                    dxf_df.columns = ["ID", "X", "Y", "Z"]

                    st.write("### 🛠️ Step A: Label Display Settings")
                    display_options = st.multiselect(
                        "Select what to display in the label:",
                        ["ID", "X Coordinate", "Y Coordinate", "Elevation (EL)"],
                        default=["ID", "X Coordinate", "Y Coordinate", "Elevation (EL)"],
                        key="dxf_display_options",
                    )

                    with st.expander("⚙️ Advanced Settings (Heights, Offsets, Colors & Point Style)", expanded=False):
                        decimal_places = st.selectbox("Decimal Places for Coordinates / EL", [3, 4], index=0, key="dxf_dec")
                        point_color = st.selectbox("Point Symbol Color", list(CAD_COLORS.keys()), index=0, key="dxf_pt_color")

                        st.markdown("---")
                        st.write("🎛️ **Individual Field Configurations**")
                        field_configs = {}
                        for field in ["ID", "X Coordinate", "Y Coordinate", "Elevation (EL)"]:
                            if field in display_options:
                                st.markdown(f"**📌 {field} Configuration**")
                                c1, c2, c3, c4 = st.columns(4)
                                with c1:
                                    h_val = st.number_input(f"{field} Height", value=1.0, step=0.1, key=f"h_{field}")
                                with c2:
                                    ox_val = st.number_input(f"{field} X Offset", value=0.5, step=0.1, key=f"ox_{field}")
                                with c3:
                                    oy_val = st.number_input(f"{field} Y Offset", value=0.5, step=0.1, key=f"oy_{field}")
                                with c4:
                                    default_c_idx = 2 if field == "Elevation (EL)" else 0
                                    c_val = st.selectbox(f"{field} Color", list(CAD_COLORS.keys()), index=default_c_idx, key=f"c_{field}")

                                field_configs[field] = {
                                    "height": h_val,
                                    "offset_x": ox_val,
                                    "offset_y": oy_val,
                                    "color_name": CAD_COLORS[c_val][0],
                                    "color_idx": CAD_COLORS[c_val][1],
                                }

                        st.markdown("---")
                        st.write("📍 **CAD Point Symbol Settings**")
                        point_style_options = {
                            "Dot (.)": 0,
                            "Plus (+)": 2,
                            "X Shape": 3,
                            "Circle (○)": 32,
                            "Square (□)": 64,
                            "Circle & Cross (◎)": 34,
                        }
                        pdmode_val = st.selectbox("Point Symbol Type", list(point_style_options.keys()), index=5, key="dxf_pdmode")
                        pdsize_val = st.number_input("Point Size", value=1.5, step=0.2, key="dxf_pdsize")

                    st.markdown("---")
                    st.markdown("### 🖥️ Live Layout Preview (Supports Mouse Roller Zoom & Pan)")

                    valid_xs, valid_ys = [], []
                    for _, row in dxf_df.iterrows():
                        try:
                            valid_xs.append(float(row["X"]))
                            valid_ys.append(float(row["Y"]))
                        except:
                            continue

                    if valid_xs and valid_ys:
                        fig = go.Figure()
                        fig.add_trace(
                            go.Scatter(
                                x=dxf_df["X"],
                                y=dxf_df["Y"],
                                mode="markers",
                                marker=dict(color=CAD_COLORS[point_color][0], size=max(6, pdsize_val * 6), symbol="circle"),
                                text=dxf_df["ID"],
                                name="Points",
                                hoverinfo="text+x+y",
                            )
                        )

                        for idx, row in dxf_df.iterrows():
                            try:
                                x_val, y_val, z_val = float(row["X"]), float(row["Y"]), float(row["Z"])
                                id_val = str(row["ID"])
                                fmt = f"{{:.{decimal_places}f}}"

                                line_spacing_offset = 0.0
                                for field in display_options:
                                    if field not in field_configs:
                                        continue
                                    cfg = field_configs[field]

                                    if field == "ID":
                                        text_content = id_val
                                    elif field == "X Coordinate":
                                        text_content = f"X: {fmt.format(x_val)}"
                                    elif field == "Y Coordinate":
                                        text_content = f"Y: {fmt.format(y_val)}"
                                    else:
                                        text_content = f"EL: {fmt.format(z_val)}"

                                    fx = x_val + cfg["offset_x"]
                                    fy = y_val + cfg["offset_y"] - line_spacing_offset

                                    fig.add_annotation(
                                        x=fx,
                                        y=fy,
                                        text=text_content,
                                        showarrow=False,
                                        font=dict(color=cfg["color_name"], size=max(10, cfg["height"] * 9)),
                                        xanchor="left",
                                        yanchor="bottom",
                                    )
                                    line_spacing_offset += cfg["height"] * 0.8
                            except:
                                continue

                        fig.update_layout(
                            paper_bgcolor="black",
                            plot_bgcolor="black",
                            xaxis_title="X Coordinate",
                            yaxis_title="Y Coordinate",
                            xaxis=dict(showgrid=True, gridcolor="rgba(50,50,50,0.8)", zeroline=True, zerolinecolor="rgba(100,100,100,0.8)"),
                            yaxis=dict(showgrid=True, gridcolor="rgba(50,50,50,0.8)", zeroline=True, zerolinecolor="rgba(100,100,100,0.8)", scaleanchor="x", scaleratio=1),
                            margin=dict(l=20, r=20, t=20, b=20),
                            height=700,
                            hovermode="closest",
                        )

                        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})
                    else:
                        st.warning("No coordinate values detected.")

                    doc = ezdxf.new(dxfversion="R2010")
                    msp = doc.modelspace()
                    doc.header["$PDMODE"] = point_style_options[pdmode_val]
                    doc.header["$PDSIZE"] = pdsize_val

                    scr_lines = ["ucs W", "Osnapcoord 1"]

                    for idx, row in dxf_df.iterrows():
                        try:
                            x_val, y_val, z_val = float(row["X"]), float(row["Y"]), float(row["Z"])
                            id_val = str(row["ID"])
                            fmt = f"{{:.{decimal_places}f}}"

                            msp.add_point((x_val, y_val, z_val), dxfattribs={"color": CAD_COLORS[point_color][1]})

                            scr_lines.append("SPHERE")
                            scr_lines.append(f"{x_val:.7f},{y_val:.7f},{z_val:.7f}")
                            scr_lines.append("D")
                            scr_lines.append(f"{pdsize_val * 0.01:.5f}")

                            line_spacing_offset = 0.0
                            for field in display_options:
                                if field not in field_configs:
                                    continue
                                cfg = field_configs[field]

                                if field == "ID":
                                    text_content = id_val
                                elif field == "X Coordinate":
                                    text_content = f"X: {fmt.format(x_val)}"
                                elif field == "Y Coordinate":
                                    text_content = f"Y: {fmt.format(y_val)}"
                                else:
                                    text_content = f"EL: {fmt.format(z_val)}"

                                fx = x_val + cfg["offset_x"]
                                fy = y_val + cfg["offset_y"] - line_spacing_offset

                                msp.add_text(
                                    text_content,
                                    dxfattribs={"insert": (fx, fy, z_val), "height": cfg["height"], "color": cfg["color_idx"]},
                                )

                                scr_lines.append(f"-TEXT {fx:.6f},{fy:.6f},{z_val:.6f} {cfg['height']:.4f} 0 {text_content}")
                                line_spacing_offset += cfg["height"] * 1.3
                        except:
                            continue

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
                        doc.saveas(tmp.name)
                        with open(tmp.name, "rb") as f:
                            dxf_data = f.read()
                    os.unlink(tmp.name)

                    scr_content = "\n".join(scr_lines)
                    scr_data = scr_content.encode("utf-8")

                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        st.download_button(
                            f"⬇️ Download {source_label_str} DXF File",
                            data=dxf_data,
                            file_name=f"{source_label_str.lower()}_station_layout.dxf",
                            mime="application/dxf",
                            use_container_width=True,
                        )
                    with col_dl2:
                        st.download_button(
                            f"⬇️ Download {source_label_str} AutoCAD Script (.SCR)",
                            data=scr_data,
                            file_name=f"{source_label_str.lower()}_station_layout.scr",
                            mime="text/plain",
                            use_container_width=True,
                        )

                    st.markdown("---")
                    st.subheader(f"📄 Comprehensive PDF Report Export ({source_label_str})")

                    include_dev_in_pdf = st.checkbox("Include '3. Deviation Analysis Report' in PDF", value=True, key="include_dev_pdf_checkbox")

                    pdf_bytes = generate_pdf_report(
                        active_cad_df,
                        active_err_df,
                        include_deviation=include_dev_in_pdf
                    )
                    st.download_button(
                        label=f"📥 Download {source_label_str} Comprehensive PDF Report",
                        data=pdf_bytes,
                        file_name=f"{source_label_str}_BestFit_Comprehensive_Report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
            else:
                st.info("👉 Please complete the individual steps for **all** defined stations in Step 2 before proceeding.")

    except Exception as e:
        st.error(f"Processing error: {e}")
else:
    st.info("👈 Please upload **Raw Data CSV** in the sidebar to start.")

                        min_dev = np.min(z_vals)
                        total_flatness = max_dev - min_dev
                        
                        # 如果在 Step 4 设定了目标 Z 值（target_z_final），则以该目标基准计算 RMS；否则默认以 0.0 为基准
                        tz = target_z_final if (source_label_str == "Step4" and 'target_z_final' in locals()) else 0.0
                        rms = np.sqrt(np.mean((z_vals - tz)**2))
                        
                        st.success(
                            f"**📏 Dimensional Control Flatness Report ({len(z_vals)} points evaluated):**\n\n"
                            f"🔹 **Max Positive Deviation:** `{max_dev:+.4f}`  \n"
                            f"🔹 **Max Negative Deviation:** `{min_dev:+.4f}`  \n"
                            f"🔹 **Total Flatness (Max - Min):** `{total_flatness:.4f}`  \n"
                            f"🔹 **RMS (Root Mean Square):** `{rms:.4f}`"
                        )
                        # ------------------------------------------------------------

                        col_res1, col_res2 = st.columns(2)

                        with col_res1:
                            st.markdown(f"#### 📋 {source_label_str} Result Preview")
                            st.dataframe(active_cad_df.style.format("{:.4f}"), use_container_width=True)

                        with col_res2:
                            st.markdown(f"#### 📊 {source_label_str} Deviation Analysis")
                            if not active_err_df.empty:
                                st.dataframe(
                                    active_err_df
                                    .style.format({"Delta E": "{:.4f}", "Delta N": "{:.4f}", "Delta El": "{:.4f}", "Total_Error": "{:.4f}"})
                                    .map(highlight_excess_error, subset=["Delta E", "Delta N", "Delta El", "Total_Error"]),
                                    use_container_width=True,
                                )
                            else:
                                st.info("No deviation data available for this source.")

                        step_csv_data = active_cad_df.reset_index().to_csv(index=False, header=False, float_format="%.4f")
                        st.download_button(
                            label=f"📥 Download [{source_label_str}_Result.CSV]",
                            data=step_csv_data,
                            file_name=f"station-Combine_All_{source_label_str}_Result.CSV",
                            mime="text/csv",
                        )
