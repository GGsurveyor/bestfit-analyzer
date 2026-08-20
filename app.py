import os
import ezdxf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
        return err


# CAD Color Mapping
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
    page_title="BestFit & DXF/SCR Converter Pro", page_icon="🏗️", layout="wide"
)

st.title(
    "🏗️ Multi-Station BestFit Pipeline & CAD DXF/SCR Converter - Made by Ng"
    " Yit Fung"
)
st.markdown(
    "Complete Raw Data editing, Station splitting, BestFit analysis, and"
    " export to CSV / DXF / SCR formats."
)

# Sidebar Uploads
st.sidebar.header("📂 Data Inputs")
uploaded_design = st.sidebar.file_uploader(
    "Upload Design Points CSV (Optional)", type=["csv"], key="design_file"
)
uploaded_ctrl = st.sidebar.file_uploader(
    "Upload Control Points CSV (Optional)", type=["csv"], key="ctrl_file"
)
uploaded_raw = st.sidebar.file_uploader(
    "Upload Raw Data CSV (Required)", type=["csv"], key="raw_file"
)

if uploaded_raw is not None:
    try:
        df_design = None
        if uploaded_design is not None:
            df_design = pd.read_csv(
                uploaded_design, header=None, names=["Point", "X", "Y", "Z"]
            )
            df_design["Point"] = df_design["Point"].astype(str).str.strip()
            df_design.set_index("Point", inplace=True)

        df_ctrl = None
        if uploaded_ctrl is not None:
            df_ctrl = pd.read_csv(
                uploaded_ctrl, header=None, names=["Point", "X", "Y", "Z"]
            )
            df_ctrl["Point"] = df_ctrl["Point"].astype(str).str.strip()
            df_ctrl.set_index("Point", inplace=True)

        df_raw_initial = pd.read_csv(
            uploaded_raw, header=None, names=["Point", "X", "Y", "Z"]
        )

        if "df_raw_edited" not in st.session_state:
            temp_init = df_raw_initial.copy()
            temp_init.index = range(1, len(temp_init) + 1)
            st.session_state["df_raw_edited"] = temp_init

        # --- Step 0 & Step 1 横向布局 ---
        st.markdown("---")
        col_main1, col_main2 = st.columns(2)

        with col_main1:
            st.subheader("✏️ Step 0: Raw Data Editor")
            current_df = st.session_state["df_raw_edited"].copy()
            current_df.index = range(1, len(current_df) + 1)

            edited_raw_df = st.data_editor(
                current_df,
                num_rows="dynamic",
                use_container_width=True,
                hide_index=False,
                column_config={
                    "Point": st.column_config.TextColumn("Point", width="medium"),
                    "X": st.column_config.NumberColumn("X", format="%.4f"),
                    "Y": st.column_config.NumberColumn("Y", format="%.4f"),
                    "Z": st.column_config.NumberColumn("Z", format="%.4f"),
                },
                key="raw_data_editor",
            )
            st.session_state["df_raw_edited"] = edited_raw_df

        df_raw = st.session_state["df_raw_edited"].copy()
        df_raw["Point"] = df_raw["Point"].astype(str).str.strip()
        total_rows = len(df_raw)

        with col_main2:
            st.subheader("🛠️ Step 1: Split Ranges")
            st.info(f"Total rows in Raw Data: **{total_rows}**.")

            num_stations = st.number_input(
                "Number of Stations", min_value=1, max_value=10, value=1, step=1
            )

            default_ranges_data = []
            chunk_size = total_rows // num_stations
            for i in range(num_stations):
                start = i * chunk_size + 1
                end = (
                    (i + 1) * chunk_size
                    if i < num_stations - 1
                    else total_rows
                )
                default_ranges_data.append({
                    "Station Name": f"Station-{i+1}",
                    "Start Row": int(start),
                    "End Row": int(end),
                })

            edited_ranges_df = st.data_editor(
                pd.DataFrame(default_ranges_data),
                num_rows="fixed",
                use_container_width=True,
                hide_index=True,
                key="station_ranges_editor",
            )

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
            st.error(
                "⚠️ Please check station row ranges (must be valid integers"
                f" between 1 and {total_rows})."
            )
        else:
            # --- Step 2: Individual Station Fit ---
            st.markdown("---")
            st.subheader(
                "🎯 Step 2: Individual Station Fit, Deviation Analysis &"
                " Download"
            )

            if "station_fitted_dfs" not in st.session_state:
                st.session_state["station_fitted_dfs"] = {}

            tabs = st.tabs(list(station_configs.keys()))

            for idx, (s_name, (s_start, s_end)) in enumerate(
                station_configs.items()
            ):
                with tabs[idx]:
                    st.markdown(
                        f"#### Managing **{s_name}** (Rows {s_start+1} to"
                        f" {s_end})"
                    )

                    stn_raw_df = df_raw.iloc[s_start:s_end].copy()
                    stn_indexed = stn_raw_df.set_index("Point")

                    if df_ctrl is not None:
                        common_ctrl = df_ctrl.index.intersection(
                            stn_indexed.index
                        )
                        col_m1, col_m2 = st.columns(2)
                        fit_method_stn = col_m1.selectbox(
                            f"Select Fit Method for {s_name}",
                            ["2D BestFit", "3D BestFit"],
                            key=f"method_{s_name}",
                        )
                        exclude_stn = col_m2.multiselect(
                            f"Exclude Control Points in {s_name}",
                            options=common_ctrl.tolist(),
                            key=f"exclude_{s_name}",
                        )

                        active_ctrl_pts = [
                            p for p in common_ctrl if p not in exclude_stn
                        ]
                        st.info(
                            f"Total common control points: {len(common_ctrl)} |"
                            f" Active for fit: {len(active_ctrl_pts)}"
                        )

                        if len(active_ctrl_pts) >= 3:
                            if st.button(
                                f"Execute {fit_method_stn} for {s_name}",
                                key=f"fit_btn_{s_name}",
                            ):
                                m_pts = stn_indexed.loc[
                                    active_ctrl_pts, ["X", "Y", "Z"]
                                ].values
                                d_pts = df_ctrl.loc[
                                    active_ctrl_pts, ["X", "Y", "Z"]
                                ].values

                                if "3D" in fit_method_stn:
                                    R, T = BestFitEngine.best_fit_3d(m_pts, d_pts)
                                else:
                                    R, T = BestFitEngine.best_fit_2d(m_pts, d_pts)

                                transformed_pts = (
                                    np.dot(
                                        stn_indexed[["X", "Y", "Z"]].values, R.T
                                    )
                                    + T
                                )
                                df_fitted = pd.DataFrame(
                                    transformed_pts,
                                    index=stn_indexed.index,
                                    columns=["X", "Y", "Z"],
                                )
                                st.session_state["station_fitted_dfs"][
                                    s_name
                                ] = df_fitted

                                df_fitted_active = df_fitted.loc[
                                    active_ctrl_pts
                                ]
                                err_stn = BestFitEngine.calculate_error(
                                    df_fitted_active,
                                    df_ctrl.loc[active_ctrl_pts],
                                )
                                st.session_state[f"err_{s_name}"] = err_stn
                                st.success(
                                    f"✅ {s_name} fitted successfully with"
                                    f" {fit_method_stn}!"
                                )
                        else:
                            st.warning(
                                "⚠️ At least 3 active common control points"
                                " are required to fit."
                            )
                    else:
                        if s_name not in st.session_state["station_fitted_dfs"]:
                            st.session_state["station_fitted_dfs"][s_name] = (
                                stn_indexed[["X", "Y", "Z"]]
                            )
                        st.info(
                            "ℹ️ Control Points not uploaded. Using raw data"
                            " directly for this station."
                        )

                    st.markdown("---")
                    col_h1, col_h2, col_h3 = st.columns(3)

                    with col_h1:
                        st.markdown(f"**📂 Raw Data ({s_name})**")
                        st.dataframe(
                            stn_raw_df.style.format({
                                "X": "{:.4f}",
                                "Y": "{:.4f}",
                                "Z": "{:.4f}",
                            }),
                            use_container_width=True,
                        )

                    with col_h2:
                        st.markdown(f"**📈 Result Preview ({s_name}_after)**")
                        if s_name in st.session_state["station_fitted_dfs"]:
                            st.dataframe(
                                st.session_state["station_fitted_dfs"][
                                    s_name
                                ].style.format("{:.4f}"),
                                use_container_width=True,
                            )
                        else:
                            st.info("Pending fit execution.")

                    with col_h3:
                        st.markdown(f"**📊 Fit Deviation Analysis**")
                        if f"err_{s_name}" in st.session_state:
                            st.dataframe(
                                st.session_state[f"err_{s_name}"].style.format({
                                    "Delta E": "{:.4f}",
                                    "Delta N": "{:.4f}",
                                    "Delta El": "{:.4f}",
                                    "Total_Error": "{:.4f}",
                                }),
                                use_container_width=True,
                            )
                        else:
                            st.info("No deviation data yet.")

                    if s_name in st.session_state["station_fitted_dfs"]:
                        stn_csv_data = (
                            st.session_state["station_fitted_dfs"][s_name]
                            .reset_index()
                            .to_csv(index=False, header=False, float_format="%.4f")
                        )
                        safe_s_name = s_name.replace(" ", "-").lower()
                        st.download_button(
                            label=f"📥 Download [{s_name}_after.CSV]",
                            data=stn_csv_data,
                            file_name=f"{safe_s_name}_after.CSV",
                            mime="text/csv",
                            key=f"dl_btn_{s_name}",
                        )

            # --- Step 3: Combine All Stations & Final BestFit ---
            st.markdown("---")
            st.subheader(
                "🚀 Step 3: Merge Stations & Final BestFit with Design Points"
            )

            if len(st.session_state["station_fitted_dfs"]) == len(
                station_configs
            ):
                combined_df = pd.concat(
                    list(st.session_state["station_fitted_dfs"].values())
                )

                if df_design is not None:
                    common_design = df_design.index.intersection(
                        combined_df.index
                    )

                    col_f1, col_f2 = st.columns(2)
                    final_method = col_f1.selectbox(
                        "Select Final Fit Method",
                        ["3D BestFit", "2D BestFit"],
                        key="f_method",
                    )
                    final_exclude = col_f2.multiselect(
                        "Exclude Design Points for Final Fit",
                        options=common_design.tolist(),
                        key="f_exclude",
                    )

                    active_design_pts = [
                        p for p in common_design if p not in final_exclude
                    ]
                    st.info(
                        f"Combined common design points: {len(common_design)} |"
                        f" Active for final fit: {len(active_design_pts)}"
                    )

                    if len(active_design_pts) >= 3:
                        if st.button(
                            "✨ Execute Final Combined Fit with Design Points",
                            type="primary",
                            use_container_width=True,
                        ):
                            m_final = combined_df.loc[
                                active_design_pts, ["X", "Y", "Z"]
                            ].values
                            d_final = df_design.loc[
                                active_design_pts, ["X", "Y", "Z"]
                            ].values

                            if "3D" in final_method:
                                R_final, T_final = BestFitEngine.best_fit_3d(
                                    m_final, d_final
                                )
                            else:
                                R_final, T_final = BestFitEngine.best_fit_2d(
                                    m_final, d_final
                                )

                            final_coords = (
                                np.dot(
                                    combined_df[["X", "Y", "Z"]].values,
                                    R_final.T,
                                )
                                + T_final
                            )
                            df_final_result = pd.DataFrame(
                                final_coords,
                                index=combined_df.index,
                                columns=["X", "Y", "Z"],
                            )
                            st.session_state["df_final_result"] = df_final_result

                            df_final_active = df_final_result.loc[
                                active_design_pts
                            ]
                            err_final = BestFitEngine.calculate_error(
                                df_final_active,
                                df_design.loc[active_design_pts],
                            )
                            st.session_state["err_final"] = err_final
                            st.success(
                                "🎉 Final Combined BestFit completed successfully!"
                            )
                    else:
                        st.warning(
                            "⚠️ At least 3 active common design points are"
                            " required."
                        )
                else:
                    st.session_state["df_final_result"] = combined_df
                    st.info(
                        "ℹ️ Design Points not uploaded. Merged stations are"
                        " ready for DXF/SCR preview and conversion below."
                    )

                if "df_final_result" in st.session_state:
                    st.markdown("#### 📋 Final Result Preview")
                    st.dataframe(
                        st.session_state["df_final_result"].style.format(
                            "{:.4f}"
                        )
                    )

                    if "err_final" in st.session_state:
                        st.markdown(
                            "📊 **Final Deviation Analysis (Design Points):**"
                        )
                        st.dataframe(
                            st.session_state["err_final"].style.format({
                                "Delta E": "{:.4f}",
                                "Delta N": "{:.4f}",
                                "Delta El": "{:.4f}",
                                "Total_Error": "{:.4f}",
                            })
                        )

                    final_csv = (
                        st.session_state["df_final_result"]
                        .reset_index()
                        .to_csv(index=False, header=False, float_format="%.4f")
                    )
                    st.download_button(
                        label=(
                            "📥 Download Final Result"
                            " [station-Combine All_after BestFit.CSV]"
                        ),
                        data=final_csv,
                        file_name=(
                            "station-Combine_All_after_BestFit_Result.CSV"
                        ),
                        mime="text/csv",
                    )

                    # --- Integration of CAD / DXF & SCR Converter & Preview ---
                    st.markdown("---")
                    st.subheader(
                        "📐 CAD Layout Preview & DXF/SCR Converter (From Result)"
                    )

                    dxf_df = st.session_state[
                        "df_final_result"
                    ].reset_index()
                    dxf_df.columns = ["ID", "X", "Y", "Z"]

                    st.write("### 🛠️ Step A: Label Display Settings")
                    display_options = st.multiselect(
                        "Select what to display in the label:",
                        [
                            "ID",
                            "X Coordinate",
                            "Y Coordinate",
                            "Elevation (EL)",
                        ],
                        default=[
                            "ID",
                            "X Coordinate",
                            "Y Coordinate",
                            "Elevation (EL)",
                        ],
                        key="dxf_display_options",
                    )

                    with st.expander(
                        "⚙️ Advanced Settings (Heights, Offsets, Colors & Point"
                        " Style)",
                        expanded=False,
                    ):
                        decimal_places = st.selectbox(
                            "Decimal Places for Coordinates / EL",
                            [3, 4],
                            index=0,
                            key="dxf_dec",
                        )
                        point_color = st.selectbox(
                            "Point Symbol Color",
                            list(CAD_COLORS.keys()),
                            index=0,
                            key="dxf_pt_color",
                        )

                        st.markdown("---")
                        st.write("🎛️ **Individual Field Configurations**")
                        field_configs = {}
                        for field in [
                            "ID",
                            "X Coordinate",
                            "Y Coordinate",
                            "Elevation (EL)",
                        ]:
                            if field in display_options:
                                st.markdown(f"**📌 {field} Configuration**")
                                c1, c2, c3, c4 = st.columns(4)
                                with c1:
                                    h_val = st.number_input(
                                        f"{field} Height",
                                        value=1.0,
                                        step=0.1,
                                        key=f"h_{field}",
                                    )
                                with c2:
                                    ox_val = st.number_input(
                                        f"{field} X Offset",
                                        value=0.5,
                                        step=0.1,
                                        key=f"ox_{field}",
                                    )
                                with c3:
                                    oy_val = st.number_input(
                                        f"{field} Y Offset",
                                        value=0.5,
                                        step=0.1,
                                        key=f"oy_{field}",
                                    )
                                with c4:
                                    default_c_idx = (
                                        2 if field == "Elevation (EL)" else 0
                                    )
                                    c_val = st.selectbox(
                                        f"{field} Color",
                                        list(CAD_COLORS.keys()),
                                        index=default_c_idx,
                                        key=f"c_{field}",
                                    )

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
                        pdmode_val = st.selectbox(
                            "Point Symbol Type",
                            list(point_style_options.keys()),
                            index=5,
                            key="dxf_pdmode",
                        )
                        pdsize_val = st.number_input(
                            "Point Size", value=1.5, step=0.2, key="dxf_pdsize"
                        )

                    # Live Preview Window
                    st.markdown("---")
                    st.markdown("### 🖥️ Live Layout Preview")

                    fig, ax = plt.subplots(figsize=(10, 8))
                    fig.patch.set_facecolor("#0e1117")
                    ax.set_facecolor("#0e1117")

                    has_valid_data = False
                    for idx, row in dxf_df.iterrows():
                        try:
                            x_val, y_val, z_val = (
                                float(row["X"]),
                                float(row["Y"]),
                                float(row["Z"]),
                            )
                            id_val = str(row["ID"])
                            fmt = f"{{:.{decimal_places}f}}"
                            has_valid_data = True

                            ax.scatter(
                                [x_val],
                                [y_val],
                                color=CAD_COLORS[point_color][0],
                                s=pdsize_val * 20,
                                marker="o",
                            )

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
                                fy = (
                                    y_val
                                    + cfg["offset_y"]
                                    - line_spacing_offset
                                )
                                ax.text(
                                    fx,
                                    fy,
                                    text_content,
                                    color=cfg["color_name"],
                                    fontsize=max(
                                        8, cfg["height"] * 6
                                    ),
                                )
                                line_spacing_offset += cfg["height"] * 0.8
                        except:
                            continue

                    if has_valid_data:
                        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
                        ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
                        ax.tick_params(colors="white")
                        ax.xaxis.label.set_color("white")
                        ax.yaxis.label.set_color("white")
                        for spine in ax.spines.values():
                            spine.set_edgecolor("gray")
                        ax.set_xlabel("X Coordinate")
                        ax.set_ylabel("Y Coordinate")
                        ax.grid(True, linestyle=":", alpha=0.3, color="gray")
                        ax.set_aspect("equal", adjustable="datalim")
                        st.pyplot(fig)
                    else:
                        st.warning(
                            "No valid coordinate data available to render."
                        )

                    # --- Generate DXF & SCR Files ---
                    doc = ezdxf.new(dxfversion="R2010")
                    msp = doc.modelspace()
                    doc.header["$PDMODE"] = point_style_options[pdmode_val]
                    doc.header["$PDSIZE"] = pdsize_val

                    scr_lines = [
                        "ucs W",
                        "Osnapcoord 1",
                    ]

                    for idx, row in dxf_df.iterrows():
                        try:
                            x_val, y_val, z_val = (
                                float(row["X"]),
                                float(row["Y"]),
                                float(row["Z"]),
                            )
                            id_val = str(row["ID"])
                            fmt = f"{{:.{decimal_places}f}}"

                            # 1. DXF 实体生成
                            msp.add_point(
                                (x_val, y_val, z_val),
                                dxfattribs={
                                    "color": CAD_COLORS[point_color][1]
                                },
                            )

                            # 2. SCR 脚本：使用分行的 SPHERE 3D 指令
                            scr_lines.append("SPHERE")
                            scr_lines.append(
                                f"{x_val:.7f},{y_val:.7f},{z_val:.7f}"
                            )
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
                                fy = (
                                    y_val
                                    + cfg["offset_y"]
                                    - line_spacing_offset
                                )

                                # DXF 文本写入
                                msp.add_text(
                                    text_content,
                                    dxfattribs={
                                        "insert": (fx, fy, z_val),
                                        "height": cfg["height"],
                                        "color": cfg["color_idx"],
                                    },
                                )

                                # SCR 脚本：使用稳定的 -TEXT 命令
                                scr_lines.append(
                                    f"-TEXT {fx:.6f},{fy:.6f},{z_val:.6f}"
                                    f" {cfg['height']:.4f} 0 {text_content}"
                                )

                                line_spacing_offset += cfg["height"] * 1.3

                        except:
                            continue

                    import tempfile

                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".dxf"
                    ) as tmp:
                        doc.saveas(tmp.name)
                        with open(tmp.name, "rb") as f:
                            dxf_data = f.read()
                    os.unlink(tmp.name)

                    scr_content = "\n".join(scr_lines)
                    scr_data = scr_content.encode("utf-8")

                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        st.download_button(
                            "⬇️ Download Converted DXF File",
                            data=dxf_data,
                            file_name="final_station_layout.dxf",
                            mime="application/dxf",
                            use_container_width=True,
                        )
                    with col_dl2:
                        st.download_button(
                            "⬇️ Download AutoCAD Script (.SCR)",
                            data=scr_data,
                            file_name="final_station_layout.scr",
                            mime="text/plain",
                            use_container_width=True,
                        )
            else:
                st.info(
                    "👉 Please complete the individual steps for **all**"
                    " defined stations in Step 2 before proceeding."
                )

    except Exception as e:
        st.error(f"Processing error: {e}")
else:
    st.info("👈 Please upload **Raw Data CSV** in the sidebar to start.")
