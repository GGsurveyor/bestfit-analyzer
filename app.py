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


st.set_page_config(
    page_title="Multi-Station BestFit Pro", page_icon="🏗️", layout="wide"
)

st.title("🏗️ 2D/3D BestFit Made By Ng Yit Fung")
st.markdown(
    "Upload Raw Data and optional Control/Design points. Complete the workflow"
    " seamlessly."
)

# Sidebar Uploads (Optional design & control points)
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

        # --- Step 0: Edit Raw Data ---
        st.markdown("---")
        st.subheader("✏️ Step 0: Raw Data Online Editor (序号从 1 开始)")

        if "df_raw_edited" not in st.session_state:
            temp_init = df_raw_initial.copy()
            temp_init.index = range(1, len(temp_init) + 1)
            st.session_state["df_raw_edited"] = temp_init

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

        # --- Step 1: Station Row Range Configuration ---
        st.markdown("---")
        st.subheader("🛠️ Step 1: Configure & Split Station Row Ranges")
        st.info(
            f"Total rows in Raw Data: **{total_rows}**. Adjust station names"
            " and start/end rows below."
        )

        col_cfg1, _ = st.columns([1, 2])
        with col_cfg1:
            num_stations = st.number_input(
                "Number of Stations", min_value=1, max_value=10, value=4, step=1
            )

        default_ranges_data = []
        chunk_size = total_rows // num_stations
        for i in range(num_stations):
            start = i * chunk_size + 1
            end = (
                (i + 1) * chunk_size if i < num_stations - 1 else total_rows
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
                    st.dataframe(
                        stn_raw_df.style.format({
                            "X": "{:.4f}",
                            "Y": "{:.4f}",
                            "Z": "{:.4f}",
                        })
                    )

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
                        # 如果没有上传 Control points，直接把原始数据存入（不做拟合）
                        if s_name not in st.session_state["station_fitted_dfs"]:
                            st.session_state["station_fitted_dfs"][s_name] = (
                                stn_indexed[["X", "Y", "Z"]]
                            )
                        st.info(
                            "ℹ️ Control Points not uploaded. Using raw data"
                            " directly for this station."
                        )

                    if s_name in st.session_state["station_fitted_dfs"]:
                        st.markdown(
                            f"**Result Preview ({s_name}_after):**"
                        )
                        st.dataframe(
                            st.session_state["station_fitted_dfs"][
                                s_name
                            ].style.format("{:.4f}")
                        )

                        if f"err_{s_name}" in st.session_state:
                            st.markdown(
                                "📊 **Fit Deviation Analysis (Control Points):**"
                            )
                            st.dataframe(
                                st.session_state[f"err_{s_name}"].style.format({
                                    "Delta E": "{:.4f}",
                                    "Delta N": "{:.4f}",
                                    "Delta El": "{:.4f}",
                                    "Total_Error": "{:.4f}",
                                })
                            )

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
                    if "df_final_result" not in st.session_state:
                        st.session_state["df_final_result"] = combined_df
                    st.info(
                        "ℹ️ Design Points not uploaded. Displaying merged"
                        " combined stations directly."
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
                            " [station-1+2+3+4_after BestFit with design"
                            " point.CSV]"
                        ),
                        data=final_csv,
                        file_name=(
                            "station-1+2+3+4_after_BestFit_with_design_point.CSV"
                        ),
                        mime="text/csv",
                    )
            else:
                st.info(
                    "👉 Please complete the individual steps for **all**"
                    " defined stations in Step 2 before proceeding to Step 3."
                )

    except Exception as e:
        st.error(f"Processing error: {e}")
else:
  st.info("👈 Please upload **Raw Data CSV** in the sidebar to start.")
