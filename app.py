import io
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


st.set_page_config(
    page_title="3D/2D BestFit & Interactive Station Splitter",
    page_icon="📐",
    layout="wide",
)

st.title(
    "📐 3D / 2D BestFit Alignment & Multi-Station Interactive Workstation Made By Ng Yit Fung"
)
st.markdown(
    "Upload your **Design / Control file** and **Raw Data file**. Edit raw"
    " data, configure station ranges interactively, exclude specific control"
    " points, and perform BestFit."
)

# Sidebar Configuration
st.sidebar.header("📂 Data Upload Center")
uploaded_design = st.sidebar.file_uploader(
    "Upload Design / Control CSV", type=["csv"], key="design"
)
uploaded_raw = st.sidebar.file_uploader(
    "Upload Raw Data CSV", type=["csv"], key="raw"
)

if uploaded_design is not None and uploaded_raw is not None:
  try:
    df_design = pd.read_csv(
        uploaded_design, header=None, names=["Point", "X", "Y", "Z"]
    )
    df_raw_initial = pd.read_csv(
        uploaded_raw, header=None, names=["Point", "X", "Y", "Z"]
    )

    # Clean point names
    df_design["Point"] = df_design["Point"].astype(str).str.strip()
    df_design.set_index("Point", inplace=True)

    # --- Feature: Edit Raw Data Table ---
    st.markdown("---")
    st.subheader("✏️ Edit Raw Data (原始数据在线编辑)")
    st.markdown(
        "You can inspect and edit the uploaded raw data table below. Any changes"
        " will be used directly for station splitting and calculations."
    )

    # Initialize raw data in session state if not present
    if "df_raw_edited" not in st.session_state:
      st.session_state["df_raw_edited"] = df_raw_initial.copy()

    edited_raw_df = st.data_editor(
        st.session_state["df_raw_edited"],
        num_rows="dynamic",
        use_container_width=True,
        key="raw_data_editor",
    )
    st.session_state["df_raw_edited"] = edited_raw_df

    # Use the edited raw data dataframe for subsequent steps
    df_raw = st.session_state["df_raw_edited"].copy()
    df_raw["Point"] = df_raw["Point"].astype(str).str.strip()
    total_rows = len(df_raw)

    # --- Step 1: Raw Data Station Row Range Configurator (with Edit Table) ---
    st.markdown("---")
    st.subheader("🛠️ Step 1: Configure & Edit Station Row Ranges")
    st.info(
        f"Total rows in Raw Data: **{total_rows}**. You can adjust the Station"
        " Name, Start Row, and End Row directly in the table below."
    )

    col_cfg1, _ = st.columns([1, 2])
    with col_cfg1:
      num_stations = st.number_input(
          "Number of Stations to Define",
          min_value=1,
          max_value=10,
          value=3,
          step=1,
      )

    default_ranges_data = []
    chunk_size = total_rows // num_stations
    for i in range(num_stations):
      start = i * chunk_size + 1
      end = (i + 1) * chunk_size if i < num_stations - 1 else total_rows
      default_ranges_data.append({
          "Station Name": f"Station-{i+1}",
          "Start Row": int(start),
          "End Row": int(end),
      })

    df_default_ranges = pd.DataFrame(default_ranges_data)

    st.markdown("#### ✏️ Editable Station Ranges Table:")
    edited_ranges_df = st.data_editor(
        df_default_ranges,
        num_rows="fixed",
        use_container_width=True,
        key="station_ranges_editor",
    )

    st.markdown("---")
    confirm_btn = st.button(
        "✅ 确定并生成独立分站文件 (Confirm & Generate Stations)",
        type="primary",
        use_container_width=True,
    )

    temp_configs = {}
    valid_config = True
    for index, row in edited_ranges_df.iterrows():
      s_name = str(row["Station Name"]).strip()
      try:
        r_start = int(row["Start Row"])
        r_end = int(row["End Row"])
      except ValueError:
        valid_config = False
        break

      if r_start > r_end or r_start < 1 or r_end > total_rows:
        valid_config = False

      temp_configs[s_name] = (r_start - 1, r_end)

    if confirm_btn or "stations_generated" in st.session_state:
      if not valid_config:
        st.error(
            "⚠️ Error: Please ensure row ranges are valid integers between 1"
            f" and {total_rows}, and Start Row <= End Row."
        )
      else:
        st.session_state["stations_generated"] = True
        st.session_state["temp_configs"] = temp_configs

        st.success(
            "🎉 Sub-stations successfully defined and generated! Independent CSV"
            " files are ready below."
        )

        st.subheader("📦 Generated Independent Sub-Station Files")
        file_cols = st.columns(len(temp_configs))
        generated_stations = {}

        for idx, (s_name, (s_start, s_end)) in enumerate(
            temp_configs.items()
        ):
          stn_df = df_raw.iloc[s_start:s_end].copy()
          generated_stations[s_name] = stn_df
          csv_bytes = stn_df.to_csv(
              index=False, header=False, float_format="%.4f"
          )

          with file_cols[idx % len(file_cols)]:
            st.markdown(f"**{s_name}** (Rows {s_start+1} to {s_end})")
            st.download_button(
                label=f"📥 Download {s_name}.CSV",
                data=csv_bytes,
                file_name=f"{s_name}.CSV",
                mime="text/csv",
                key=f"dl_{s_name}",
            )

        st.markdown("---")
        # --- Step 2: Choose Station, Mode & Exclude Points ---
        st.subheader("🎯 Step 2: Select Station, Mode & Exclude Control Points")

        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
          selected_station_to_fit = st.selectbox(
              "Choose Target Station to BestFit to Design",
              options=list(generated_stations.keys()),
          )
        with col_sel2:
          alignment_mode = st.selectbox(
              "Select BestFit Mode",
              options=[
                  "3D BestFit (Full 6 DOF)",
                  "2D BestFit (XY Plane Constraint)",
              ],
          )

        active_df = generated_stations[selected_station_to_fit].copy()
        active_df["Point"] = active_df["Point"].astype(str).str.strip()
        active_indexed = active_df.set_index("Point")

        all_station_common_points = df_design.index.intersection(
            active_indexed.index
        )

        excluded_points = st.multiselect(
            "🚫 Exclude Control Points from Calculation (不在计算/拟合范围内)",
            options=all_station_common_points.tolist(),
            default=[],
            help=(
                "Selected points will be ignored during BestFit matrix"
                " calculation and deviation analysis."
            ),
        )

        station_common_points = all_station_common_points.difference(
            excluded_points
        )

        if len(station_common_points) < 3:
          st.error(
              f"Error: After exclusion, station [{selected_station_to_fit}] has"
              f" only {len(station_common_points)} valid common points. At"
              " least 3 common points are required for BestFit!"
          )
        else:
          design_pts = df_design.loc[
              station_common_points, ["X", "Y", "Z"]
          ].values
          raw_pts = active_indexed.loc[
              station_common_points, ["X", "Y", "Z"]
          ].values

          if "3D" in alignment_mode:
            R, T = BestFitEngine.best_fit_3d(raw_pts, design_pts)
          else:
            R, T = BestFitEngine.best_fit_2d(raw_pts, design_pts)

          all_station_pts = active_indexed[["X", "Y", "Z"]].values
          transformed_active_pts = np.dot(all_station_pts, R.T) + T
          df_active_after = pd.DataFrame(
              transformed_active_pts,
              index=active_indexed.index,
              columns=["X", "Y", "Z"],
          )

          transformed_all_common = (
              np.dot(
                  active_indexed.loc[
                      all_station_common_points, ["X", "Y", "Z"]
                  ].values,
                  R.T,
              )
              + T
          )
          df_common_after = pd.DataFrame(
              transformed_all_common,
              index=all_station_common_points,
              columns=["After_X", "After_Y", "After_Z"],
          )

          error_df = df_design.loc[all_station_common_points].copy()
          error_df.rename(
              columns={"X": "Design_X", "Y": "Design_Y", "Z": "Design_Z"},
              inplace=True,
          )
          error_df["After_X"] = df_common_after["After_X"]
          error_df["After_Y"] = df_common_after["After_Y"]
          error_df["After_Z"] = df_common_after["After_Z"]

          error_df["Delta E"] = error_df["After_X"] - error_df["Design_X"]
          error_df["Delta N"] = error_df["After_Y"] - error_df["Design_Y"]
          error_df["Delta El"] = error_df["After_Z"] - error_df["Design_Z"]
          error_df["Total_Error"] = np.sqrt(
              error_df["Delta E"] ** 2
              + error_df["Delta N"] ** 2
              + error_df["Delta El"] ** 2
          )

          error_df["Status"] = [
              (
                  "Excluded (不参与计算)"
                  if pt in excluded_points
                  else "Active (参与计算)"
              )
              for pt in error_df.index
          ]

          # --- Main UI Results Display ---
          st.markdown("---")
          st.subheader(
              f"📊 BestFit Results for [{selected_station_to_fit}] using"
              f" {alignment_mode}"
          )

          col_res1, col_res2 = st.columns(2)
          with col_res1:
            st.text("Rotation Matrix R:")
            st.write(
                np.array2string(
                    R, formatter={"float_kind": lambda x: "%.4f" % x}
                )
            )
          with col_res2:
            st.text("Translation Vector T:")
            st.write(
                np.array2string(
                    T, formatter={"float_kind": lambda x: "%.4f" % x}
                )
            )

          # 1. Fit Deviations Analysis
          st.markdown("---")
          st.subheader(
              "🔍 2D/3D Fit Deviations (Design vs Aligned Station Data)"
          )
          st.markdown(
              f"Total common control points in **{selected_station_to_fit}**:"
              f" **{len(all_station_common_points)}** (Active for fit:"
              f" **{len(station_common_points)}**, Excluded: "
              f"**{len(excluded_points)}**)"
          )
          st.dataframe(
              error_df[[
                  "Status",
                  "Design_X",
                  "Design_Y",
                  "Design_Z",
                  "Delta E",
                  "Delta N",
                  "Delta El",
                  "Total_Error",
              ]].style.format({
                  "Design_X": "{:.4f}",
                  "Design_Y": "{:.4f}",
                  "Design_Z": "{:.4f}",
                  "Delta E": "{:.4f}",
                  "Delta N": "{:.4f}",
                  "Delta El": "{:.4f}",
                  "Total_Error": "{:.4f}",
              })
          )

          # 2. Aligned Coordinates Preview & Download Button at the bottom
          st.markdown("---")
          st.subheader(
              f"📋 Aligned Coordinates Preview ({selected_station_to_fit}_after)"
          )
          st.dataframe(df_active_after.style.format("{:.4f}"))

          active_csv_data = df_active_after.reset_index().to_csv(
              index=False, header=False, float_format="%.4f"
          )
          safe_name = selected_station_to_fit.replace(" ", "_").lower()
          st.download_button(
              label=(
                  f"📥 Download Aligned [{selected_station_to_fit}] Result"
                  " (.CSV)"
              ),
              data=active_csv_data,
              file_name=f"{safe_name}_after.CSV",
              mime="text/csv",
          )

  except Exception as e:
    st.error(f"An error occurred during processing: {e}")
else:
  st.info(
      "👈 Please upload both the **Design / Control CSV** and **Raw Data CSV**"
      " files in the sidebar to begin."
  )
