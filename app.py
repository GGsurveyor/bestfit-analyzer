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
    page_title="Multi-Station Pipeline BestFit", page_icon="🏗️", layout="wide"
)

st.title("🏗️ Professional Pipeline: Multi-Station 2D -> Combined 3D BestFit")
st.markdown(
    "Upload Design points, Control points, and Raw Data. Complete the entire"
    " workflow in one single interface without external Excel steps."
)

# Sidebar File Uploads (Fixed unique keys)
st.sidebar.header("📂 Data Upload Center")
uploaded_design = st.sidebar.file_uploader(
    "Upload Design / Control CSV", type=["csv"], key="design_file"
)
uploaded_ctrl = st.sidebar.file_uploader(
    "Upload Control Point CSV (for 2D)", type=["csv"], key="ctrl_file"
)
uploaded_raw = st.sidebar.file_uploader(
    "Upload Raw Data CSV", type=["csv"], key="raw_file"
)

if (
    uploaded_design is not None
    and uploaded_ctrl is not None
    and uploaded_raw is not None
):
  try:
    df_design = pd.read_csv(
        uploaded_design, header=None, names=["Point", "X", "Y", "Z"]
    )
    df_ctrl = pd.read_csv(
        uploaded_ctrl, header=None, names=["Point", "X", "Y", "Z"]
    )
    df_raw_initial = pd.read_csv(
        uploaded_raw, header=None, names=["Point", "X", "Y", "Z"]
    )

    df_design["Point"] = df_design["Point"].astype(str).str.strip()
    df_design.set_index("Point", inplace=True)

    df_ctrl["Point"] = df_ctrl["Point"].astype(str).str.strip()
    df_ctrl.set_index("Point", inplace=True)

    # --- Step 0: Edit Raw Data ---
    st.markdown("---")
    st.subheader("✏️ Step 0: Raw Data Online Editor")

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
    st.subheader("🛠️ Step 1: Configure & Split Stations")

    col_cfg1, _ = st.columns([1, 2])
    with col_cfg1:
      num_stations = st.number_input(
          "Number of Stations", min_value=1, max_value=10, value=4, step=1
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

    edited_ranges_df = st.data_editor(
        pd.DataFrame(default_ranges_data),
        num_rows="fixed",
        use_container_width=True,
        hide_index=True,
        key="station_ranges_editor",
    )

    # Parse ranges
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
      st.error("⚠️ Please check station row ranges (must be valid integers).")
    else:
      # --- Step 2: Individual Station 2D BestFit with Control Points ---
      st.markdown("---")
      st.subheader("🎯 Step 2: Individual 2D BestFit for Each Station")
      st.markdown(
          "Perform 2D BestFit on each station independently using Control"
          " Points."
      )

      if "station_fitted_dfs" not in st.session_state:
        st.session_state["station_fitted_dfs"] = {}

      tabs = st.tabs(list(station_configs.keys()))

      for idx, (s_name, (s_start, s_end)) in enumerate(
          station_configs.items()
      ):
        with tabs[idx]:
          st.markdown(f"#### Managing **{s_name}** (Rows {s_start+1} to {s_end})")
          stn_raw_df = df_raw.iloc[s_start:s_end].copy()

          st.dataframe(stn_raw_df.style.format({"X": "{:.4f}", "Y": "{:.4f}", "Z": "{:.4f}"}))

          stn_indexed = stn_raw_df.set_index("Point")
          common_ctrl = df_ctrl.index.intersection(stn_indexed.index)

          st.info(f"Found {len(common_ctrl)} matching control points for {s_name}.")

          if len(common_ctrl) >= 3:
            if st.button(
                f"Execute 2D BestFit for {s_name}", key=f"fit_2d_{s_name}"
            ):
              m_pts = stn_indexed.loc[common_ctrl, ["X", "Y", "Z"]].values
              d_pts = df_ctrl.loc[common_ctrl, ["X", "Y", "Z"]].values

              R_2d, T_2d = BestFitEngine.best_fit_2d(m_pts, d_pts)
              transformed_pts = (
                  np.dot(stn_indexed[["X", "Y", "Z"]].values, R_2d.T) + T_2d
              )

              df_fitted = pd.DataFrame(
                  transformed_pts,
                  index=stn_indexed.index,
                  columns=["X", "Y", "Z"],
              )
              st.session_state["station_fitted_dfs"][s_name] = df_fitted
              st.success(
                  f"✅ {s_name} 2D BestFit completed and stored in memory!"
              )

          if s_name in st.session_state["station_fitted_dfs"]:
            st.markdown(f"**Result Preview ({s_name}_after):**")
            st.dataframe(
                st.session_state["station_fitted_dfs"][s_name].style.format(
                    "{:.4f}"
                )
            )

      # --- Step 3: Combine All Stations & Final 3D BestFit with Design Points ---
      st.markdown("---")
      st.subheader("🚀 Step 3: Combine All Stations & Final 3D BestFit")

      if len(st.session_state["station_fitted_dfs"]) == len(station_configs):
        if st.button(
            "✨ Merge All Stations & Perform Final 3D BestFit with Design Points",
            type="primary",
            use_container_width=True,
        ):
          # Merge all fitted stations together
          combined_df = pd.concat(
              list(st.session_state["station_fitted_dfs"].values())
          )

          common_design = df_design.index.intersection(combined_df.index)

          if len(common_design) < 3:
            st.error(
                "Error: Not enough common points between combined stations and"
                " design points for 3D fit."
            )
          else:
            m_final = combined_df.loc[common_design, ["X", "Y", "Z"]].values
            d_final = df_design.loc[common_design, ["X", "Y", "Z"]].values

            R_3d, T_3d = BestFitEngine.best_fit_3d(m_final, d_final)

            # Transform absolute all points
            final_coords = (
                np.dot(combined_df[["X", "Y", "Z"]].values, R_3d.T) + T_3d
            )
            df_final_result = pd.DataFrame(
                final_coords, index=combined_df.index, columns=["X", "Y", "Z"]
            )

            st.session_state["df_final_result"] = df_final_result
            st.success(
                "🎉 Final Combined 3D BestFit completed successfully inside"
                " the app!"
            )

        if "df_final_result" in st.session_state:
          st.markdown("#### 📋 Final Result: Combined Stations After 3D BestFit")
          st.dataframe(
              st.session_state["df_final_result"].style.format("{:.4f}")
          )

          final_csv = st.session_state["df_final_result"].reset_index().to_csv(
              index=False, header=False, float_format="%.4f"
          )
          st.download_button(
              label=(
                  "📥 Download Final Result"
                  " [station-1+2+3+4_after 3D BestFit with design point.CSV]"
              ),
              data=final_csv,
              file_name=(
                  "station-1+2+3+4_after 3D BestFit with design point.CSV"
              ),
              mime="text/csv",
          )
      else:
        st.info(
            "👉 Please complete the 2D BestFit for **all** defined stations"
            " above before proceeding to Step 3."
        )

  except Exception as e:
    st.error(f"Processing error: {e}")
else:
  st.info(
      "👈 Please upload **Design points**, **Control points**, and **Raw Data**"
      " CSV files in the sidebar to start."
  )
