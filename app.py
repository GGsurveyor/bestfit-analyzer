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
    page_title="3D/2D BestFit & Multi-Station Selector System",
    page_icon="📐",
    layout="wide",
)

st.title("📐 3D / 2D BestFit Alignment & Sub-Station Selection System")
st.markdown(
    "Upload your **Design / Control file**, **Raw Data file**, and optional"
    " **Sub-Station files (STN1, STN2, STN3)**. Choose your target sub-station"
    " to analyze."
)

# Sidebar Configuration
st.sidebar.header("⚙️ Alignment Settings")
alignment_mode = st.sidebar.selectbox(
    "Select BestFit Mode",
    options=["3D BestFit (Full 6 DOF)", "2D BestFit (XY Plane Constraint)"],
)

st.sidebar.markdown("---")
st.sidebar.header("📂 Data Upload Center")
uploaded_design = st.sidebar.file_uploader(
    "Upload Design / Control CSV", type=["csv"], key="design"
)
uploaded_raw = st.sidebar.file_uploader(
    "Upload Raw Data CSV", type=["csv"], key="raw"
)

st.sidebar.markdown("### 🏢 Sub-Station Files (Optional)")
uploaded_stn1 = st.sidebar.file_uploader(
    "Upload STN 1 CSV", type=["csv"], key="stn1"
)
uploaded_stn2 = st.sidebar.file_uploader(
    "Upload STN 2 CSV", type=["csv"], key="stn2"
)
uploaded_stn3 = st.sidebar.file_uploader(
    "Upload STN 3 CSV", type=["csv"], key="stn3"
)

if uploaded_design is not None and uploaded_raw is not None:
  try:
    df_design = pd.read_csv(
        uploaded_design, header=None, names=["Point", "X", "Y", "Z"]
    )
    df_raw = pd.read_csv(
        uploaded_raw, header=None, names=["Point", "X", "Y", "Z"]
    )

    # Clean point names
    df_design["Point"] = df_design["Point"].astype(str).str.strip()
    df_raw["Point"] = df_raw["Point"].astype(str).str.strip()
    df_design.set_index("Point", inplace=True)
    df_raw_indexed = df_raw.set_index("Point")

    # --- Build Sub-Station Pool ---
    station_pool = {"Global Raw Data": df_raw}

    if uploaded_stn1 is not None:
      df_s1 = pd.read_csv(uploaded_stn1, header=None, names=["Point", "X", "Y", "Z"])
      df_s1["Point"] = df_s1["Point"].astype(str).str.strip()
      station_pool["Sub-Station 1 (STN1)"] = df_s1

    if uploaded_stn2 is not None:
      df_s2 = pd.read_csv(uploaded_stn2, header=None, names=["Point", "X", "Y", "Z"])
      df_s2["Point"] = df_s2["Point"].astype(str).str.strip()
      station_pool["Sub-Station 2 (STN2)"] = df_s2

    if uploaded_stn3 is not None:
      df_s3 = pd.read_csv(uploaded_stn3, header=None, names=["Point", "X", "Y", "Z"])
      df_s3["Point"] = df_s3["Point"].astype(str).str.strip()
      station_pool["Sub-Station 3 (STN3)"] = df_s3

    # If no individual STN files uploaded, automatically partition raw data by BS
    if len(station_pool) == 1:
      bs_indices = df_raw[
          df_raw["Point"].str.contains("^BS", case=False, na=False)
      ].index.tolist()
      if bs_indices:
        for i in range(len(bs_indices)):
          start_idx = bs_indices[i]
          end_idx = (
              bs_indices[i + 1] if i + 1 < len(bs_indices) else len(df_raw)
          )
          bs_name = df_raw.loc[start_idx, "Point"]
          stn_df = df_raw.iloc[start_idx:end_idx].copy()
          station_pool[f"Auto-Station {i+1} ({bs_name})"] = stn_df

    # --- Sidebar Sub-Station Selector ---
    st.sidebar.markdown("---")
    st.sidebar.header("🎯 Sub-Station Selector")
    selected_station_name = st.sidebar.selectbox(
        "Choose Target Sub-Station", options=list(station_pool.keys())
    )

    # Get common points for BestFit calculation
    common_points = df_design.index.intersection(df_raw_indexed.index)

    if len(common_points) < 3:
      st.error(
          f"Error: Found only {len(common_points)} common points between Design"
          " and Raw data. At least 3 common points are required!"
      )
    else:
      design_pts = df_design.loc[common_points, ["X", "Y", "Z"]].values
      raw_pts = df_raw_indexed.loc[common_points, ["X", "Y", "Z"]].values

      if "3D" in alignment_mode:
        R, T = BestFitEngine.best_fit_3d(raw_pts, design_pts)
      else:
        R, T = BestFitEngine.best_fit_2d(raw_pts, design_pts)

      # Target active sub-station transformation
      active_df = station_pool[selected_station_name]
      active_indexed = active_df.set_index("Point")
      active_pts = active_indexed[["X", "Y", "Z"]].values
      transformed_active_pts = np.dot(active_pts, R.T) + T
      df_active_after = pd.DataFrame(
          transformed_active_pts,
          index=active_indexed.index,
          columns=["X", "Y", "Z"],
      )

      # Global Error Analysis for Common Points
      all_raw_pts = df_raw_indexed[["X", "Y", "Z"]].values
      transformed_all_pts = np.dot(all_raw_pts, R.T) + T
      df_after_global = pd.DataFrame(
          transformed_all_pts,
          index=df_raw_indexed.index,
          columns=["X", "Y", "Z"],
      )

      error_df = df_design.loc[common_points].copy()
      error_df.rename(
          columns={"X": "Design_X", "Y": "Design_Y", "Z": "Design_Z"},
          inplace=True,
      )
      error_df["After_X"] = df_after_global.loc[common_points, "X"]
      error_df["After_Y"] = df_after_global.loc[common_points, "Y"]
      error_df["After_Z"] = df_after_global.loc[common_points, "Z"]

      error_df["Err_X"] = error_df["After_X"] - error_df["Design_X"]
      error_df["Err_Y"] = error_df["After_Y"] - error_df["Design_Y"]
      error_df["Err_Z"] = error_df["After_Z"] - error_df["Design_Z"]
      error_df["Total_Error"] = np.sqrt(
          error_df["Err_X"] ** 2
          + error_df["Err_Y"] ** 2
          + error_df["Err_Z"] ** 2
      )

      # --- Main UI Display ---
      st.subheader(
          f"📊 Active Sub-Station: {selected_station_name} ({alignment_mode})"
      )

      col1, col2 = st.columns(2)
      with col1:
        st.text("Rotation Matrix R:")
        st.write(
            np.array2string(R, formatter={"float_kind": lambda x: "%.4f" % x})
        )
      with col2:
        st.text("Translation Vector T:")
        st.write(
            np.array2string(T, formatter={"float_kind": lambda x: "%.4f" % x})
        )

      st.markdown("---")
      st.subheader(
          f"📋 Aligned Coordinates Preview ({selected_station_name})"
      )
      st.dataframe(df_active_after.style.format("{:.4f}"))

      active_csv_data = df_active_after.reset_index().to_csv(
          index=False, header=False, float_format="%.4f"
      )
      safe_name = (
          selected_station_name.replace(" ", "_")
          .replace("(", "")
          .replace(")", "")
      )
      st.download_button(
          label=f"📥 Download Aligned {selected_station_name} Result (.CSV)",
          data=active_csv_data,
          file_name=f"{safe_name}_after.CSV",
          mime="text/csv",
      )

      st.markdown("---")
      st.subheader(
          "🔍 Common Points Error Analysis (Design vs Aligned Result)"
      )
      st.markdown(
          f"Found **{len(common_points)}** common points: "
          f"{', '.join(common_points.tolist())}"
      )
      st.dataframe(
          error_df[[
              "Design_X",
              "Design_Y",
              "Design_Z",
              "Err_X",
              "Err_Y",
              "Err_Z",
              "Total_Error",
          ]].style.format("{:.4f}")
      )

  except Exception as e:
    st.error(f"An error occurred during processing: {e}")
else:
  st.info(
      "👈 Please upload both the **Design / Control CSV** and **Raw Data CSV**"
      " files in the left sidebar to begin."
  )
