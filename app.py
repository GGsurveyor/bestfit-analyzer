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
    page_title="3D/2D BestFit & Multi-Station Analysis",
    page_icon="📐",
    layout="wide",
)

st.title("📐 3D / 2D BestFit Alignment & Multi-Station Analysis System")
st.markdown(
    "Upload your **Design/Control file** and **Raw Data file** (containing"
    " multi-station measurements)."
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

if uploaded_design is not None and uploaded_raw is not None:
  try:
    df_design = pd.read_csv(
        uploaded_design, header=None, names=["Point", "X", "Y", "Z"]
    )
    df_raw = pd.read_csv(
        uploaded_raw, header=None, names=["Point", "X", "Y", "Z"]
    )

    # Clean point names to avoid int vs str type mismatches
    df_design["Point"] = df_design["Point"].astype(str).str.strip()
    df_raw["Point"] = df_raw["Point"].astype(str).str.strip()

    df_design.set_index("Point", inplace=True)

    # --- Automatic Sub-Station (STN) Detection from Raw Data ---
    # Detect rows starting with 'BS' as station/back-sight boundaries
    bs_indices = df_raw[
        df_raw["Point"].str.contains("^BS", case=False, na=False)
    ].index.tolist()

    stations = {}
    if bs_indices:
      for i in range(len(bs_indices)):
        start_idx = bs_indices[i]
        end_idx = (
            bs_indices[i + 1] if i + 1 < len(bs_indices) else len(df_raw)
        )
        bs_name = df_raw.loc[start_idx, "Point"]
        stn_df = df_raw.iloc[start_idx:end_idx].copy()
        stations[f"Station {i+1} ({bs_name})"] = stn_df
    else:
      # If no BS found, treat the whole raw data as one station
      stations["All Raw Data"] = df_raw

    # Global BestFit calculation using all common points between design and raw
    # For finding common points across all raw data:
    df_raw_indexed = df_raw.set_index("Point")
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

      # Transform entire Raw Data
      all_raw_pts = df_raw_indexed[["X", "Y", "Z"]].values
      transformed_raw_pts = np.dot(all_raw_pts, R.T) + T
      df_after_global = pd.DataFrame(
          transformed_raw_pts,
          index=df_raw_indexed.index,
          columns=["X", "Y", "Z"],
      )

      # Error Analysis for Common Points
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

      # --- UI Layout ---
      st.markdown("---")
      view_tab = st.radio(
          "📊 Select View Mode",
          options=[
              "Global Transformation & Error Analysis",
              "Sub-Station Inspection & Transformation",
          ],
          horizontal=True,
      )

      if view_tab == "Global Transformation & Error Analysis":
        st.subheader(f"📊 Spatial Transformation Results ({alignment_mode})")
        col1, col2 = st.columns(2)
        with col1:
          st.text("Rotation Matrix R:")
          st.write(
              np.array2string(
                  R, formatter={"float_kind": lambda x: "%.4f" % x}
              )
          )
        with col2:
          st.text("Translation Vector T:")
          st.write(
              np.array2string(
                  T, formatter={"float_kind": lambda x: "%.4f" % x}
              )
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

        st.markdown("---")
        st.subheader("📋 Full Calculated After Bestfit Coordinates Preview")
        st.dataframe(df_after_global.style.format("{:.4f}"))

        csv_data = df_after_global.reset_index().to_csv(
            index=False, header=False, float_format="%.4f"
        )
        st.download_button(
            label="📥 Download Global After Bestfit Result File (.CSV)",
            data=csv_data,
            file_name="after_Bestfit_global.CSV",
            mime="text/csv",
        )

      else:
        st.subheader(
            "🏢 Sub-Station Selection & Transformation (Extracted from Raw Data)"
        )
        selected_stn_name = st.selectbox(
            "Select Sub-Station", options=list(stations.keys())
        )
        stn_df = stations[selected_stn_name]

        st.markdown(
            f"**Raw Data Preview for {selected_stn_name} (Rows:"
            f" {len(stn_df)}):**"
        )
        st.dataframe(
            stn_df.set_index("Point").style.format(
                {"X": "{:.4f}", "Y": "{:.4f}", "Z": "{:.4f}"}
            )
        )

        # Apply transformation to this sub-station
        stn_pts = stn_df[["X", "Y", "Z"]].values
        transformed_stn_pts = np.dot(stn_pts, R.T) + T
        df_stn_after = pd.DataFrame(
            transformed_stn_pts, index=stn_df["Point"], columns=["X", "Y", "Z"]
        )

        st.markdown(f"**Aligned Coordinates Result for {selected_stn_name}:**")
        st.dataframe(df_stn_after.style.format("{:.4f}"))

        stn_csv_data = df_stn_after.reset_index().to_csv(
            index=False, header=False, float_format="%.4f"
        )
        safe_stn_name = selected_stn_name.replace(" ", "_").replace("(", "").replace(")", "")
        st.download_button(
            label=f"📥 Download Aligned {selected_stn_name} Result File (.CSV)",
            data=stn_csv_data,
            file_name=f"{safe_stn_name}_after.CSV",
            mime="text/csv",
        )

  except Exception as e:
    st.error(f"An error occurred during processing: {e}")
else:
  st.info(
      "👈 Please upload both the **Design / Control CSV** and **Raw Data CSV**"
      " files in the left sidebar to begin."
  )
