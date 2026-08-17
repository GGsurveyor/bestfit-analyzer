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
    page_title="3D/2D BestFit Alignment Tool", page_icon="📐", layout="wide"
)

st.title("📐 3D / 2D BestFit Alignment & Error Analysis System")
st.markdown(
    "Upload your **Design CSV file** and **Before Bestfit CSV file** to compute"
    " alignment and view detailed point errors."
)

st.sidebar.header("⚙️ Alignment Settings")
alignment_mode = st.sidebar.selectbox(
    "Select BestFit Mode",
    options=["3D BestFit (Full 6 DOF)", "2D BestFit (XY Plane Constraint)"],
)

st.sidebar.markdown("---")
st.sidebar.header("📂 Upload Measurement Data")
uploaded_design = st.sidebar.file_uploader(
    "Upload Design CSV File", type=["csv"]
)
uploaded_before = st.sidebar.file_uploader(
    "Upload Before Bestfit CSV File", type=["csv"]
)

if uploaded_design is not None and uploaded_before is not None:
  try:
    df_design = pd.read_csv(
        uploaded_design, header=None, names=["Point", "X", "Y", "Z"]
    )
    df_before = pd.read_csv(
        uploaded_before, header=None, names=["Point", "X", "Y", "Z"]
    )

    # Convert Point column to string and strip spaces to avoid int vs str type mismatch
    df_design["Point"] = df_design["Point"].astype(str).str.strip()
    df_before["Point"] = df_before["Point"].astype(str).str.strip()

    df_design.set_index("Point", inplace=True)
    df_before.set_index("Point", inplace=True)

    common_points = df_design.index.intersection(df_before.index)

    if len(common_points) < 3:
      st.error(
          f"Error: Found only {len(common_points)} common points. At least 3"
          " common points are required to perform BestFit calculation!"
      )
    else:
      design_pts = df_design.loc[common_points, ["X", "Y", "Z"]].values
      before_pts = df_before.loc[common_points, ["X", "Y", "Z"]].values

      if "3D" in alignment_mode:
        R, T = BestFitEngine.best_fit_3d(before_pts, design_pts)
      else:
        R, T = BestFitEngine.best_fit_2d(before_pts, design_pts)

      # Apply transformation to all data
      all_before_pts = df_before[["X", "Y", "Z"]].values
      transformed_pts = np.dot(all_before_pts, R.T) + T
      df_after = pd.DataFrame(
          transformed_pts, index=df_before.index, columns=["X", "Y", "Z"]
      )

      # Calculate error for common points (Design vs Calculated After)
      error_df = df_design.loc[common_points].copy()
      error_df.rename(
          columns={"X": "Design_X", "Y": "Design_Y", "Z": "Design_Z"},
          inplace=True,
      )
      error_df["After_X"] = df_after.loc[common_points, "X"]
      error_df["After_Y"] = df_after.loc[common_points, "Y"]
      error_df["After_Z"] = df_after.loc[common_points, "Z"]

      error_df["Err_X"] = error_df["After_X"] - error_df["Design_X"]
      error_df["Err_Y"] = error_df["After_Y"] - error_df["Design_Y"]
      error_df["Err_Z"] = error_df["After_Z"] - error_df["Design_Z"]
      error_df["Total_Error"] = np.sqrt(
          error_df["Err_X"] ** 2
          + error_df["Err_Y"] ** 2
          + error_df["Err_Z"] ** 2
      )

      # UI Display
      st.markdown("---")
      st.subheader(f"📊 Spatial Transformation Results ({alignment_mode})")
      col1, col2 = st.columns(2)
      with col1:
        st.text("Rotation Matrix R:")
        # Format matrix values to 4 decimal places for display
        st.write(np.array2string(R, formatter={"float_kind": lambda x: "%.4f" % x}))
      with col2:
        st.text("Translation Vector T:")
        st.write(np.array2string(T, formatter={"float_kind": lambda x: "%.4f" % x}))

      st.markdown("---")
      st.subheader(
          "🔍 Common Points Error Analysis (Design vs Aligned Result)"
      )
      st.markdown(
          f"Found **{len(common_points)}** common points: "
          f"{', '.join(common_points.tolist())}"
      )
      # Format table values to 4 decimal places
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
      # Format table values to 4 decimal places
      st.dataframe(df_after.style.format("{:.4f}"))

      csv_data = df_after.reset_index().to_csv(index=False, header=False, float_format="%.4f")
      st.download_button(
          label="📥 Download After Bestfit Result File (.CSV)",
          data=csv_data,
          file_name="LP22B_AW_calculated_after.CSV",
          mime="text/csv",
      )

  except Exception as e:
    st.error(f"An error occurred during processing: {e}")
else:
  st.info(
      "👈 Please select your mode and upload both the **Design** and"
      " **Before** CSV files in the left sidebar to begin."
  )
