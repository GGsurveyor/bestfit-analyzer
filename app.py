import io
import numpy as np
import pandas as pd
import streamlit as st


class BestFitEngine:

  @staticmethod
  def best_fit_3d(measured, design):
    """3D BestFit: Full 6 DOF (X, Y, Z translation + 3D rotation)"""
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
    """2D BestFit: Restricted to XY plane (X, Y translation + 2D rotation around Z)"""
    # Extract only X and Y coordinates for 2D alignment computation
    m_xy = measured[:, :2]
    d_xy = design[:, :2]

    centroid_m = np.mean(m_xy, axis=0)
    centroid_d = np.mean(d_xy, axis=0)

    m_centered = m_xy - centroid_m
    d_centered = d_xy - centroid_d

    H = np.dot(m_centered.T, d_centered)
    U, S, Vt = np.linalg.svd(H)

    # 2D Rotation matrix
    R_2d = np.dot(Vt.T, U.T)
    if np.linalg.det(R_2d) < 0:
      Vt[1, :] *= -1
      R_2d = np.dot(Vt.T, U.T)

    T_2d = centroid_d - np.dot(R_2d, centroid_m)

    # Construct full 3D transformation matrices where Z has no rotation/translation change
    R = np.eye(3)
    R[:2, :2] = R_2d

    T = np.zeros(3)
    T[:2] = T_2d
    # Z translation can be kept 0 or averaged, usually 0 in pure 2D profile alignment
    T[2] = np.mean(design[:, 2]) - np.mean(measured[:, 2])

    return R, T


# Streamlit Web Page Layout
st.set_page_config(
    page_title="3D/2D BestFit Alignment Tool", page_icon="📐", layout="wide"
)

st.title("📐 3D / 2D BestFit Alignment & Analysis System")
st.markdown(
    "Upload your **Design CSV file** and **Before Bestfit CSV file**, select"
    " your alignment mode, and compute the results."
)

# Sidebar Options & File Uploader
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
    # Read Data
    df_design = pd.read_csv(
        uploaded_design, header=None, names=["Point", "X", "Y", "Z"]
    )
    df_before = pd.read_csv(
        uploaded_before, header=None, names=["Point", "X", "Y", "Z"]
    )

    df_design.set_index("Point", inplace=True)
    df_before.set_index("Point", inplace=True)

    # Extract common points for computation
    common_points = df_design.index.intersection(df_before.index)

    if len(common_points) < 3:
      st.error(
          "Error: The number of common points is less than 3, unable to"
          " perform BestFit calculation!"
      )
    else:
      design_pts = df_design.loc[common_points, ["X", "Y", "Z"]].values
      before_pts = df_before.loc[common_points, ["X", "Y", "Z"]].values

      # Run Algorithm based on user choice
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

      # Display Transformation Results
      st.markdown("---")
      st.subheader(f"📊 Spatial Transformation Results ({alignment_mode})")
      col1, col2 = st.columns(2)
      with col1:
        st.text("Rotation Matrix R:")
        st.write(R)
      with col2:
        st.text("Translation Vector T:")
        st.write(T)

      # Preview Aligned Coordinates
      st.markdown("---")
      st.subheader("📋 Calculated After Bestfit Coordinates Preview")
      st.dataframe(df_after)

      # Download Button
      csv_data = df_after.reset_index().to_csv(index=False, header=False)
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
