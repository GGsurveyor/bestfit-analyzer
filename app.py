import io
import numpy as np
import pandas as pd
import streamlit as st


# Define BestFit Core Calculation Engine
class BestFitEngine:

  @staticmethod
  def best_fit(measured, design):
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
  def apply_transform(measured, R, T):
    return np.dot(measured, R.T) + T


# Streamlit Web Page Layout
st.set_page_config(
    page_title="3D BestFit Alignment Tool", page_icon="📐", layout="wide"
)

st.title("📐 3D BestFit Alignment & Analysis System - Made by Ng Yit Fung")
st.markdown(
    "Upload your **Design CSV file** and **Before Bestfit CSV file**."
    " The system will automatically calculate the optimal rotation matrix,"
    " translation vector, and aligned coordinates."
)

# Sidebar File Uploader
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
          " perform 3D BestFit calculation!"
      )
    else:
      design_pts = df_design.loc[common_points, ["X", "Y", "Z"]].values
      before_pts = df_before.loc[common_points, ["X", "Y", "Z"]].values

      # Run Algorithm
      R, T = BestFitEngine.best_fit(before_pts, design_pts)

      # Apply to all data
      all_before_pts = df_before[["X", "Y", "Z"]].values
      transformed_pts = BestFitEngine.apply_transform(all_before_pts, R, T)

      df_after = pd.DataFrame(
          transformed_pts, index=df_before.index, columns=["X", "Y", "Z"]
      )

      # Display Transformation Matrices
      st.markdown("---")
      st.subheader("📊 Spatial Transformation Matrix Results")
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
      "👈 Please upload both the **Design** and **Before** CSV files in the"
      " left sidebar to begin."
  )
