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
    page_title="3D/2D BestFit Multi-Station Alignment System",
    page_icon="📐",
    layout="wide",
)

st.title("📐 3D / 2D BestFit Multi-Station Alignment & Error Analysis System")
st.markdown(
    "Upload your **Design/Control file**, **Raw/Before file**, and optional"
    " **Sub-Station (STN) files** to compute alignment and analyze point"
    " errors."
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
    "Upload Raw / Before Bestfit CSV", type=["csv"], key="raw"
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

    # Clean point names to avoid int vs str type mismatches
    df_design["Point"] = df_design["Point"].astype(str).str.strip()
    df_raw["Point"] = df_raw["Point"].astype(str).str.strip()

    df_design.set_index("Point", inplace=True)
    df_raw.set_index("Point", inplace=True)

    common_points = df_design.index.intersection(df_raw.index)

    if len(common_points) < 3:
      st.error(
          f"Error: Found only {len(common_points)} common points between Design"
          " and Raw data. At least 3 common points are required!"
      )
    else:
      design_pts = df_design.loc[common_points, ["X", "Y", "Z"]].values
      raw_pts = df_raw.loc[common_points, ["X", "Y", "Z"]].values

      if "3D" in alignment_mode:
        R, T = BestFitEngine.best_fit_3d(raw_pts, design_pts)
      else:
        R, T = BestFitEngine.best_fit_2d(raw_pts, design_pts)

      # Transform Raw Data
      all_raw_pts = df_raw[["X", "Y", "Z"]].values
      transformed_raw_pts = np.dot(all_raw_pts, R.T) + T
      df_after = pd.DataFrame(
          transformed_raw_pts, index=df_raw.index, columns=["X", "Y", "Z"]
      )

      # Error Analysis for Common Points
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

      # --- UI Layout & Sub-Station Selection ---
      st.markdown("---")
      view_tab = st.radio(
          "📊 Select View Mode",
          options=[
              "Global Transformation & Error Analysis",
              "Sub-Station Analysis (STN)",
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
        st.dataframe(df_after.style.format("{:.4f}"))

        csv_data = df_after.reset_index().to_csv(
            index=False, header=False, float_format="%.4f"
        )
        st.download_button(
            label="📥 Download After Bestfit Result File (.CSV)",
            data=csv_data,
            file_name="P1W-SS16-B-W_after_Bestfit.CSV",
            mime="text/csv",
        )

      else:
        st.subheader("🏢 Sub-Station (STN) Data Inspection & Transformation")
        stn_dict = {
            "STN 1": uploaded_stn1,
            "STN 2": uploaded_stn2,
            "STN 3": uploaded_stn3,
        }
        available_stns = {
            k: v for k, v in stn_dict.items() if v is not None
        }

        if not available_stns:
          st.warning(
              "⚠️ Please upload at least one Sub-Station (STN 1, STN 2, or STN 3)"
              " file in the sidebar to use this feature."
          )
        else:
          selected_stn_name = st.selectbox(
              "Select Sub-Station to View", options=list(available_stns.keys())
          )
          stn_file = available_stns[selected_stn_name]

          df_stn = pd.read_csv(
              stn_file, header=None, names=["Point", "X", "Y", "Z"]
          )
          df_stn["Point"] = df_stn["Point"].astype(str).str.strip()

          st.markdown(f"**Raw Data Preview for {selected_stn_name}:**")
          st.dataframe(df_stn.set_index("Point").style.format("{:.4f}"))

          # Apply global transformation R and T to this sub-station
          stn_pts = df_stn[["X", "Y", "Z"]].values
          transformed_stn_pts = np.dot(stn_pts, R.T) + T
          df_stn_after = pd.DataFrame(
              transformed_stn_pts,
              index=df_stn["Point"],
              columns=["X", "Y", "Z"],
          )

          st.markdown(
              f"**Aligned Coordinates Result for {selected_stn_name}:**"
          )
          st.dataframe(df_stn_after.style.format("{:.4f}"))

          stn_csv_data = df_stn_after.reset_index().to_csv(
              index=False, header=False, float_format="%.4f"
          )
          st.download_button(
              label=(
                  f"📥 Download Aligned {selected_stn_name} Result File (.CSV)"
              ),
              data=stn_csv_data,
              file_name=f"P1W-SS16-B-W_{selected_stn_name.replace(' ', '_')}_after.CSV",
              mime="text/csv",
          )

  except Exception as e:
    st.error(f"An error occurred during processing: {e}")
else:
  st.info(
      "👈 Please upload both the **Design / Control CSV** and **Raw / Before"
      " CSV** files in the left sidebar to begin."
  )
