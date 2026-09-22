if "df_final_result" in st.session_state:
                        st.markdown("### 📊 Active Source Preview, Deviation & Flatness Analysis")
                        
                        # --- 🌟 新增：Dimensional Control Flatness 自动分析报告 ---
                        z_vals = active_cad_df["Z"].astype(float).values
                        max_dev = np.max(z_vals)
                        min_dev = np.min(z_vals)
                        total_flatness = max_dev - min_dev
                        
                        # 如果在 Step 4 设定了目标 Z 值（target_z_final），则以该目标基准计算 RMS；否则默认以 0.0 为基准
                        tz = target_z_final if (source_label_str == "Step4" and 'target_z_final' in locals()) else 0.0
                        rms = np.sqrt(np.mean((z_vals - tz)**2))
                        
                        st.success(
                            f"**📏 Dimensional Control Flatness Report ({len(z_vals)} points evaluated):**\n\n"
                            f"🔹 **Max Positive Deviation:** `{max_dev:+.4f}`  \n"
                            f"🔹 **Max Negative Deviation:** `{min_dev:+.4f}`  \n"
                            f"🔹 **Total Flatness (Max - Min):** `{total_flatness:.4f}`  \n"
                            f"🔹 **RMS (Root Mean Square):** `{rms:.4f}`"
                        )
                        # ------------------------------------------------------------

                        col_res1, col_res2 = st.columns(2)

                        with col_res1:
                            st.markdown(f"#### 📋 {source_label_str} Result Preview")
                            st.dataframe(active_cad_df.style.format("{:.4f}"), use_container_width=True)

                        with col_res2:
                            st.markdown(f"#### 📊 {source_label_str} Deviation Analysis")
                            if not active_err_df.empty:
                                st.dataframe(
                                    active_err_df
                                    .style.format({"Delta E": "{:.4f}", "Delta N": "{:.4f}", "Delta El": "{:.4f}", "Total_Error": "{:.4f}"})
                                    .map(highlight_excess_error, subset=["Delta E", "Delta N", "Delta El", "Total_Error"]),
                                    use_container_width=True,
                                )
                            else:
                                st.info("No deviation data available for this source.")

                        step_csv_data = active_cad_df.reset_index().to_csv(index=False, header=False, float_format="%.4f")
                        st.download_button(
                            label=f"📥 Download [{source_label_str}_Result.CSV]",
                            data=step_csv_data,
                            file_name=f"station-Combine_All_{source_label_str}_Result.CSV",
                            mime="text/csv",
                        )
