""")

courses_input = st.text_area("Paste courses here", height=200, placeholder="Course Title | Course Description")

if st.button("🔍 Predict Transfers"):
    if not courses_input.strip():
        st.warning("⚠️ Please enter at least one course!")
    else:
        with st.spinner("Analyzing all courses..."):
            lines = [line.strip() for line in courses_input.split("\n") if line.strip()]
            
            results = []
            for line in lines:
                if "|" in line:
                    title, desc = [p.strip() for p in line.split("|", 1)]
                    res = predict_transfer(title, desc)
                    results.append(res)
                else:
                    results.append({
                        "Input Title": line,
                        "Closest WM Match": "❌ Invalid format",
                        "Transfer Probability (%)": "-",
                        "Result": "❌ Skipped (No description)",
                        "Status": "Skipped"
                    })
            
            df_results = pd.DataFrame(results)

        # ✅ Color-code likely/unlikely
        def color_rows(row):
            if row["Status"] == "Likely":
                return ["background-color: #c6f6d5; color: #22543d;"] * len(row)  # green
            elif row["Status"] == "Unlikely":
                return ["background-color: #fed7d7; color: #742a2a;"] * len(row)  # red
            else:
                return ["background-color: #edf2f7; color: #4a5568;"] * len(row)  # gray

        # ✅ Show results nicely
        st.subheader("📊 Predictions")
        st.dataframe(
            df_results.style.apply(color_rows, axis=1),
            use_container_width=True
        )

        # ✅ Summary counts
        likely_count = sum(df_results["Status"] == "Likely")
        unlikely_count = sum(df_results["Status"] == "Unlikely")
        skipped_count = sum(df_results["Status"] == "Skipped")

        st.markdown("---")
        st.subheader("📈 Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("✅ Likely Transfers", likely_count)
        col2.metric("❌ Unlikely Transfers", unlikely_count)
        col3.metric("⏭️ Skipped", skipped_count)

        # ✅ Allow CSV download
        csv_buffer = BytesIO()
        df_results.to_csv(csv_buffer, index=False)
        st.download_button(
            label="💾 Download Results as CSV",
            data=csv_buffer.getvalue(),
            file_name="transferai_results.csv",
            mime="text/csv",
        )
