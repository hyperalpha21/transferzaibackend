import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from scipy.special import expit
from io import BytesIO
import os

# ============================
# LOGISTIC REGRESSION COEFFICIENTS
# ============================
B0 = -8.980140381396076
B1 = 7.763385577321117
B2 = 6.11064786318868

# Path to WM catalog
CSV_PATH = "wm_courses_2025.csv"

# ============================
# PAGE CONFIG
# ============================
st.set_page_config(page_title="TransferzAI", layout="wide", page_icon="🎓")
st.title("🎓 TransferzAI – Course Transfer Predictor")
st.write("Paste one or more courses in `Title | Description` format to find the closest WM match & transfer likelihood.")

# ============================
# CACHED LOADING
# ============================
@st.cache_data
def load_data():
    if not os.path.exists(CSV_PATH):
        st.error(f"CSV file `{CSV_PATH}` not found. Please upload it.")
        st.stop()
    df = pd.read_csv(CSV_PATH, encoding="latin1").fillna("")
    df["full_text"] = df["course_title"] + ". " + df["course_description"]
    return df

@st.cache_resource
def load_model_and_embeddings():
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    df = load_data()
    embeddings = model.encode(
        df["full_text"].tolist(),
        batch_size=32,
        convert_to_tensor=True,
        normalize_embeddings=True
    )
    return model, df, embeddings

# ============================
# LOAD MODEL + CATALOG
# ============================
with st.spinner("⏳ Loading WM catalog & AI model… (first run ~30s)"):
    model, wm_df, wm_embeddings = load_model_and_embeddings()
st.success("✅ Model & WM course catalog ready!")

# ============================
# PREDICTION FUNCTION
# ============================
def predict_transfer(title: str, desc: str):
    # Encode input course
    title_emb = model.encode([title], convert_to_tensor=True, normalize_embeddings=True)
    desc_emb  = model.encode([desc],  convert_to_tensor=True, normalize_embeddings=True)

    # Find closest WM match by combined similarity
    sim_title = util.cos_sim(title_emb, wm_embeddings)[0]
    sim_desc  = util.cos_sim(desc_emb, wm_embeddings)[0]
    sim_combined = 0.5 * sim_title + 0.5 * sim_desc
    best_idx = int(sim_combined.argmax())
    best_match = wm_df.iloc[best_idx]

    # Encode best match components
    best_title_emb = model.encode([best_match["course_title"]], convert_to_tensor=True, normalize_embeddings=True)
    best_desc_emb  = model.encode([best_match["course_description"]], convert_to_tensor=True, normalize_embeddings=True)

    # Compute exact similarities for logistic regression
    title_match_prob = float(util.cos_sim(title_emb, best_title_emb)[0][0])
    desc_match_prob  = float(util.cos_sim(desc_emb,  best_desc_emb)[0][0])
    prob = float(expit(B0 + B1 * title_match_prob + B2 * desc_match_prob))

    # Tier classification
    if prob >= 0.80:
        tier, status = "✅ Very Likely Transfer", "Likely"
    elif prob >= 0.60:
        tier, status = "⚠️ Likely (Needs Review)", "Likely"
    elif prob >= 0.38:
        tier, status = "🤔 Weak Match", "Unlikely"
    else:
        tier, status = "❌ Unlikely Transfer", "Unlikely"

    return {
        "Title": title,
        "Closest WM Match": f"{best_match['course_code']} | {best_match['course_title']}",
        "Transfer Probability (%)": round(prob * 100, 2),
        "Result": tier,
        "Status": status
    }

# ============================
# USER INPUT SECTION
# ============================
st.subheader("📋 Enter Courses to Check")
st.write("""
Format:  
""")

input_text = st.text_area("Paste multiple courses here:", height=200)

# ============================
# PROCESS BUTTON
# ============================
if st.button("🔍 Predict Transfers"):
    if not input_text.strip():
        st.warning("⚠️ Please enter at least one course in the correct format!")
    else:
        with st.spinner("Analyzing all entered courses…"):
            lines = [line.strip() for line in input_text.split("\n") if line.strip()]
            results = []
            for line in lines:
                if "|" in line:
                    title, desc = [p.strip() for p in line.split("|", 1)]
                    results.append(predict_transfer(title, desc))
                else:
                    results.append({
                        "Title": line,
                        "Closest WM Match": "❌ Invalid format",
                        "Transfer Probability (%)": "-",
                        "Result": "❌ Skipped",
                        "Status": "Skipped"
                    })

            df_results = pd.DataFrame(results)

        # ============================
        # DISPLAY RESULTS
        # ============================
        def color_rows(row):
            if row["Status"] == "Likely":
                return ["background-color: #d4fcdc; color: #22543d;"] * len(row)
            elif row["Status"] == "Unlikely":
                return ["background-color: #fcd4d4; color: #742a2a;"] * len(row)
            else:
                return ["background-color: #f0f0f0; color: #555;"] * len(row)

        st.subheader("📊 Predictions")
        st.dataframe(df_results.style.apply(color_rows, axis=1), use_container_width=True)

        # ============================
        # SUMMARY COUNTS
        # ============================
        likely_count   = (df_results["Status"] == "Likely").sum()
        unlikely_count = (df_results["Status"] == "Unlikely").sum()
        skipped_count  = (df_results["Status"] == "Skipped").sum()

        st.markdown("---")
        st.subheader("📈 Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("✅ Likely Transfers", likely_count)
        col2.metric("❌ Unlikely Transfers", unlikely_count)
        col3.metric("⏭️ Skipped", skipped_count)

        # ============================
        # DOWNLOAD OPTION
        # ============================
        buffer = BytesIO()
        df_results.to_csv(buffer, index=False)
        st.download_button(
            label="💾 Download Results as CSV",
            data=buffer.getvalue(),
            file_name="transferai_results.csv",
            mime="text/csv"
        )
