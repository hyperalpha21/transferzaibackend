import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from scipy.special import expit
from io import BytesIO
import os

# === Logistic regression coefficients (new model) ===
B0, B1, B2 = -8.980140381396076, 7.763385577321117, 6.11064786318868

# === CSV path ===
CSV_PATH = "wm_courses_2025.csv"

# === Streamlit basic config ===
st.set_page_config(
    page_title="TransferzAI",
    layout="wide",
    page_icon="🎓"
)
st.title("🎓 TransferzAI – Course Transfer Predictor")
st.write("Paste one or more courses in `Title | Description` format to find the closest WM match & transfer likelihood.")

# === Cache model, data, embeddings ===
@st.cache_resource
def load_model_and_data():
    if not os.path.exists(CSV_PATH):
        st.error(f"❌ Could not find `{CSV_PATH}`. Make sure it's in the repo.")
        st.stop()
    
    # Load catalog
    df = pd.read_csv(CSV_PATH, encoding="latin1").fillna("")
    df["full_text"] = df["course_title"] + ". " + df["course_description"]

    # Load model
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    # Precompute embeddings
    embeddings = model.encode(
        df["full_text"].tolist(),
        batch_size=32,
        convert_to_tensor=True,
        normalize_embeddings=True
    )
    return model, df, embeddings

# === Load once on startup ===
with st.spinner("⏳ Loading model & catalog… (first time ≈30s)"):
    model, wm_df, wm_embeddings = load_model_and_data()
st.success("✅ Model & catalog loaded!")

# === Core function ===
def predict_transfer(title, desc):
    # Encode user course
    t_emb = model.encode([title], convert_to_tensor=True, normalize_embeddings=True)
    d_emb = model.encode([desc],  convert_to_tensor=True, normalize_embeddings=True)

    # Find closest WM course by similarity
    sim_t = util.cos_sim(t_emb, wm_embeddings)[0]
    sim_d = util.cos_sim(d_emb, wm_embeddings)[0]
    combined_sim = 0.5 * sim_t + 0.5 * sim_d
    idx = int(combined_sim.argmax())
    best_course = wm_df.iloc[idx]

    # Logistic regression probability
    best_t_emb = model.encode([best_course["course_title"]], convert_to_tensor=True, normalize_embeddings=True)
    best_d_emb = model.encode([best_course["course_description"]], convert_to_tensor=True, normalize_embeddings=True)
    title_match = float(util.cos_sim(t_emb, best_t_emb)[0][0])
    desc_match  = float(util.cos_sim(d_emb, best_d_emb)[0][0])
    prob = float(expit(B0 + B1 * title_match + B2 * desc_match))

    # Confidence tier
    if prob >= 0.80:
        tier, status = "✅ Very Likely Transfer", "Likely"
    elif prob >= 0.60:
        tier, status = "⚠️ Likely (Needs Review)", "Likely"
    elif prob >= 0.38:
        tier, status = "🤔 Weak Match", "Unlikely"
    else:
        tier, status = "❌ Unlikely Transfer", "Unlikely"

    return {
        "Input Title": title,
        "Closest WM Match": f"{best_course['course_code']} | {best_course['course_title']}",
        "Transfer Probability (%)": round(prob * 100, 2),
        "Result": tier,
        "Status": status
    }

# === UI ===
st.subheader("📋 Enter Multiple Courses")
st.write(
    "Format each line as:\n```\nIntro to Economics | Supply & demand, markets\nLinear Algebra | Matrices, vectors, transformations\n```"
)

user_input = st.text_area("Paste courses here", height=200)

if st.button("🔍 Predict Transfers"):
    if not user_input.strip():
        st.warning("⚠️ Please enter at least one course!")
    else:
        with st.spinner("Analyzing all courses…"):
            lines = [line.strip() for line in user_input.split("\n") if line.strip()]
            results = []
            for line in lines:
                if "|" in line:
                    title, desc = [p.strip() for p in line.split("|", 1)]
                    results.append(predict_transfer(title, desc))
                else:
                    results.append({
                        "Input Title": line,
                        "Closest WM Match": "❌ Invalid format",
                        "Transfer Probability (%)": "-",
                        "Result": "❌ Skipped",
                        "Status": "Skipped"
                    })
            df_results = pd.DataFrame(results)

        # === Color rows ===
        def color_rows(row):
            if row["Status"] == "Likely":
                return ["background-color: #c6f6d5; color: #22543d;"] * len(row)  # green
            elif row["Status"] == "Unlikely":
                return ["background-color: #fed7d7; color: #742a2a;"] * len(row)  # red
            else:
                return ["background-color: #edf2f7; color: #4a5568;"] * len(row)  # gray

        # Show table
        st.subheader("📊 Predictions")
        st.dataframe(df_results.style.apply(color_rows, axis=1), use_container_width=True)

        # Summary counts
        likely_count = (df_results["Status"] == "Likely").sum()
        unlikely_count = (df_results["Status"] == "Unlikely").sum()
        skipped_count = (df_results["Status"] == "Skipped").sum()

        st.markdown("---")
        st.subheader("📈 Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("✅ Likely Transfers", likely_count)
        c2.metric("❌ Unlikely Transfers", unlikely_count)
        c3.metric("⏭️ Skipped", skipped_count)

        # CSV download
        csv_buffer = BytesIO()
        df_results.to_csv(csv_buffer, index=False)
        st.download_button(
            label="💾 Download Results as CSV",
            data=csv_buffer.getvalue(),
            file_name="transferai_results.csv",
            mime="text/csv",
        )
