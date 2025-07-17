import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from scipy.special import expit
from io import BytesIO
import os

# === Logistic regression coefficients ===
B0, B1, B2 = -8.980140381396076, 7.763385577321117, 6.11064786318868

# === File path ===
CSV_PATH = "wm_courses_2025.csv"

# === Streamlit page config ===
st.set_page_config(page_title="TransferzAI", layout="wide", page_icon="🎓")
st.title("TransferzAI – Course Transfer Predictor")
st.write(
    "Paste one or more courses in `Title | Description` format to find the closest WM match & transfer likelihood."
)

# === Single cached loader ===
@st.cache_resource
def load_model_and_data():
    """Load the model, CSV, and precompute embeddings once."""
    # Check CSV exists
    if not os.path.exists(CSV_PATH):
        st.error(f"❌ CSV file `{CSV_PATH}` not found. Please add it to the repo.")
        st.stop()

    # Load course data
    df = pd.read_csv(CSV_PATH, encoding="latin1").fillna("")
    df["full_text"] = df["course_title"] + ". " + df["course_description"]

    # Load model & compute embeddings
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    embeddings = model.encode(
        df["full_text"].tolist(),
        batch_size=32,
        convert_to_tensor=True,
        normalize_embeddings=True
    )
    return model, df, embeddings

# === Load everything on startup ===
with st.spinner("⏳ Loading model & course catalog (first run may take ~30s)..."):
    model, wm_df, wm_embeddings = load_model_and_data()
st.success("✅ Model & catalog ready!")

# === Prediction function ===
def predict_transfer(title, desc):
    """Predict closest WM course and likelihood."""
    # Encode input text
    title_emb = model.encode([title], convert_to_tensor=True, normalize_embeddings=True)
    desc_emb  = model.encode([desc],  convert_to_tensor=True, normalize_embeddings=True)

    # Find closest WM course
    sim_title = util.cos_sim(title_emb, wm_embeddings)[0]
    sim_desc  = util.cos_sim(desc_emb, wm_embeddings)[0]
    sim_combined = 0.5 * sim_title + 0.5 * sim_desc
    best_idx = int(sim_combined.argmax())
    best_course = wm_df.iloc[best_idx]

    # Compute exact features for logistic model
    best_title_emb = model.encode([best_course["course_title"]], convert_to_tensor=True, normalize_embeddings=True)
    best_desc_emb  = model.encode([best_course["course_description"]], convert_to_tensor=True, normalize_embeddings=True)
    title_match = float(util.cos_sim(title_emb, best_title_emb)[0][0])
    desc_match  = float(util.cos_sim(desc_emb, best_desc_emb)[0][0])
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
        "Title": title,
        "WM Match": f"{best_course['course_code']} | {best_course['course_title']}",
        "Prob (%)": round(prob * 100, 2),
        "Tier": tier,
        "Status": status
    }

# === User input ===
st.subheader("Enter Courses")
st.write(
    "Format each line as:\n```\nIntro to Economics | Supply & demand, markets\nLinear Algebra | Matrices, vectors, transformations\n```"
)

input_text = st.text_area("Paste courses here", height=200)

# === When user clicks Predict ===
if st.button("Predict Transfers"):
    if not input_text.strip():
        st.warning("⚠️ Please enter at least one course.")
    else:
        with st.spinner("🔍 Analyzing courses..."):
            # Split lines and process
            lines = [line.strip() for line in input_text.split("\n") if line.strip()]
            results = []
            for line in lines:
                if "|" in line:
                    title, desc = [p.strip() for p in line.split("|", 1)]
                    results.append(predict_transfer(title, desc))
                else:
                    results.append({
                        "Title": line,
                        "WM Match": "❌ Invalid format",
                        "Prob (%)": None,
                        "Tier": "Skipped",
                        "Status": "Skipped"
                    })

            df = pd.DataFrame(results)

        # === Color rows ===
        def highlight_rows(row):
            if row.Status == "Likely":
                return ["background-color: #d4fcdc"] * len(row)
            elif row.Status == "Unlikely":
                return ["background-color: #fcd4d4"] * len(row)
            else:
                return [""] * len(row)

        st.subheader("Results")
        st.dataframe(df.style.apply(highlight_rows, axis=1), use_container_width=True)

        # === Summary ===
        likely = (df.Status == "Likely").sum()
        unlikely = (df.Status == "Unlikely").sum()
        skipped = (df.Status == "Skipped").sum()

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        col1.metric("Likely Transfers", likely)
        col2.metric("Unlikely Transfers", unlikely)
        col3.metric("Skipped", skipped)

        # === CSV Download ===
        buf = BytesIO()
        df.to_csv(buf, index=False)
        st.download_button(
            "💾 Download CSV",
            buf.getvalue(),
            "transfer_results.csv",
            "text/csv"
        )
