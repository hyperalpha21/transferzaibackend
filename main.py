import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from scipy.special import expit
from io import BytesIO
import os

# Logistic regression coefficients
B0, B1, B2 = -8.980140381396076, 7.763385577321117, 6.11064786318868

# ============================
# PAGE CONFIG
# ============================
st.set_page_config(
    page_title="TransferzAI",
    layout="wide",
    page_icon="🎓"
)
st.title("TransferzAI – Course Transfer Predictor")
st.write(
    "Paste one or more courses (Title | Description) to find the closest William & Mary match "
    "and transfer likelihood."
)

CSV_PATH = "wm_courses_2025.csv"

# ============================
# CACHED LOADING
# ============================
@st.cache_data
def load_data():
    if not os.path.exists(CSV_PATH):
        st.error(f"CSV file `{CSV_PATH}` not found in app directory.")
        st.stop()
    df = pd.read_csv(CSV_PATH, encoding="latin1").fillna("")
    df["full_text"] = df["course_title"] + ". " + df["course_description"]
    return df

@st.cache_resource
def load_model():
    try:
        return SentenceTransformer("BAAI/bge-small-en-v1.5")
    except Exception as e:
        st.error("❌ Model failed to load. Check internet or model availability.")
        st.stop()

@st.cache_resource
def load_embeddings():
    """Compute embeddings for WM courses at startup."""
    model_temp = SentenceTransformer("BAAI/bge-small-en-v1.5")
    df = pd.read_csv(CSV_PATH, encoding="latin1").fillna("")
    df["full_text"] = df["course_title"] + ". " + df["course_description"]
    return model_temp.encode(
        df["full_text"].tolist(),
        batch_size=32,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

# ============================
# LOAD EVERYTHING
# ============================
with st.spinner("Loading data and model… (first run may take up to ~30s)"):
    wm_df = load_data()
    model = load_model()
    wm_embeddings = load_embeddings()
st.success("✅ Data & model ready!")

# ============================
# PREDICTION FUNCTION
# ============================
def predict_transfer(title: str, desc: str):
    """Predict closest WM course and likelihood."""
    # Encode input
    t_emb = model.encode([title], convert_to_tensor=True, normalize_embeddings=True)
    d_emb = model.encode([desc],  convert_to_tensor=True, normalize_embeddings=True)

    # Find closest match
    sim_t = util.cos_sim(t_emb, wm_embeddings)[0]
    sim_d = util.cos_sim(d_emb, wm_embeddings)[0]
    sim = 0.5 * sim_t + 0.5 * sim_d
    idx = int(sim.argmax())
    course = wm_df.iloc[idx]

    # Compute logistic features
    ct = model.encode([course["course_title"]], convert_to_tensor=True, normalize_embeddings=True)
    cd = model.encode([course["course_description"]], convert_to_tensor=True, normalize_embeddings=True)
    mp_t = float(util.cos_sim(t_emb, ct)[0][0])
    mp_d = float(util.cos_sim(d_emb, cd)[0][0])
    prob = float(expit(B0 + B1 * mp_t + B2 * mp_d))

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
        "WM Match": f"{course['course_code']} | {course['course_title']}",
        "Prob (%)": round(prob * 100, 2),
        "Tier": tier,
        "Status": status
    }

# ============================
# MULTI-COURSE INPUT UI
# ============================
st.subheader("Enter Courses")
st.write(
    "Format each line as `Title | Description`. Example:\n\n"
    "```\nIntro to Economics | Supply & demand, markets\n"
    "Linear Algebra | Matrices, vectors, transformations\n```"
)

courses_text = st.text_area(
    "Paste courses here",
    height=200,
    placeholder="Intro to Economics | Supply & demand, markets…\nLinear Algebra | Vectors, matrices, determinants…"
)

# ============================
# PROCESS PREDICTIONS
# ============================
if st.button("Predict Transfers"):
    if not courses_text.strip():
        st.warning("⚠️ Please enter at least one course.")
    else:
        with st.spinner("Analyzing…"):
            lines = [line.strip() for line in courses_text.split("\n") if line.strip()]
            results = []
            for line in lines:
                if "|" in line:
                    title, desc = [p.strip() for p in line.split("|", 1)]
                    results.append(predict_transfer(title, desc))
                else:
                    # Invalid format
                    results.append({
                        "Title": line,
                        "WM Match": "Invalid format",
                        "Prob (%)": None,
                        "Tier": "Skipped",
                        "Status": "Skipped"
                    })
            df = pd.DataFrame(results)

        # Color rows for clarity
        def color_row(r):
            if r.Status == "Likely":
                return ["background-color: #d4fcdc"] * len(r)
            elif r.Status == "Unlikely":
                return ["background-color: #fcd4d4"] * len(r)
            else:
                return [""] * len(r)

        st.subheader("Results")
        st.dataframe(
            df.style.apply(color_row, axis=1),
            use_container_width=True
        )

        # Summary
        likely_count = (df.Status == "Likely").sum()
        unlikely_count = (df.Status == "Unlikely").sum()
        skipped_count = (df.Status == "Skipped").sum()

        st.markdown("---")
        st.subheader("Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Likely", likely_count)
        col2.metric("Unlikely", unlikely_count)
        col3.metric("Skipped", skipped_count)

        # Download results as CSV
        buf = BytesIO()
        df.to_csv(buf, index=False)
        st.download_button(
            label="💾 Download CSV",
            data=buf.getvalue(),
            file_name="transfer_results.csv",
            mime="text/csv"
        )
