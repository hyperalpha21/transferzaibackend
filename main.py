import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from scipy.special import expit
import torch

# === Logistic regression coefficients ===
B0 = -8.980140381396076
B1 = 7.763385577321117
B2 = 6.11064786318868

st.set_page_config(page_title="TransferAI", layout="centered")

st.title("🎓 TransferAI Course Matcher")
st.write("Find the most likely WM equivalent course & transfer probability.")

# ============================
# ✅ CACHE LOADING FUNCTIONS
# ============================

@st.cache_resource
def load_model():
    """Load SentenceTransformer model (cached for performance)."""
    return SentenceTransformer("BAAI/bge-small-en-v1.5")

@st.cache_data
def load_data():
    """Load WM courses CSV (cached)."""
    df = pd.read_csv("wm_courses_2025.csv", encoding="latin1").fillna("")
    df["full_text"] = df["course_title"] + ". " + df["course_description"]
    return df

@st.cache_resource
def precompute_embeddings(model, df):
    """Precompute embeddings for WM catalog."""
    return model.encode(
        df["full_text"].tolist(),
        batch_size=32,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

# ============================
# ✅ INIT
# ============================
st.write("⏳ Loading WM catalog & model...")
wm_df = load_data()
model = load_model()
wm_embeddings = precompute_embeddings(model, wm_df)
st.success("✅ Catalog & model loaded!")

# ============================
# ✅ UTILITY FUNCTION
# ============================
def get_transfer_prediction(title: str, desc: str):
    """Predict the closest course & transfer probability."""
    title_emb_input = model.encode([title], convert_to_tensor=True, normalize_embeddings=True)
    desc_emb_input  = model.encode([desc],  convert_to_tensor=True, normalize_embeddings=True)

    # Find closest WM course
    sim_title_all = util.cos_sim(title_emb_input, wm_embeddings)[0]
    sim_desc_all  = util.cos_sim(desc_emb_input, wm_embeddings)[0]
    sim_combined  = 0.5 * sim_title_all + 0.5 * sim_desc_all
    best_idx = int(sim_combined.argmax())
    best_course = wm_df.iloc[best_idx]

    # Compute exact title/desc similarities for logit
    best_title_emb = model.encode([best_course['course_title']], convert_to_tensor=True, normalize_embeddings=True)
    best_desc_emb  = model.encode([best_course['course_description']], convert_to_tensor=True, normalize_embeddings=True)
    title_mp = float(util.cos_sim(title_emb_input, best_title_emb)[0][0])
    des_mp   = float(util.cos_sim(desc_emb_input, best_desc_emb)[0][0])
    prob = float(expit(B0 + B1 * title_mp + B2 * des_mp))

    # Confidence tier
    if prob >= 0.80:
        tier = "✅ **Very Likely Transfer**"
    elif prob >= 0.60:
        tier = "⚠️ **Likely (Needs Review)**"
    elif prob >= 0.38:
        tier = "🤔 **Weak Match**"
    else:
        tier = "❌ **Unlikely Transfer**"

    return {
        "closest_match": f"{best_course['course_code']} | {best_course['course_title']}",
        "transfer_probability": round(prob * 100, 2),
        "tier": tier
    }

# ============================
# ✅ STREAMLIT FORM
# ============================
st.subheader("Enter a Course to Evaluate")

with st.form("course_form"):
    course_title = st.text_input("📖 Course Title", placeholder="e.g. Introduction to Microeconomics")
    course_desc = st.text_area("📝 Course Description", placeholder="Paste the catalog description here...")
    submitted = st.form_submit_button("🔍 Predict Transfer")

if submitted:
    if not course_title or not course_desc:
        st.warning("⚠️ Please enter both a title and a description!")
    else:
        with st.spinner("Analyzing..."):
            result = get_transfer_prediction(course_title, course_desc)
        
        st.success("✅ Prediction Complete!")
        st.write(f"**Closest Match:** {result['closest_match']}")
        st.write(f"**Transfer Probability:** {result['transfer_probability']}%")
        st.write(result["tier"])
