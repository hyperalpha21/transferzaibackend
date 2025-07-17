import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from scipy.special import expit
from io import BytesIO

# === Logistic regression coefficients ===
B0, B1, B2 = -8.980140381396076, 7.763385577321117, 6.11064786318868

# ============================
# 1) PAGE CONFIG & STYLES
# ============================
st.set_page_config(
    page_title="TransferAI | Course Matcher",
    layout="wide",
    page_icon="🎓",
)

st.markdown(
    """
    <style>
    /* Background & fonts */
    .stApp {
        background: linear-gradient(135deg, #f7f8fa, #eef1f7);
        font-family: "Segoe UI", sans-serif;
    }
    /* Titles */
    h1, h2, h3 {
        color: #1a365d;
    }
    /* Buttons */
    .stButton > button {
        background-color: #2b6cb0;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6em 1em;
    }
    .stButton > button:hover {
        background-color: #2c5282;
    }
    /* Metric text */
    div[data-testid="stMetricValue"] {
        color: #2b6cb0;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎓 TransferAI – Multi-Course Transfer Predictor")
st.write("Paste in **one or more courses** (Title | Description) to find the closest WM match & transfer likelihood.")

# ============================
# 2) CACHED LOADING
# ============================
@st.cache_data
def load_data():
    df = pd.read_csv("wm_courses_2025.csv", encoding="latin1").fillna("")
    df["full_text"] = df["course_title"] + ". " + df["course_description"]
    return df

@st.cache_resource
def load_model():
    return SentenceTransformer("BAAI/bge-small-en-v1.5")

@st.cache_resource
def load_embeddings():
    m = SentenceTransformer("BAAI/bge-small-en-v1.5")
    df = pd.read_csv("wm_courses_2025.csv", encoding="latin1").fillna("")
    df["full_text"] = df["course_title"] + ". " + df["course_description"]
    return m.encode(df["full_text"].tolist(), batch_size=32,
                    convert_to_tensor=True, normalize_embeddings=True)

with st.spinner("⏳ Loading model & catalog… (first time ≈30s)"):
    wm_df = load_data()
    model = load_model()
    wm_embeddings = load_embeddings()
st.success("✅ Model & embeddings ready!")

# ============================
# 3) PREDICTION FUNC
# ============================
def predict_transfer(title: str, desc: str):
    t_emb = model.encode([title], convert_to_tensor=True, normalize_embeddings=True)
    d_emb = model.encode([desc],  convert_to_tensor=True, normalize_embeddings=True)
    sim_t = util.cos_sim(t_emb, wm_embeddings)[0]
    sim_d = util.cos_sim(d_emb, wm_embeddings)[0]
    sim = 0.5 * sim_t + 0.5 * sim_d
    idx = int(sim.argmax())
    course = wm_df.iloc[idx]
    # compute features for logistic
    ct = model.encode([course["course_title"]], convert_to_tensor=True, normalize_embeddings=True)
    cd = model.encode([course["course_description"]], convert_to_tensor=True, normalize_embeddings=True)
    mp_t = float(util.cos_sim(t_emb, ct)[0][0])
    mp_d = float(util.cos_sim(d_emb, cd)[0][0])
    p = float(expit(B0 + B1 * mp_t + B2 * mp_d))
    if p >= 0.80:
        tier, status = "✅ Very Likely Transfer", "Likely"
    elif p >= 0.60:
        tier, status = "⚠️ Likely (Needs Review)", "Likely"
    elif p >= 0.38:
        tier, status = "🤔 Weak Match", "Unlikely"
    else:
        tier, status = "❌ Unlikely Transfer", "Unlikely"
    return {
        "Input Title": title,
        "Closest WM Match": f"{course['course_code']} | {course['course_title']}",
        "Transfer Probability (%)": round(p * 100, 2),
        "Result": tier,
        "Status": status,
    }

# ============================
# 4) UI: MULTI-COURSE INPUT
# ============================
st.subheader("📋 Enter Multiple Courses")
st.write("""
**Format:**  
`Course Title | Course Description`  
(one per line)  
""")

courses_text = st.text_area(
    "Paste courses here",
    height=200,
    placeholder="Intro to Economics | Supply & demand, markets…\nLinear Algebra | Vectors, matrices, determinants…"
)

if st.button("🔍 Predict Transfers"):
    if not courses_text.strip():
        st.warning("⚠️ Please enter at least one course!")
    else:
        with st.spinner("Analyzing all courses…"):
            lines = [l.strip() for l in courses_text.split("\n") if l.strip()]
            rows = []
            for l in lines:
                if "|" in l:
                    t, d = [p.strip() for p in l.split("|", 1)]
                    rows.append(predict_transfer(t, d))
                else:
                    rows.append({
                        "Input Title": l,
                        "Closest WM Match": "❌ Invalid format",
                        "Transfer Probability (%)": "-",
                        "Result": "❌ Skipped", 
                        "Status": "Skipped",
                    })
            df = pd.DataFrame(rows)

        # color rows
        def color_rows(r):
            if r["Status"] == "Likely":
                return ["background-color: #c6f6d5; color: #22543d"] * len(r)
            if r["Status"] == "Unlikely":
                return ["background-color: #fed7d7; color: #742a2a"] * len(r)
            return ["background-color: #edf2f7; color: #4a5568"] * len(r)

        st.subheader("📊 Predictions")
        st.dataframe(df.style.apply(color_rows, axis=1), use_container_width=True)

        # summary metrics
        l = (df["Status"] == "Likely").sum()
        u = (df["Status"] == "Unlikely").sum()
        s = (df["Status"] == "Skipped").sum()
        st.markdown("---")
        st.subheader("📈 Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("✅ Likely Transfers", l)
        c2.metric("❌ Unlikely Transfers", u)
        c3.metric("⏭️ Skipped", s)

        # CSV download
        buf = BytesIO()
        df.to_csv(buf, index=False)
        st.download_button(
            "💾 Download Results as CSV",
            data=buf.getvalue(),
            file_name="transferai_results.csv",
            mime="text/csv"
        )
