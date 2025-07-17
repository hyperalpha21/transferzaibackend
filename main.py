import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from scipy.special import expit
from io import BytesIO

# Logistic regression coefficients
B0, B1, B2 = -8.980140381396076, 7.763385577321117, 6.11064786318868

st.set_page_config(page_title="TransferAI", layout="wide", page_icon="🎓")
st.title("TransferAI – Course Transfer Predictor")
st.write("Paste one or more courses (Title | Description) to find the closest WM match and transfer likelihood.")

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
    return m.encode(df["full_text"].tolist(),
                    batch_size=32,
                    convert_to_tensor=True,
                    normalize_embeddings=True)

with st.spinner("Loading model and data… (first run may take ~30s)"):
    wm_df = load_data()
    model = load_model()
    wm_embeddings = load_embeddings()
st.success("Ready!")

def predict_transfer(title: str, desc: str):
    t_emb = model.encode([title], convert_to_tensor=True, normalize_embeddings=True)
    d_emb = model.encode([desc],  convert_to_tensor=True, normalize_embeddings=True)
    sim_t = util.cos_sim(t_emb, wm_embeddings)[0]
    sim_d = util.cos_sim(d_emb, wm_embeddings)[0]
    sim = 0.5 * sim_t + 0.5 * sim_d
    idx = int(sim.argmax())
    course = wm_df.iloc[idx]

    # logistic features
    ct = model.encode([course["course_title"]], convert_to_tensor=True, normalize_embeddings=True)
    cd = model.encode([course["course_description"]], convert_to_tensor=True, normalize_embeddings=True)
    mp_t = float(util.cos_sim(t_emb, ct)[0][0])
    mp_d = float(util.cos_sim(d_emb, cd)[0][0])
    p = float(expit(B0 + B1 * mp_t + B2 * mp_d))

    if p >= 0.80:
        tier, status = "Very Likely Transfer", "Likely"
    elif p >= 0.60:
        tier, status = "Likely (Review)", "Likely"
    elif p >= 0.38:
        tier, status = "Weak Match", "Unlikely"
    else:
        tier, status = "Unlikely Transfer", "Unlikely"

    return {
        "Title": title,
        "WM Match": f"{course['course_code']} | {course['course_title']}",
        "Prob (%)": round(p * 100, 2),
        "Tier": tier,
        "Status": status
    }

st.subheader("Enter Courses")
st.write("Format each line as `Title | Description`. Example:\n```\nIntro to Economics | Supply & demand, markets\nLinear Algebra | Matrices, vectors, transformations\n```")

courses_text = st.text_area("Paste courses here", height=200)

if st.button("Predict Transfers"):
    if not courses_text.strip():
        st.warning("Please enter at least one course.")
    else:
        with st.spinner("Analyzing…"):
            lines = [l.strip() for l in courses_text.split("\n") if l.strip()]
            results = []
            for l in lines:
                if "|" in l:
                    t, d = [p.strip() for p in l.split("|", 1)]
                    results.append(predict_transfer(t, d))
                else:
                    results.append({
                        "Title": l,
                        "WM Match": "Invalid format",
                        "Prob (%)": None,
                        "Tier": "Skipped",
                        "Status": "Skipped"
                    })
            df = pd.DataFrame(results)

        # color rows
        def color_row(r):
            if r.Status == "Likely":
                return ["background-color: #d4fcdc"] * len(r)
            if r.Status == "Unlikely":
                return ["background-color: #fcd4d4"] * len(r)
            return [""] * len(r)

        st.subheader("Results")
        st.dataframe(df.style.apply(color_row, axis=1), use_container_width=True)

        # summary
        l = (df.Status == "Likely").sum()
        u = (df.Status == "Unlikely").sum()
        s = (df.Status == "Skipped").sum()
        st.markdown("---")
        st.subheader("Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Likely", l)
        col2.metric("Unlikely", u)
        col3.metric("Skipped", s)

        # download
        buf = BytesIO()
        df.to_csv(buf, index=False)
        st.download_button("Download CSV", buf.getvalue(), "results.csv", "text/csv")
