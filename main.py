from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from scipy.special import expit
import pandas as pd

# === Logistic regression coefficients ===
B0 = -8.980140381396076
B1 = 7.763385577321117
B2 = 6.11064786318868

print("✅ Loading WM catalog...")
wm_df = pd.read_csv("wm_courses_2025.csv", encoding="latin1").fillna("")
wm_df["full_text"] = wm_df["course_title"] + ". " + wm_df["course_description"]

print("✅ Loading embedding model...")
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

print("✅ Precomputing embeddings...")
wm_embeddings = model.encode(
    wm_df["full_text"].tolist(),
    batch_size=32,
    convert_to_tensor=True,
    normalize_embeddings=True
)

app = FastAPI(title="TransferAI Backend")

# Request format
class CourseInput(BaseModel):
    title: str
    desc: str

@app.get("/")
def root():
    return {"message": "✅ TransferAI backend is live!"}

@app.post("/predict")
def predict_transfer(course: CourseInput):
    # Encode input
    title_emb_input = model.encode([course.title], convert_to_tensor=True, normalize_embeddings=True)
    desc_emb_input  = model.encode([course.desc],  convert_to_tensor=True, normalize_embeddings=True)

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
        tier = "✅ Very Likely Transfer"
    elif prob >= 0.60:
        tier = "⚠️ Likely (Needs Review)"
    elif prob >= 0.38:
        tier = "🤔 Weak Match"
    else:
        tier = "❌ Unlikely Transfer"

    return {
        "closest_match": f"{best_course['course_code']} | {best_course['course_title']}",
        "transfer_probability": round(prob * 100, 2),
        "tier": tier
    }
