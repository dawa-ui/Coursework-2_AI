from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__, static_folder='.')
CORS(app)

# --- Configuration & Paths ---
MODEL_DIR = r"c:\Users\adhik\OneDrive\Desktop\Coursework -2 AI\job_recommendation_system\backend\models"
JOBS_CSV = os.path.join(MODEL_DIR, "jobs_with_features.csv")
EMBEDDINGS_NPY = os.path.join(MODEL_DIR, "sbert_job_embeddings.npy")
MODEL_NAME_TXT = os.path.join(MODEL_DIR, "sbert_model_name.txt")

ALPHA = 0.8  # Hybrid weight (α for semantic, 1-α for collaborative)

# --- Load Resources ---
print("Loading model name...")
with open(MODEL_NAME_TXT, "r") as f:
    sbert_model_name = f.read().strip()

print(f"Loading SBERT model: {sbert_model_name}...")
model = SentenceTransformer(sbert_model_name)

print("Loading job data and embeddings...")
jobs_df = pd.read_csv(JOBS_CSV)
job_embeddings = np.load(EMBEDDINGS_NPY)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    """
    POST /recommend endpoint
    Computes hybrid scores using semantic similarity and simulated collaborative scores.
    """
    try:
        data = request.get_json()
        profile_text = data.get('profile_text', '')
        location_pref = data.get('location', '').lower()
        job_type_pref = data.get('job_type', '').lower()

        if not profile_text:
            return jsonify({"error": "Profile text is required"}), 400

        # 1. Compute Semantic Similarity
        # Encode user input and compute cosine similarity against precomputed embeddings
        user_embedding = model.encode([profile_text])
        semantic_scores = cosine_similarity(user_embedding, job_embeddings).flatten()

        # 2. Simulated Collaborative Filtering Score
        # In a real system, this would be based on user history. 
        # Here we simulate it to demonstrate the hybrid logic.
        np.random.seed(42) # Fixed seed for demonstration consistency
        collaborative_scores = np.random.uniform(0, 1, len(jobs_df))

        # 3. Hybrid Score Calculation
        # final_score = α × semantic_similarity + (1 − α) × collaborative_score
        final_scores = ALPHA * semantic_scores + (1 - ALPHA) * collaborative_scores

        # Create a copy for results and add scores
        results_df = jobs_df.copy()
        results_df['hybrid_score'] = final_scores

        # Optional: Apply hard filters if location or job_type are specified
        if location_pref:
            results_df = results_df[results_df['location'].str.lower().str.contains(location_pref, na=False)]
        
        if job_type_pref:
            results_df = results_df[results_df['job_type'].str.lower().str.contains(job_type_pref, na=False)]

        # Get top 5 recommendations
        top_5 = results_df.sort_values(by='hybrid_score', ascending=False).head(5)

        # Fallback if filters are too restrictive
        if top_5.empty:
            results_df = jobs_df.copy()
            results_df['hybrid_score'] = final_scores
            top_5 = results_df.sort_values(by='hybrid_score', ascending=False).head(5)

        # Prepare JSON response
        output = []
        for _, row in top_5.iterrows():
            output.append({
                "job title": row['job_title'],
                "company name": row['company_name'],
                "location": row['location'],
                "job type": row['job_type'],
                "salary": row['salary_formatted'] if pd.notna(row['salary_formatted']) else "N/A"
            })

        return jsonify(output)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Backend running at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
