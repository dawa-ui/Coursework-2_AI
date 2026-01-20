# Job Recommendation System - Academic Prototype

This prototype demonstrates a **Hybrid Recommendation System** that combines SBERT-based semantic similarity with a simulated collaborative filtering score.

## Features
- **SBERT Semantic Similarity**: Uses the `all-MiniLM-L6-v2` model to compute similarity between user input and job descriptions.
- **Hybrid Scoring**: Combines content-based scores with simulated collaborative filtering scores.
- **Dynamic UI**: Simple, clean academic interface built with Vanilla JavaScript and CSS.

### Hybrid Score Formula
The final score is computed as:
`final_score = α × semantic_similarity + (1 − α) × collaborative_filtering_score`
Where:
- `α = 0.8` (Fixed constant)
- `collaborative_filtering_score` is simulated for demonstration purposes.

## How to Run
1. Ensure you have Python installed.
2. Navigate to the `prototype` directory.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Flask application:
   ```bash
   python app.py
   ```
5. Open your browser and go to `http://127.0.0.1:5000`.

## Files
- `app.py`: Flask backend implementing the recommendation logic.
- `index.html`: Frontend UI.
- `requirements.txt`: Python package dependencies.
