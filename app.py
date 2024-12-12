from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from pydantic import BaseModel
from typing import List
import re
app = FastAPI()


# Load the LDA model and the dictionary
dictionary = Dictionary.load("artifacts/dictionary.pkl")
lda_model = LdaModel.load("artifacts/lda_model.pkl", mmap=None)

origins = ["*"]
app.add_middleware(
	CORSMiddleware,
	allow_origins=origins,
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

# Define a Pydantic model for the request body
class SentenceRequest(BaseModel):
    sentence: str

# Preprocess text function (ensure it matches training preprocessing)
def preprocess_text(text: str) -> List[str]:
    # Basic text cleaning (you can customize this as per your model's preprocessing)
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)
    tokens = text.split()
    return tokens

@app.post("/api/predict-topic")
async def predict_topic(request: SentenceRequest):
    sentence = request.sentence

    tokens = preprocess_text(sentence)
    if not tokens:
        raise HTTPException(status_code=400, detail="The input sentence could not be processed into valid tokens.")

    # Convert to Bag-of-Words
    bow = dictionary.doc2bow(tokens)

    # Get topic probabilities
    topic_probs = lda_model.get_document_topics(bow)

    if not topic_probs:
        return {"sentence": sentence, "topic": "No dominant topic", "topic_probabilities": topic_probs}

    # Find the dominant topic
    dominant_topic = max(topic_probs, key=lambda x: x[1])

    print(topic_probs)
    return {
        "sentence": sentence,
        "dominant_topic": {
        	"topic_id": int(dominant_topic[0]),
        	"probability": float(dominant_topic[1]),
        },
        "topic_probabilities": [
            {
            	"topic_id": int(topic[0]),
            	"probability": float(topic[1])
            } for topic in topic_probs
        ]
    }

    """
    return {
        "sentence": sentence,
        "dominant_topic": dominant_topic[0],
        "dominant_topic_probability": dominant_topic[1],
        "topic_probabilities": topic_probs
    }
    """

@app.get("/api/test")
async def test():
	return "Trabajo de Mineria!"
