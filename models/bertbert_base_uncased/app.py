from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertForSequenceClassification, BertTokenizer
import torch

# Initialize FastAPI app
app = FastAPI()

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained("./final_model", trust_remote_code=True)
tokenizer = BertTokenizer.from_pretrained("./final_model", trust_remote_code=True)

# Define input schema
class Tweet(BaseModel):
    text: str

# Root endpoint for status check
@app.get("/")
def read_root():
    return {"message": "🚨 Disaster Urgency Classification API is running! Use POST /predict to classify tweets."}

# Prediction endpoint
@app.post("/predict")
def predict(tweet: Tweet):
    # Tokenize the input text
    inputs = tokenizer(tweet.text, padding=True, truncation=True, return_tensors="pt")
    
    # Get model prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    predicted_class = torch.argmax(logits, dim=-1).item()

    # Map prediction to label
    label = "Urgent" if predicted_class == 1 else "Not urgent"
    
    # Return as JSON
    return {
        "text": tweet.text,
        "prediction": label
    }