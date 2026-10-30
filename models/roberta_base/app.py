from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# Initialize FastAPI app
app = FastAPI()

# Check if GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the RoBERTa model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained('./roberta_model')
model = RobertaForSequenceClassification.from_pretrained('./roberta_model', num_labels=2)

# Move model to the appropriate device (GPU/CPU)
model.to(device)

# Define input data model
class TextInput(BaseModel):
    text: str

# Function to predict urgency based on input text
def predict_urgency(text: str):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Move inputs to the same device as the model
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Make the prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Convert logits to predicted labels
    predictions = torch.argmax(logits, dim=-1)

    # Map the predicted label to its meaning (0 = Not Urgent, 1 = Urgent)
    label_mapping = {0: "Not Urgent", 1: "Urgent"}
    predicted_label = label_mapping[predictions.item()]

    return predicted_label

# Define the API endpoint for urgency prediction
@app.post("/predict/")
async def predict_urgency_endpoint(text_input: TextInput):
    # Get the urgency prediction
    prediction = predict_urgency(text_input.text)
    return {"prediction": prediction}

# Root endpoint for status check
@app.get("/")
def read_root():
    return {"message": "🚨 Disaster Urgency Classification API is running! Use POST /predict to classify tweets."}