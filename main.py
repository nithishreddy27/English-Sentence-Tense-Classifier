from fastapi import FastAPI, HTTPException
from transformers import AlbertTokenizer, AlbertModel
import torch
import torch.nn as nn

# Define the TenseClassifier class
class TenseClassifier(nn.Module):
    def __init__(self, Albert_model, num_classes):
        super(TenseClassifier, self).__init__()
        self.Albert = Albert_model
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(self.Albert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.Albert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding
        x = self.relu1(pooled_output)
        x = self.relu2(x)
        logits = self.fc(x)
        return logits

# Initialize the FastAPI app
app = FastAPI()

# Load tokenizer and model once when the application starts
tokenizer = AlbertTokenizer.from_pretrained('./Albert')
Albert_model = AlbertModel.from_pretrained('./Albert')
device = 'cpu'  # Change to 'cuda' if you want to use GPU
model = TenseClassifier(Albert_model, num_classes=12).to(device)

# Load the classifier state_dict (your saved model weights)
model.load_state_dict(torch.load('./AlBert_tense_classifier_model.pth', map_location=device))
model.eval()

# Define the tense labels
tense_labels = {
    'present': 0,
    'future': 1,
    'past': 2,
    'present perfect continuous': 3,
    'future perfect': 4,
    'past perfect': 5,
    'future continuous': 6,
    'past perfect continuous': 7,
    'present continuous': 8,
    'past continuous': 9,
    'future perfect continuous': 10,
    'present perfect': 11,
}

@app.get("/")
def root_path():
    return {"message": "Hello World!!"}

@app.post("/predict")
def predict_tense(sentence: str):
    # Check if the sentence is valid
    if not sentence:
        raise HTTPException(status_code=400, detail="Sentence is required")

    try:
        # Tokenize the sentence
        encoded_sentence = tokenizer(sentence, padding=True, truncation=True, return_tensors="pt").to(device)

        # Use the model to predict the tense
        with torch.no_grad():
            logits = model(encoded_sentence['input_ids'], encoded_sentence['attention_mask'])
            predicted_label = torch.argmax(logits, dim=1).item()

        # Convert predicted label back to tense
        predicted_tense = [k for k, v in tense_labels.items() if v == predicted_label][0]

        return {"predicted_tense": predicted_tense}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
