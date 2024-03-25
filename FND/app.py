from flask import Flask, render_template, request
import torch
import torch.nn as nn
from transformers import AutoModel, BertTokenizerFast

app = Flask(__name__)

# Define the BERT architecture
class BERTClassifier(nn.Module):
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        self.bert = bert 
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Function to load English model
def load_eng_model():
    bert_model = AutoModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = BERTClassifier(bert_model)
    model.load_state_dict(torch.load('engDetect.pt'))
    model.eval()
    return model, tokenizer

# Define preprocessing function
def preprocess_input(news_title, news_content, tokenizer):
    inputs = tokenizer(news_title, news_content, return_tensors="pt", padding=True, truncation=True)
    return inputs

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    news_title = request.form['news_title']
    news_content = request.form['news_content']
    
    # Load English model
    model, tokenizer = load_eng_model()
    
    # Preprocess input data
    inputs = preprocess_input(news_title, news_content, tokenizer)

    # Make prediction
    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
        predicted_class = torch.argmax(outputs, dim=1).item()
        detection_result = "Real" if predicted_class == 1 else "Fake"

    return render_template('index.html', detection_result=detection_result)

if __name__ == '__main__':
    app.run(debug=True)
