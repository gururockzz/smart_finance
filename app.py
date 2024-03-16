from flask import Flask, request, jsonify
import random
import json
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import nltk
import os
# Set NLTK data path to the directory in your Git repo
nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
nltk.data.path.append(nltk_data_path)

# Download 'punkt' tokenizer data if not already downloaded
nltk.download('punkt')
# Define your LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Define your Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, d_model, num_heads, num_encoder_layers):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(input_size, d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=num_heads, num_encoder_layers=num_encoder_layers)
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.embedding(x)
        out = self.transformer(x)
        out = self.fc(out[-1, :, :])  # Take the last output from the transformer
        return out

# Define your hybrid chatbot model
class HybridChatbot(nn.Module):
    def __init__(self, lstm_input_size, lstm_hidden_size, transformer_model):
        super(HybridChatbot, self).__init__()
        self.lstm = LSTMModel(lstm_input_size, lstm_hidden_size, lstm_hidden_size)
        self.transformer = transformer_model

    def forward(self, input_sequence):
        lstm_output = self.lstm(input_sequence)
        transformer_output = self.transformer(input_sequence)
        
        # Combine the outputs of LSTM and Transformer (you can customize how to combine)
        combined_output = lstm_output + transformer_output
        
        return combined_output
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
import numpy as np
import nltk
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


def tokenize(sentence):
    """
    split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    """
    return nltk.word_tokenize(sentence)


def stem(word):
    """
    stemming = find the root form of the word
    examples:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag
import requests

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model and data
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Greek"


def get_response(msg):

    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.80:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])

    return "I do not understand..."


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_input = data['user_input']

        # Get translated response
        response = get_response(user_input)

        return jsonify({'response': response})

    except Exception as e:
        print(e)
        return jsonify({'error': 'Invalid input'})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
