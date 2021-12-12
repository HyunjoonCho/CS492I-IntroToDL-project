import torch
from pytorch_transformers import BertTokenizer, BertForSequenceClassification
import os
from pathlib import Path
import re
import FinanceDataReader as fdr
from torch import nn
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler

def load_model(model, clf_type, device, mode='best'):
    if os.path.exists(Path('pretrained_models') / f'{type(model).__name__}_{clf_type}_{mode}.ckpt'):
        model.load_state_dict(torch.load(Path('pretrained_models') / f'{type(model).__name__}_{clf_type}_{mode}.ckpt', map_location=device))

def split_data(stock):
    data_raw = stock.to_numpy()
    data = []
    lookback = 21
    
    for index in range(len(data_raw) - lookback): 
        data.append(data_raw[index: index + lookback])
    
    data = np.array(data)
    val_and_test_set_size = int(np.round(0.125 * data.shape[0]))
    train_set_size = data.shape[0] - 2 * (val_and_test_set_size)

    x_train = data[:train_set_size, :-1, :]
    y_train = data[:train_set_size, -1, :]

    x_val = data[train_set_size:train_set_size + val_and_test_set_size, :-1, :]
    y_val = data[train_set_size:train_set_size + val_and_test_set_size, -1, :]
    
    x_test = data[train_set_size + val_and_test_set_size: , :-1, :]
    y_test = data[train_set_size + val_and_test_set_size:, -1, :]
    
    return list(map(lambda x : torch.from_numpy(x).type(torch.Tensor), [x_train, y_train, x_val, y_val, x_test, y_test]))

def classify_fluctuation(fluctuation):
    if fluctuation < -2.5:
        return 0
    elif fluctuation < 0:
        return 1
    elif fluctuation < 2.5:
        return 2
    else:
        return 3

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out

def train_lstm(dataset, device):
    x_train, y_train, x_val, y_val, x_test, y_test = list(map(lambda x : x.to(device), dataset))

    num_epochs = 100
    model = LSTM(input_dim=1, hidden_dim=32, output_dim=1, num_layers=2)
    model.to(device)
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    start_time = time.time()

    for t in range(num_epochs):
        model.train()
        y_train_pred = model(x_train)
        loss = criterion(y_train_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (t + 1) % 10 == 0:
            print(f'[Epoch {t + 1}/{num_epochs}] -> Train Loss: {loss.item():.4f}')

        with torch.no_grad():
            model.eval()
            y_val_pred = model(x_val)
            loss = criterion(y_val_pred, y_val)
            if (t + 1) % 10 == 0:
                print(f'[Epoch {t + 1}/{num_epochs}] -> Val Loss: {loss.item():.4f}') 

    training_time = time.time() - start_time
    print("Training time: {}".format(training_time))

    with torch.no_grad():
        model.eval()
        y_test_pred = model(x_test)
        loss = criterion(y_test_pred, y_test)
        print(f'Test Loss: {loss.item():.4f}') 

    return model

class WebHelper:
    def __init__(self):
        self.max_len = 64
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.bert_category_clf = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=8).to(self.device)
        self.bert_category_clf.eval()
        load_model(self.bert_category_clf, 'Category', self.device)
        self.bert_fluctuation_clf = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=4).to(self.device)
        self.bert_fluctuation_clf.eval()
        load_model(self.bert_fluctuation_clf, 'Fluctuation', self.device)
    
    def _apply_model(self, seq, model):
        tmp = [seq]
        encoded_list = [self.tokenizer.encode(t,add_special_tokens=True) for t in tmp]
        padded_list =  [e + [0] * (512-len(e)) for e in encoded_list]
        sample = torch.tensor(padded_list)

        labels = torch.tensor([1]).unsqueeze(0)
        sample = sample.to(self.device)
        labels = labels.to(self.device)
        _, logits = model(sample, labels=labels)

        pred = torch.argmax(logits.softmax(dim=1), dim=1)
        confidence = logits.softmax(dim=1).max().item() * 100

        return pred, confidence

    def filter_by_category(self, company_name, headline):
        if company_name in headline :
            processed_headline = re.sub(company_name, '', str(headline.encode('utf-8')))
            pred, confidence = self._apply_model(processed_headline, self.bert_category_clf)
            cate = ["정치","경제","사회", "생활/문화","세계","기술/IT", "연예", "스포츠"]
            print(f'[Headline] {headline} [Category] {cate[pred]} [Confidence] {confidence:.2f}')

            if pred == 1 or pred == 5 :
                return True
        return False

    def predict_fluctuation_by_news(self, aggregated_titles):
        return self._apply_model(aggregated_titles[:64], self.bert_fluctuation_clf)
    
    def predict_fluctuation_by_chart(self, ticker, past_20d_data):
        last_price = past_20d_data['Close'].values[-1]

        total_data = fdr.DataReader(ticker)[['Close']]
        scaler = MinMaxScaler(feature_range=(-1, 1))
        total_data['Close'] = scaler.fit_transform(total_data['Close'].values.reshape(-1, 1))
        lstm_price_predictor = train_lstm(split_data(total_data), self.device)

        past_20d_data['Close'] = scaler.fit_transform(past_20d_data['Close'].values.reshape(-1, 1))
        processed_20d_data = torch.from_numpy(past_20d_data.to_numpy()).type(torch.Tensor).unsqueeze(0)
        predicted_price = int(scaler.inverse_transform(lstm_price_predictor(processed_20d_data).detach().numpy()).item())
        fluctuation = (predicted_price - last_price) / last_price * 100

        print(f'Last Trading Day Price {last_price}, Predicted Price {predicted_price}, Fluctuation {fluctuation}')

        return classify_fluctuation(fluctuation), predicted_price