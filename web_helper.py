import torch
import torch.nn.functional as F
from pytorch_transformers import BertTokenizer, BertForSequenceClassification
import os
from pathlib import Path
import re

def load_model(model, clf_type, device, mode='best'):
    if os.path.exists(Path('pretrained_models') / f'{type(model).__name__}_{clf_type}_{mode}.ckpt'):
        model.load_state_dict(torch.load(Path('pretrained_models') / f'{type(model).__name__}_{clf_type}_{mode}.ckpt', map_location=device))

class WebHelper:
    def __init__(self):
        self.max_len = 64
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.bert_category_clf = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=8).to(self.device)
        self.bert_category_clf.eval()
        load_model(self.bert_category_clf, 'Category', self.device)
        # self.bert_fluctuation_clf = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=4).to(self.device)
        # self.bert_fluctuation_clf.eval()
        # load_model(self.bert_fluctuation_clf, 'Fluctuation', self.device)

    def filter_by_category(self, company_name, headline):
        if company_name in headline :
            processed_headline = re.sub(company_name, '', str(headline.encode('utf-8')))
            tmp = [processed_headline]
            encoded_list = [self.tokenizer.encode(t, add_special_tokens=True) for t in tmp]
            padded_list =  [e + [0] * (512 - len(e)) for e in encoded_list]
            sample = torch.tensor(padded_list)

            labels = torch.tensor([1]).unsqueeze(0)
            sample = sample.to(self.device)
            labels = labels.to(self.device)
            _, logits = self.bert_category_clf(sample, labels=labels)

            pred = torch.argmax(F.softmax(logits), dim=1).item()
            cate = ["정치","경제","사회", "생활/문화","세계","기술/IT", "연예", "스포츠"]
            print(f'{company_name} :: {headline} :: {cate[pred]}')

            if pred == 1 or pred == 5 :
                return True
        return False