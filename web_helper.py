import torch
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

    def predict_fluctuation(self, aggregated_titles):
        return self._apply_model(aggregated_titles[:64], self.bert_fluctuation_clf)