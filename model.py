import argparse
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel
from tqdm import tqdm


def load_dataset(path):
    indices, texts = [], []
    for file in Path(path).iterdir():
        if file.suffix == '.txt':
            indices.append(file.stem)
            texts.append(file.read_text(encoding='utf-8'))
    return pd.DataFrame({'index': indices, 'text': texts})


class TextDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512):
        self.indices = df['index'].tolist()
        self.texts = df['text'].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'index': self.indices[idx],
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }


class BERTClassifier(nn.Module):
    def __init__(self, hidden_dim=384, output_dim=1, dropout_p=0.2):
        super().__init__()
        self.bert = AutoModel.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

        for param in self.bert.parameters():
            param.requires_grad = False
        for layer in self.bert.encoder.layer[-3:]: # 2 -> 3
            for param in layer.parameters():
                param.requires_grad = True

        embedding_dim = self.bert.config.hidden_size  
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:,0,:]
        return self.classifier(pooled).squeeze(-1)


def collate_fn(batch):
    indices = [item['index'] for item in batch]
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    return indices, input_ids, attention_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('model_name')
    parser.add_argument('--output', '-o', default='result.tsv')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='Batch size for prediction')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.model_name, map_location=device)
    model = checkpoint['model']
    tokenizer = checkpoint['tokenizer']

    df = load_dataset(args.data_dir)
    dataset = TextDataset(df, tokenizer, max_len=512)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    model.to(device)
    model.eval()

    results = []
    sigmoid = nn.Sigmoid()

    for indices, input_ids, attention_mask in tqdm(loader, desc='Predicting'):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = sigmoid(outputs).cpu()
        preds = (probs >= 0.5).long().tolist()
        for idx, pred in zip(indices, preds):
            label = 'pos' if pred == 1 else 'neg'
            results.append({'index': idx, 'label': label})

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output, sep='\t', index=False)
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
