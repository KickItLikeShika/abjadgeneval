from tqdm import tqdm
import torch
import pandas as pd
from torch.amp import autocast
from torch.utils.data import DataLoader
from arch import MedicalClassifier
from data import CustomDataset, collate_fn
# 
model_name = "intfloat/multilingual-e5-large"
num_classes = 2
num_dropouts = 5
dropout_rate = 0.1
max_length = 512
batch_size = 16  # reduced for larger model

@torch.no_grad()
def predict(model, eval_loader):
    model.eval()
    all_preds = []
    all_labels = []

    for batch in tqdm(eval_loader, desc="evaluating"):
        batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
        # labels = batch.pop('labels')

        with autocast('cuda', enabled=False):
            outputs = model(**batch)
        preds = torch.argmax(outputs['logits'], dim=1)

        all_preds.extend(preds.cpu().tolist())
        # all_labels.extend(labels.cpu().tolist())

    return all_preds


# df = pd.read_csv("ground_truth.csv")
df = pd.read_csv("final_test_unlabeled.csv")
ds = CustomDataset(model_name, df['content'].tolist(), max_length=max_length)
dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
model = MedicalClassifier(model_name, num_classes, num_dropouts, dropout_rate)
model.load_state_dict(torch.load('arabert_best.pt'))
model.cuda()
preds = predict(model, dl)
df['label'] = preds
df['label'] = df['label'].map({0: 'human', 1: 'machine'})
df.drop(['id', 'content'], inplace=True, axis=1)
df.to_csv('predictions.csv', index=False)
# print(preds)