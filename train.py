import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import logging
from datetime import datetime
from arch import MedicalClassifier, LABEL_SMOOTHING
from data import CustomDataset, collate_fn

model_name = "intfloat/multilingual-e5-large"
EVAL_SIZE = 2
BATCH_SIZE = 16 
GRAD_ACCUM_STEPS = 4
EPOCHS = 2 
LR = 2e-5
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
SEED = 0
MAX_LENGTH = 512
NUM_DROPOUTS = 5
DROPOUT_RATE = 0.1
LLRD_DECAY = 0.95 
NUM_WORKERS = 4
USE_AMP = False
LOG_FILE = f"train_single_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@torch.no_grad()
def evaluate(model, eval_loader):
    model.eval()
    all_preds = []
    all_labels = []

    for batch in tqdm(eval_loader, desc="evaluating"):
        batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
        labels = batch.pop('labels')

        with autocast('cuda', enabled=USE_AMP):
            outputs = model(**batch)
        preds = torch.argmax(outputs['logits'], dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")

    return {"accuracy": acc, "f1_macro": f1_macro}


if __name__ == "__main__":
    logger.info(f"config: model={model_name}, batch={BATCH_SIZE}x{GRAD_ACCUM_STEPS}, lr={LR}, wd={WEIGHT_DECAY}")
    logger.info(f"multi-sample dropout: {NUM_DROPOUTS}x, rate={DROPOUT_RATE}, label_smoothing={LABEL_SMOOTHING}")
    logger.info(f"log file: {LOG_FILE}")

    logger.info(f"loading data")
    df = pd.read_csv("ground_truth.csv")
    # df2 = pd.read_csv("kfupm-processed.csv")
    df.drop('id', inplace=True, axis=1)
    # df = pd.concat([df1, df2])
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)
    df['label'] = df['label'].map({'human': 0, 'machine': 1})
    num_labels = 2
    logger.info(f"total samples: {len(df)} | num labels: {num_labels}")
    logger.info(df.head())
    # stratified split
    train_df, eval_df = train_test_split(
        df,
        test_size=EVAL_SIZE,
        stratify=df["label"],
        random_state=SEED
    )
    train_df = train_df.reset_index(drop=True)
    eval_df = eval_df.reset_index(drop=True)

    logger.info(f"train: {len(train_df)} | eval: {len(eval_df)}")
    logger.info(f"train labels: {train_df['label'].value_counts()} | eval labels: {eval_df['label'].value_counts()}")

    # datasets
    train_ds = CustomDataset(
        model_name,
        train_df['content'].tolist(),
        train_df['label'].tolist(),
        max_length=MAX_LENGTH
    )
    eval_ds = CustomDataset(
        model_name,
        eval_df['content'].tolist(),
        eval_df['label'].tolist(),
        max_length=MAX_LENGTH
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    # model
    model = MedicalClassifier(
        model_name,
        num_labels,
        num_dropouts=NUM_DROPOUTS,
        dropout_rate=DROPOUT_RATE
    ).cuda()

    # optimizer with layer-wise learning rate decay (LLRD)
    def get_optimizer_params(model, lr, weight_decay, llrd_decay):
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        optimizer_params = []
        
        # encoder layers with LLRD
        num_layers = model.encoder.config.num_hidden_layers
        for layer_idx in range(num_layers):
            layer_lr = lr * (llrd_decay ** (num_layers - layer_idx - 1))
            layer_params = [
                (n, p) for n, p in model.encoder.named_parameters()
                if f'layer.{layer_idx}.' in n
            ]
            optimizer_params.append({
                'params': [p for n, p in layer_params if not any(nd in n for nd in no_decay)],
                'lr': layer_lr,
                'weight_decay': weight_decay
            })
            optimizer_params.append({
                'params': [p for n, p in layer_params if any(nd in n for nd in no_decay)],
                'lr': layer_lr,
                'weight_decay': 0.0
            })
        
        # embeddings (lowest LR)
        embed_lr = lr * (llrd_decay ** num_layers)
        embed_params = [(n, p) for n, p in model.encoder.named_parameters() if 'embeddings' in n]
        optimizer_params.append({
            'params': [p for n, p in embed_params if not any(nd in n for nd in no_decay)],
            'lr': embed_lr,
            'weight_decay': weight_decay
        })
        optimizer_params.append({
            'params': [p for n, p in embed_params if any(nd in n for nd in no_decay)],
            'lr': embed_lr,
            'weight_decay': 0.0
        })
        
        # classification head (highest LR)
        head_params = [
            (n, p) for n, p in model.named_parameters()
            if not n.startswith('encoder.')
        ]
        optimizer_params.append({
            'params': [p for n, p in head_params if not any(nd in n for nd in no_decay)],
            'lr': lr,
            'weight_decay': weight_decay
        })
        optimizer_params.append({
            'params': [p for n, p in head_params if any(nd in n for nd in no_decay)],
            'lr': lr,
            'weight_decay': 0.0
        })
        
        return [g for g in optimizer_params if len(g['params']) > 0]

    optimizer_params = get_optimizer_params(model, LR, WEIGHT_DECAY, LLRD_DECAY)
    optimizer = torch.optim.AdamW(optimizer_params, lr=LR)

    total_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * EPOCHS
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    scaler = GradScaler('cuda', enabled=USE_AMP)

    logger.info(f"total steps: {total_steps} | warmup: {warmup_steps} | grad_accum: {GRAD_ACCUM_STEPS}")
    logger.info(f"effective batch size: {BATCH_SIZE * GRAD_ACCUM_STEPS}")
    logger.info(f"AMP: {USE_AMP} | LLRD decay: {LLRD_DECAY}")
    logger.info("=" * 50)

    best_f1 = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        optimizer.zero_grad(set_to_none=True)  # zero at epoch start
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{EPOCHS}")

        for step, batch in enumerate(pbar):
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            labels = batch.pop('labels')

            # mixed precision forward
            with autocast('cuda', enabled=USE_AMP):
                outputs = model(**batch, labels=labels)
                loss = outputs['loss'] / GRAD_ACCUM_STEPS

            # scaled backward
            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                # unscale for gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * GRAD_ACCUM_STEPS
            pbar.set_postfix({"loss": f"{loss.item() * GRAD_ACCUM_STEPS:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

        # handle remaining gradients at epoch end
        if (step + 1) % GRAD_ACCUM_STEPS != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        avg_loss = total_loss / len(train_loader)

        logger.info(f"evaluating...")
        metrics = evaluate(model, eval_loader)

        logger.info(
            f"epoch {epoch+1}/{EPOCHS} | "
            f"loss: {avg_loss:.4f} | "
            f"accuracy: {metrics['accuracy']:.4f} | "
            f"f1: {metrics['f1_macro']:.4f}"
        )

        best_f1 = metrics["f1_macro"]
        torch.save(model.state_dict(), "e5_best.pt")
        logger.info(f"saved checkpoint.")

        logger.info("-" * 50)

    logger.info(f"training complete. best f1: {best_f1:.4f}")

