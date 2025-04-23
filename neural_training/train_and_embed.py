import os
import torch
from torch.xpu import device
from tqdm import tqdm
from itertools import batched

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def get_tr_outdir(args):
    model_name = args.model_name.split("/")[-1]
    dataset_name = args.dataset_name.split("/")[-1]
    outdir = os.path.join("models", dataset_name, model_name)
    os.makedirs(outdir, exist_ok=True)
    return outdir


def get_embed_outdir(args):
    model_name = args.model_name.split("/")[-1]
    dataset_name = args.dataset_name.split("/")[-1]
    outdir = os.path.join("embeds", dataset_name, model_name)
    os.makedirs(outdir, exist_ok=True)
    return outdir


def compute_clf_metrics(preds):
    _preds = preds.predictions.argmax(axis=1)
    _labels = preds.label_ids
    acc = accuracy_score(y_true=_labels, y_pred=_preds)
    recall = recall_score(y_true=_labels, y_pred=_preds)
    precision = precision_score(y_true=_labels, y_pred=_preds)
    f1 = f1_score(y_true=_labels, y_pred=_preds, average="micro")
    return {"acc": acc, "recall": recall, "precision": precision, "f1": f1}


def get_classifier(model_name, n_classes=2, use_bfloat16=False, device="cuda"):
    torch_dtype = torch.bfloat16 if use_bfloat16 else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=n_classes,
                                                               torch_dtype=torch_dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def get_training_args(args):
    training_outdir = get_tr_outdir(args)
    trainer_args = TrainingArguments(
        output_dir=training_outdir,
        do_train=True,
        learning_rate=args.lr,
        num_train_epochs=args.nepochs,
        per_device_train_batch_size=args.train_batchsize,
        per_device_eval_batch_size=args.train_batchsize * 4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        eval_strategy="steps",
        eval_steps=500,
        logging_steps=50,
        bf16=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        eval_on_start=True,
        load_best_model_at_end=True,
    )

    return trainer_args


def get_cls_bertlike(x):
    # get representation associated with the "CLS" token. In BERT-like models this is usually assigned with the first token of the input sequence (idx=0)
    cls_emebds = x[:, 0, :]
    return cls_emebds


def embed(model, tokenizer, data, selection_strategy, args):
    split_logits = []
    split_hidden_states = []
    split_labels = []
    for batch in batched(tqdm(data), n=args.embed_batchsize):
        texts, labels = zip(*((d["text"], d["label"]) for d in batch))
        with torch.no_grad():
            model_inputs = tokenizer(texts, truncation=True, max_length=args.max_length, padding="max_length",
                                     return_tensors="pt")  # pad each batch to max_length
        output = model(**model_inputs.to(args.device), output_hidden_states=True)
        logits = output.logits
        hidden_states = output.hidden_states
        last_hidden_states = hidden_states[-1]

        split_hidden_states.append(selection_strategy(last_hidden_states.cpu().detach()))
        split_logits.append(logits.cpu().detach())
        split_labels.append(torch.tensor(labels, device='cpu'))

    split_logits = torch.vstack(split_logits)
    split_hidden_states = torch.vstack(split_hidden_states)
    split_labels = torch.concat(split_labels)

    return split_logits, split_hidden_states, split_labels


def prepare_dataset(dataset_name):
    dataset_name_map = {
        'imdb': "stanfordnlp/imdb",
        'yelp': "Yelp/yelp_review_full",
        'ag_news': "fancyzhx/ag_news",
        # 'sst2': "stanfordnlp/sst2"
        'rt': "cornell-movie-review-data/rotten_tomatoes"
    }
    dataset_fullname = dataset_name_map.get(dataset_name, None)
    if dataset_fullname is None:
        raise ValueError(f'unrecognized dataset name {dataset_name}; valid ones are {dataset_name_map.keys()}')

    dataset = load_dataset(dataset_fullname)

    if dataset_name == 'yelp':
        def binarize_example(example):
            if example["label"] in [0, 1]:  # negative
                return {"text": example["text"], "label": 0}
            elif example["label"] in [3, 4]:  # positive
                return {"text": example["text"], "label": 1}
            else:
                return None # filter out 3 stars

        dataset = dataset.map(binarize_example).filter(lambda x: x is not None)

    if dataset_name == 'ag_news':
        def binarize_example(example):
            # "World" vs the rest ("Sports", "Business", "Sci/tec")
            example["label"] = 0 if example["label"] == 0 else 1
            return example

        dataset = dataset.map(binarize_example).filter(lambda x: x is not None)

    return dataset


def extract_tensors(model, tokenizer, dataset, split, prefix, outdir):
    split_data = dataset[split]
    split_logits, split_last_hiddens, split_labels = (
        embed(model, tokenizer, data=split_data, selection_strategy=get_cls_bertlike, args=args)
    )

    torch.save(split_logits, os.path.join(outdir, f"{prefix}.{split}.logits.pt"))
    torch.save(split_last_hiddens, os.path.join(outdir, f"{prefix}.{split}.hidden_states.pt"))
    torch.save(split_labels, os.path.join(outdir, f"{prefix}.{split}.labels.pt"))


def set_learnable_parameters(model, train_backbone):
    # freeze base model -> train only fresh init classification head
    if not train_backbone:
        print("- freezing base model weights")
        for _, layer_weights in model.base_model.named_parameters():
            layer_weights.requires_grad = False

        # check trainable layers
        trainable_layers = []
        for layer_name, layer_weights in model.named_parameters():
            if layer_weights.requires_grad == True:
                trainable_layers.append(layer_name)
        print(f"- trainable layers: {trainable_layers}")


def main(args):
    print(f"- model: {args.model_name}")
    print(f"- dataset: {args.dataset_name}")

    dataset = prepare_dataset(args.dataset_name)

    # create validation split if does not exist
    if "validation" not in dataset:
        print("splitting training set into train/validation...")
        _tmp_dataset = dataset["train"].train_test_split(test_size=args.val_size)
        dataset["train"] = _tmp_dataset["train"]
        dataset["validation"] = _tmp_dataset["test"]

    n_inferred_classes = dataset["train"].shape[-1]
    if args.num_classes != n_inferred_classes:
        print(f"number of inferred target classes ({n_inferred_classes}) != number of given target classes "
              f"({args.num_classes})")

    model, tokenizer = get_classifier(model_name=args.model_name, n_classes=args.num_classes, device=args.device)

    set_learnable_parameters(model, train_backbone=args.train_backbone)

    def tokenize_dataset(sample):
        return tokenizer(sample["text"], truncation=True, max_length=args.max_length, padding="max_length",
                         return_tensors="pt")

    # tokenize dataset
    dataset = dataset.map(tokenize_dataset, batched=True, num_proc=16)

    # train model
    trainer_args = get_training_args(args)
    print(f"- storing model in {trainer_args.output_dir}")
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=trainer_args,
        compute_metrics=compute_clf_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],  # stop if it does not improve after 5 validation steps
    )
    print("\nTraining...")
    trainer.train()

    # Get embedddings and logits
    embeds_outdir = get_embed_outdir(args)
    print("\nEmbedding...")
    print(f"- storing embeddings in {embeds_outdir}")
    splits = ["train", "validation", "test"]
    for split in splits:
        extract_tensors(model, tokenizer, dataset, split=split, prefix='source', outdir=embeds_outdir)

    sentiment_datasets = ['imdb', 'yelp', 'rt']
    if args.dataset_name in sentiment_datasets:
        target_dataset_names = [dataname for dataname in sentiment_datasets if dataname!=args.dataset_name]
        for target_dataset_name in target_dataset_names:
            target_dataset = prepare_dataset(target_dataset_name)
            extract_tensors(model, tokenizer, target_dataset, split='test', prefix=f'target_{target_dataset_name}', outdir=embeds_outdir)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google-bert/bert-base-uncased")
    parser.add_argument("--dataset_name", type=str, default="yelp")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--nepochs", type=int, default=5)
    parser.add_argument("--train_batchsize", type=int, default=64)
    parser.add_argument("--embed_batchsize", type=int, default=512)
    parser.add_argument("--val_size", type=int, default=5_000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--train_backbone", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    main(args)

    # model_names:
    #   - "google-bert/bert-base-uncased"
    #   - "FacebookAI/roberta-base"
    #   - "distilbert/distilbert-base-uncased"

    # datasets:
    #   - "imdb" ("stanfordnlp/imdb", binary)
    #   - "yelp" ("Yelp/yelp_review_full", 5 stars rating, {1,2}->0, {4,5}->1)
    #   - "ag_news" ("fancyzhx/ag_news", 4 classes (0='world',1='sports',2='business',3='sci/tec'), {'world'}->0, {'sports', 'business', 'sci/tec'}->1) <-- only 1 epoch!
    #   - "sst2" ("stanfordnlp/sst2") <-- not used, too big and similar to rotten tomatoes
    #   - "rt" ("cornell-movie-review-data/rotten_tomatoes")