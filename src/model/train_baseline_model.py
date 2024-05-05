from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, load_metric

import torch
if torch.cuda.is_available():
    print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Training on CPU")

def load_and_prepare_data(dataset_path_en, dataset_path_es, dataset_path_lr, tokenizer):
    dataset_en = load_dataset("conll2003")
    dataset_es = load_dataset("conll2002", "es")
    dataset_lr = load_dataset('masakhane/masakhaner2', 'zul')

    def tokenize_and_align_labels(examples):
    # Tokenisiere die Eingabe
      tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, padding="max_length", max_length=512)

      # Initialisiere das 'labels'-Feld in 'tokenized_inputs' als eine leere Liste
      tokenized_inputs["labels"] = []

      for i, label in enumerate(examples["ner_tags"]):
          word_ids = tokenized_inputs.word_ids(batch_index=i)  # Holt die Wort-IDs für das aktuelle Beispiel
          label_ids = []
          previous_word_idx = None
          for word_idx in word_ids:
              # Setze Labels für Sonder-Token auf -100, damit sie im Training ignoriert werden
              if word_idx is None or word_idx == previous_word_idx:
                  label_ids.append(-100)
              else:
                  label_ids.append(label[word_idx])
              previous_word_idx = word_idx
          # Füge die generierten 'label_ids' der 'labels'-Liste in 'tokenized_inputs' hinzu
          tokenized_inputs["labels"].append(label_ids)

      return tokenized_inputs


    # Wende die Tokenisierung und Label-Anpassung auf die Datensätze an
    tokenized_dataset_en = dataset_en.map(tokenize_and_align_labels, batched=True)
    tokenized_dataset_es = dataset_es.map(tokenize_and_align_labels, batched=True)
    tokenized_dataset_lr = dataset_lr.map(tokenize_and_align_labels, batched=True)


    return tokenized_dataset_en, tokenized_dataset_es, tokenized_dataset_lr

# Initialisiere das Tokenizer- und Modell
model_checkpoint = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=9)

# Lade und bereite Datensätze vor
dataset_path_en = "./data/raw/conll2003_en"
dataset_path_es = "./data/raw/conll2002_es"
dataset_path_lr = "./data/raw/african_ner"
tokenized_dataset_en, tokenized_dataset_es, tokenized_dataset_lr = load_and_prepare_data(dataset_path_en, dataset_path_es, dataset_path_lr, tokenizer)

# Definiere TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    gradient_accumulation_steps=2,
)

# Trainer-Instanz
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset_en["train"],
    eval_dataset=tokenized_dataset_en["validation"],
    tokenizer=tokenizer,
)

# Starte das Training
trainer.train()
