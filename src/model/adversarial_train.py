import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict, concatenate_datasets
import textattack
from textattack.transformations import WordSwapEmbedding
from textattack.augmentation import Augmenter
from datasets import load_from_disk, DatasetDict, concatenate_datasets

# Pfad zu den vorverarbeiteten Datensätzen
dataset_path_en = "./data/raw/conll2003_en"
dataset_path_es = "./data/raw/conll2002_es"
dataset_path_lr = "./data/raw/african_ner"

# Initialisiere den Tokenizer und das Modell
model_checkpoint = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=9)

# Lade die vorverarbeiteten Datensätze
dataset_en = load_from_disk(dataset_path_en)
dataset_es = load_from_disk(dataset_path_es)
dataset_lr = load_from_disk(dataset_path_lr)


# Stelle sicher, dass dein Dataset das erwartete Format hat. Diese Funktion muss möglicherweise angepasst werden.
def check_and_prepare_data(dataset):
    if 'text' not in dataset.column_names:
        dataset = dataset.map(lambda examples: {'text': [' '.join(tokens) for tokens in examples['tokens']]}, batched=True)
    return dataset

dataset_lr = check_and_prepare_data(dataset_lr)

# Funktion zur Erzeugung von adversarial examples
def augment_data(examples):
    augmented_texts = [augmenter.augment(text)[0] for text in examples['text']]  # Augmenter liefert eine Liste von augmentierten Versionen pro Eingabe
    return {'text': augmented_texts}

# Verwende TextAttack, um adversarial examples zu erzeugen
transformation = WordSwapEmbedding(max_candidates=5)
augmenter = Augmenter(transformation=transformation, pct_words_to_swap=0.1, transformations_per_example=1)

# Erzeuge adversarial examples und füge sie zu den Trainingsdaten hinzu
adversarial_dataset_lr = dataset_lr.map(augment_data, batched=True)
combined_dataset = concatenate_datasets([dataset_en, dataset_es, adversarial_dataset_lr])

# Tokenisiere das kombinierte Dataset für das Training
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

tokenized_dataset = combined_dataset.map(tokenize_function, batched=True)

# Training Arguments definieren
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Trainer-Instanz erstellen
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
)

# Starte das Training
trainer.train()


# import torch
# from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer, TrainingArguments
# from datasets import load_dataset, DatasetDict, concatenate_datasets
# from textattack.transformations import WordSwapEmbedding
# from textattack.augmentation import Augmenter

# # Pfad zu den Rohdatensätzen
# dataset_path_en = "./data/raw/conll2003_en"
# dataset_path_es = "./data/raw/conll2002_es"

# # Initialisiere den Tokenizer und das Modell
# model_checkpoint = "bert-base-multilingual-cased"
# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
# model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=9)

# # Lade die Rohdatensätze
# dataset_en = load_dataset('conll2003', split='train')
# dataset_es = load_dataset('conll2002', 'es', split='train')

# # Tokenisiere das Dataset
# def tokenize_function(examples):
#     return tokenizer(examples['tokens'], truncation=True, padding="max_length", is_split_into_words=True)

# tokenized_dataset_en = dataset_en.map(tokenize_function, batched=True)
# tokenized_dataset_es = dataset_es.map(tokenize_function, batched=True)

# # Erzeuge adversarial examples für das spanische Dataset
# transformation = WordSwapEmbedding(max_candidates=5)
# augmenter = Augmenter(transformation=transformation, pct_words_to_swap=0.1, transformations_per_example=1)

# def augment_data(examples):
#     augmented_texts = []
#     for text in examples['text']:
#         # TextAttack könnte möglicherweise mehr als ein augmentiertes Beispiel zurückgeben
#         augmented_example = augmenter.augment(text)
#         augmented_texts.extend(augmented_example)
#     return {'text': augmented_texts}

# # Es ist wichtig, die augment_data Funktion nur auf dem spanischen Dataset anzuwenden
# augmented_dataset_es = tokenized_dataset_es.map(augment_data, batched=True)

# # Kombiniere die Datasets
# combined_dataset = concatenate_datasets([tokenized_dataset_en, augmented_dataset_es])

# # Training Arguments definieren
# training_args = TrainingArguments(
#     output_dir='./results',
#     num_train_epochs=3,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     warmup_steps=500,
#     weight_decay=0.01,
#     logging_dir='./logs',
#     logging_steps=10,
# )

# # Trainer-Instanz erstellen
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=combined_dataset,
#     # Es wird empfohlen, ein separates Validierungsset zu verwenden
#     eval_dataset=combined_dataset.select(range(10))  # Beispiel für die Auswahl eines Validierungssets
# )

# # Starte das Training
# trainer.train()
