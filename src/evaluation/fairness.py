import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate

def load_and_prepare_data(tokenizer):
    # Load validation splits of datasets for evaluation
    dataset_en = load_dataset("conll2003", split='validation')
    dataset_es = load_dataset("conll2002", "es", split='validation')
    dataset_lr = load_dataset('masakhane/masakhaner2', 'zul', split='validation')

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, padding="max_length", max_length=512)
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = [-100 if word_id is None else label[word_id] for word_id in word_ids]
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_dataset_en = dataset_en.map(tokenize_and_align_labels, batched=True)
    tokenized_dataset_es = dataset_es.map(tokenize_and_align_labels, batched=True)
    tokenized_dataset_lr = dataset_lr.map(tokenize_and_align_labels, batched=True)

    return {
        "validation_en": tokenized_dataset_en,
        "validation_es": tokenized_dataset_es,
        "validation_lr": tokenized_dataset_lr
    }

# Load tokenizer and model with pretrained weights from baseline training
model_checkpoint = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained("./results/checkpoint-2500", num_labels=9)

# Prepare datasets
datasets = load_and_prepare_data(tokenizer)

# Define evaluation arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_eval_batch_size=8,
    report_to="none"  # Prevents connection attempts to WandB or other platforms
)

# Load the metric
seqeval_metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Flatten lists for error analysis
    flat_predictions = [p for sublist in predictions for p in sublist if p != -100]
    flat_labels = [l for sublist in labels for l in sublist if l != -100]

    true_predictions = [
        [model.config.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [model.config.id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = seqeval_metric.compute(predictions=true_predictions, references=true_labels)
    error_results = error_analysis(flat_predictions, flat_labels)

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
        **error_results
    }

def error_analysis(predictions, labels):
    true_positives = sum(1 for i in range(len(labels)) if labels[i] == predictions[i] and labels[i] != 'O')
    false_positives = sum(1 for i in range(len(labels)) if labels[i] != predictions[i] and predictions[i] != 'O')
    false_negatives = sum(1 for i in range(len(labels)) if labels[i] != predictions[i] and labels[i] != 'O')

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }

# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics
)

# Perform evaluation for each language and print results
eval_results_en = trainer.evaluate(datasets["validation_en"])
eval_results_es = trainer.evaluate(datasets["validation_es"])
eval_results_lr = trainer.evaluate(datasets["validation_lr"])

print("Evaluation results English:", eval_results_en)
print("Evaluation results Spanish:", eval_results_es)
print("Evaluation results Low-Resource Language:", eval_results_lr)

# Generate plots for visualization
metrics = ['precision', 'recall', 'f1', 'accuracy', 'true_positives', 'false_positives', 'false_negatives']
results = [eval_results_en, eval_results_es, eval_results_lr]
labels = ['English', 'Spanish', 'Low-Resource']

for metric in metrics:
    values = [result[f'eval_{metric}'] for result in results]
    plt.figure(figsize=(8, 4))
    plt.bar(labels, values, color=['blue', 'green', 'red'])
    plt.title(f'{metric.title()} Score Across Languages')
    plt.ylabel(f'{metric.title()} Score')
    plt.show()
