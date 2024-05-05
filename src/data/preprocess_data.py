import spacy
from datasets import load_from_disk

def preprocess_dataset(dataset_path, lang_model=None):
    # Lade das Spacy-Modell, wenn eines angegeben ist
    nlp = spacy.load(lang_model) if lang_model else None

    # Lade den Datensatz
    dataset = load_from_disk(dataset_path)

    def preprocess(batch):
        if nlp:
            # Verarbeite die Texte mit Spacy, wenn ein Modell geladen ist
            texts = [" ".join(tokens) if isinstance(tokens, list) else tokens for tokens in batch["tokens"]]
            docs = list(nlp.pipe(texts))
            processed_texts = [
                " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])
                for doc in docs
            ]
        else:
            # Führe eine einfachere Vorverarbeitung durch, wenn kein Spacy-Modell vorhanden ist
            processed_texts = [
                " ".join(tokens) if isinstance(tokens, list) else tokens
                for tokens in batch["tokens"]
            ]

        return {"tokens": processed_texts}

    # Wende die Preprocessing-Funktion auf den Datensatz an, mit Batch-Verarbeitung
    processed_dataset = dataset.map(preprocess, batched=True, batch_size=1000)

    return processed_dataset

if __name__ == "__main__":
    # Pfad zu den rohen Datensätzen
    dataset_path_en = "./data/raw/conll2003_en"
    dataset_path_es = "./data/raw/conll2002_es"
    dataset_path_lr = "./data/raw/african_ner"

    # Verarbeite die Datensätze
    processed_dataset_en = preprocess_dataset(dataset_path_en, "en_core_web_sm")
    processed_dataset_es = preprocess_dataset(dataset_path_es, "es_core_news_sm")
    processed_dataset_lr = preprocess_dataset(dataset_path_lr)

    # Speichere die verarbeiteten Datensätze
    processed_dataset_en.save_to_disk("./data/processed/conll2003_en_processed")
    processed_dataset_es.save_to_disk("./data/processed/conll2002_es_processed")
    processed_dataset_lr.save_to_disk("./data/processed/african_ner_processed")
