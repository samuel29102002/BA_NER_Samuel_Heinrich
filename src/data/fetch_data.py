from datasets import load_dataset

def download_datasets():
    # CoNLL-2003 für Englisch
    dataset_en = load_dataset("conll2003")
    # CoNLL-2002 für Spanisch
    dataset_es = load_dataset("conll2002", "es")
    # Fiktiver Platzhalter für den afrikanischen NER-Datensatz
    # Für die tatsächliche Implementierung müsstest du den Pfad anpassen,
    # um deinen spezifischen Low-Resource-Datensatz zu laden
    dataset_lr = load_dataset('masakhane/masakhaner2', 'zul')

    # Speichern der Datensätze für spätere Verarbeitung
    dataset_en.save_to_disk("./data/raw/conll2003_en")
    dataset_es.save_to_disk("./data/raw/conll2002_es")
    dataset_lr.save_to_disk("./data/raw/african_ner")

if __name__ == "__main__":
    download_datasets()
