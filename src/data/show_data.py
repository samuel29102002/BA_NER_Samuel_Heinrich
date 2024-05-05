
import pyarrow.parquet as pq

# Pfad zur Arrow-Datei
path_to_arrow_file = './data/raw/conll2002_es/validation/data-00000-of-00001.arrow'

# Laden der Daten aus der Arrow-Datei
data = pq.read_table(path_to_arrow_file)

# Umwandeln in ein Pandas DataFrame
df = data.to_pandas()

# Anzeigen der ersten Zeilen des DataFrame
print(df.head())
