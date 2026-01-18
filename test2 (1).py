import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

df = pd.read_csv('mental_heath_unbanlanced.csv')
texts = df['text'].tolist()

embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True
)

np.save("dataset_embeddings.npy", embeddings)
print("✅ Embeddings saved")