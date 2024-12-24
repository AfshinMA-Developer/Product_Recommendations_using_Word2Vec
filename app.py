# Project 6 - Product Recommendations using Word2Vec

# Course Name :         Applied Machine Learning
# Course instructor:    Sohail Tehranipour
# Student Name :        Afshin Masoudi Ashtiani
# Project 6 -           Product Recommendations using Word2Vec
# Date :                September 2024

## Step 1 : Install required libraries
"""
# Commented out IPython magic to ensure Python compatibility.
# %pip install numpy pandas openpyxl tqdm gensim matplotlib seaborn plotly umap-learn
"""

## Step 2 : Import required libraries
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from gensim.models import Word2Vec
import umap
import warnings

# Suppress warnings for better readability
warnings.filterwarnings('ignore')

# Google Colab Drive Mounting
# from google.colab import drive
# drive.mount('/content/drive', force_remount=True)

## Step 3 : Load the dataset
dataset_path = '/content/drive/My Drive/Applied Machine Learning/Project 6 : Product Recommendations using Word2Vec/datasets/OnlineRetail.xlsx'
df = pd.read_excel(dataset_path)
print(f'> Shape of the dataset is {df.shape}')
df.sample(5)

## Step 4 : Data Cleaning
def display_feature_values(df: pd.DataFrame):
    """Display feature types and unique values in the DataFrame."""
    for dtype in df.dtypes.unique():
        feature_list = df.columns[df.dtypes == dtype].tolist()
        print(f'> {dtype}: {feature_list}')

# display_feature_values(df)

# Correcting data types
df['StockCode'] = df['StockCode'].astype(str)
df['Quantity'] = df['Quantity'].abs()

# Handling missing values and duplicates
print(f'> Number of samples before handling null values: {len(df)}')
df.dropna(inplace=True)
print(f'> Number of samples after handling null values: {len(df)}')

print(f'> Number of samples before handling duplicates: {len(df)}')
df.drop_duplicates(inplace=True)
print(f'> Number of samples after handling duplicates: {len(df)}')

# Handling outliers using IQR
def remove_outliers_iqr(data: pd.DataFrame, column_name: str, threshold: float = 1.5) -> pd.DataFrame:
    """Remove outliers using IQR method."""
    Q1 = data[column_name].quantile(0.25)
    Q3 = data[column_name].quantile(0.75)
    IQR = Q3 - Q1
    return data[~((data[column_name] < (Q1 - threshold * IQR)) | (data[column_name] > (Q3 + threshold * IQR)))]

for col in ['Quantity', 'UnitPrice']:
    print(f'> Number of samples before handling {col} outliers: {len(df)}')
    df = remove_outliers_iqr(df, col)
    print(f'> Number of samples after handling {col} outliers: {len(df)}')

# Save the cleaned dataset
cleaned_dataset_path = '/content/drive/My Drive/Applied Machine Learning/Project 6 : Product Recommendations using Word2Vec/datasets/cleaned_OnlineRetail.csv'
df.to_csv(cleaned_dataset_path, index=False)
print(f"> Cleaned dataset saved to {cleaned_dataset_path}")

## Step 5 : Shuffle customers and split the dataset
train_rate = 0.9
customers = df['CustomerID'].unique().tolist()
random.shuffle(customers)
split_index = int(len(customers) * train_rate)
customers_train = customers[:split_index]

train_df = df[df['CustomerID'].isin(customers_train)]
validation_df = df[~df['CustomerID'].isin(customers_train)]
print(f'> Number of Train samples: {len(train_df)}.')
print(f'> Number of Validation samples: {len(validation_df)}.')

## Step 6 : Create purchase history sequences
def create_purchase_sequences(df, customer_ids):
    """Create purchase sequences for customers."""
    return [df[df["CustomerID"] == customer_id]["StockCode"].tolist() for customer_id in tqdm(customer_ids)]

purchases_train = create_purchase_sequences(train_df, customers_train)
purchases_validation = create_purchase_sequences(validation_df, validation_df['CustomerID'].unique())
print(f'> Number of Train Purchases: {len(purchases_train)}.')
print(f'> Number of Validation Purchases: {len(purchases_validation)}.')

## Step 7 : Train Word2Vec model
model = Word2Vec(sentences=purchases_train, vector_size=100, window=10, sg=1, hs=0, negative=10, epochs=10, seed=14)
model.init_sims(replace=True)
print(model)

# Save the Word2Vec model
model_path = '/content/drive/My Drive/Applied Machine Learning/Project 6 : Product Recommendations using Word2Vec/models/word2vec_model.model'
model.save(model_path)
print(f'> The Word2Vec model saved to {model_path}.')

## Step 8 : Validate results and prepare product descriptions
products = train_df[["StockCode", "Description"]].drop_duplicates('StockCode')
products_dict = products.groupby('StockCode')['Description'].apply(list).to_dict()

def similar_products(product_id, n=6):
    """Get similar products to a given product ID."""
    try:
        similar_items = model.wv.similar_by_key(product_id, topn=n + 1)[1:]
        return [(products_dict[item[0]][0], item[1]) for item in similar_items]
    except KeyError:
        return []

def aggregate_vectors(product_ids):
    """Aggregate vectors from the given product IDs."""
    valid_vectors = [model.wv[pid] for pid in product_ids if pid in model.wv.key_to_index]
    return np.mean(valid_vectors, axis=0) if valid_vectors else None

# Validate results
recommendation = similar_products('90019A')
print(recommendation)

user_vector = aggregate_vectors(purchases_validation[0])
if user_vector is not None:
    recommendations = similar_products(user_vector)
    print(recommendations)

user_last_ten_vector = aggregate_vectors(purchases_validation[0][-10:])
if user_last_ten_vector is not None:
    last_ten_recommendations = similar_products(user_last_ten_vector)
    print(last_ten_recommendations)

## Step 9 : Visualize embeddings using UMAP
X = model.wv.vectors
umap_embeddings = umap.UMAP(n_neighbors=30, min_dist=0.0, n_components=2, random_state=42).fit_transform(X)
plt.figure(figsize=(10, 9))
plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], s=3, cmap='Spectral')
plt.title('UMAP projection of Word2Vec embeddings')
plt.show()