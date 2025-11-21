import torch
import numpy as np
import pandas as pd
from torch.utils.data import WeightedRandomSampler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def load_full_dataset(file_path):
    df = pd.read_json(file_path, lines=True)
    df['metric_vector'] = df['metric_name']
    return df


def get_sampler_weights(labels_series):
    """
    Calculates weights for WeightedRandomSampler from a pandas Series.
    Works for both 'score' (numeric) and 'metric_id' (numeric).
    """
    counts = labels_series.value_counts().sort_index()
    
    # Create a full lookup from the sparse counts
    # Get max label + 1 to create a dense tensor
    max_label = labels_series.max()
    all_counts = torch.zeros(max_label + 1, dtype=torch.float32)
    for label, count in counts.items():
        all_counts[label] = count
            
    # Calculate weight = 1.0 / (count + epsilon)
    class_weights = 1.0 / (all_counts + 1e-6)
    
    # Create the final list of weights for every sample
    sample_weights = torch.tensor(
        [class_weights[label] for label in labels_series.values],
        dtype=torch.float32
    )
    return sample_weights

def fit_and_transform_pca(df_train, df_val, n_comp_a, n_comp_bcd):
    """
    Handles the full, leak-proof PCA pipeline.
    Fits on train, transforms both train and val.
    Returns transformed data and fitted transformers for reuse.
    """
    print("Starting leak-proof PCA pipeline...")
    
    # --- 1. Prepare raw arrays ---
    print("Stacking raw vectors...")
    A_train_raw = np.stack(df_train['metric_vector'].values)
    BCD_train_raw = np.concatenate([
        np.stack(df_train['user_prompt'].values),
        np.stack(df_train['system_prompt'].values),
        np.stack(df_train['response'].values)
    ], axis=1)
    
    A_val_raw = np.stack(df_val['metric_vector'].values)
    BCD_val_raw = np.concatenate([
        np.stack(df_val['user_prompt'].values),
        np.stack(df_val['system_prompt'].values),
        np.stack(df_val['response'].values)
    ], axis=1)

    # --- 2. Fit Scalers and PCAs ONLY on training data ---
    print("Fitting Scalers and PCA models on *training data only*...")
    scaler_A = StandardScaler().fit(A_train_raw)
    scaler_BCD = StandardScaler().fit(BCD_train_raw)
    
    pca_A = PCA(n_components=n_comp_a).fit(scaler_A.transform(A_train_raw))
    pca_BCD = PCA(n_components=n_comp_bcd).fit(scaler_BCD.transform(BCD_train_raw))
    
    # --- 3. Transform both train and val data ---
    print("Transforming all data with fitted models...")
    A_train_pca = pca_A.transform(scaler_A.transform(A_train_raw))
    BCD_train_pca = pca_BCD.transform(scaler_BCD.transform(BCD_train_raw))
    
    A_val_pca = pca_A.transform(scaler_A.transform(A_val_raw))
    BCD_val_pca = pca_BCD.transform(scaler_BCD.transform(BCD_val_raw))
    
    print("PCA transformation complete.")
    
    # Return both transformed data and fitted transformers
    transformers = {
        'scaler_A': scaler_A,
        'scaler_BCD': scaler_BCD,
        'pca_A': pca_A,
        'pca_BCD': pca_BCD
    }
    
    return A_train_pca, BCD_train_pca, A_val_pca, BCD_val_pca, transformers