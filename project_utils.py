import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
import os
import json
import shutil
import tempfile
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import root_mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
import lightgbm as lgb

class FeatureEngineer:
    def __init__(self, df, config):
        """
        Initialize the FeatureEngineer with a DataFrame and configuration.

        :param df: Input DataFrame
        :param config: Dictionary specifying features to compute and fields
        """
        self.df = df
        self.config = config

    def original_features(self, field):
        return self.df(field)


    def euclidean_distance(self, field1, field2):
        """
        Compute the Euclidean distance between two vector fields.
        Ensure proper broadcasting for 2D arrays.
        """
        field1_values = np.stack(self.df[field1].values)
        field2_values = np.stack(self.df[field2].values)
        
        # Ensure both fields are 2D arrays
        if field1_values.ndim == 1:
            field1_values = field1_values[:, np.newaxis]
        if field2_values.ndim == 1:
            field2_values = field2_values[:, np.newaxis]
        
        return np.sqrt(np.sum((field1_values - field2_values) ** 2, axis=1))

    def difference_vector(self, field1, field2):
        """
        Compute the difference between two vector fields.
        Ensure proper broadcasting for 2D arrays.
        """
        field1_values = np.stack(self.df[field1].values)
        field2_values = np.stack(self.df[field2].values)
        
        # Ensure both fields are 2D arrays
        if field1_values.ndim == 1:
            field1_values = field1_values[:, np.newaxis]
        if field2_values.ndim == 1:
            field2_values = field2_values[:, np.newaxis]
        
        return field1_values - field2_values

    def inner_product(self, field1, field2):
        """
        Compute the row-wise inner product between two vector fields.
        """
        field1_values = np.stack(self.df[field1].values)
        field2_values = np.stack(self.df[field2].values)

        # Compute the row-wise dot product
        return np.einsum('ij,ij->i', field1_values, field2_values)

    def manhattan_distance(self, field1, field2):
        """
        Compute the Manhattan distance (L1 norm) between two vector fields.
        """
        field1_values = np.stack(self.df[field1].values)
        field2_values = np.stack(self.df[field2].values)

        # Ensure both fields are 2D arrays
        if field1_values.ndim == 1:
            field1_values = field1_values[:, np.newaxis]
        if field2_values.ndim == 1:
            field2_values = field2_values[:, np.newaxis]

        return np.sum(np.abs(field1_values - field2_values), axis=1)

    def cosine_similarity(self, field1, field2):
        """
        Compute the cosine similarity between two vector fields.
        """
        field1_values = np.stack(self.df[field1].values)
        field2_values = np.stack(self.df[field2].values)

        # Ensure both fields are 2D arrays
        if field1_values.ndim == 1:
            field1_values = field1_values[:, np.newaxis]
        if field2_values.ndim == 1:
            field2_values = field2_values[:, np.newaxis]

        dot_product = np.einsum('ij,ij->i', field1_values, field2_values)
        norm1 = np.linalg.norm(field1_values, axis=1)
        norm2 = np.linalg.norm(field2_values, axis=1)

        return dot_product / (norm1 * norm2 + 1e-8)  # Add epsilon to avoid division by zero

    def elementwise_product(self, field1, field2):
        """
        Compute the element-wise product (agreement vector) between two vector fields.
        """
        field1_values = np.stack(self.df[field1].values)
        field2_values = np.stack(self.df[field2].values)

        # Ensure both fields are 2D arrays
        if field1_values.ndim == 1:
            field1_values = field1_values[:, np.newaxis]
        if field2_values.ndim == 1:
            field2_values = field2_values[:, np.newaxis]

        return field1_values * field2_values

    def create_features(self):
        """
        Dynamically create features based on the configuration, including self-features.
        """
        feature_dict = {}  # Use a dictionary to collect all columns
        for feature, field_pairs in self.config.items():
            if feature == "original_features":
                # Handle original features dynamically
                for field in self.df.columns:
                    print(f'Adding original feature: {field}')
                    field_values = self.df[field].values
                    if field_values.ndim == 1:
                        feature_dict[f"{feature}--{field}--dim0"] = field_values
                    else:
                        for dim in range(field_values.shape[1]):
                            feature_dict[f"{feature}--{field}--dim{dim}"] = field_values[:, dim]
                    print(f'Added original feature: {field}')
            else:
                for field1, field2 in field_pairs:
                    method = getattr(self, feature, None)
                    if method:
                        print(f'Creating feature: {feature}--{field1}-{field2}')
                        feature_matrix = method(field1, field2)
                        # Expand 2D array into multiple columns
                        if feature_matrix.ndim == 2:
                            for dim in range(feature_matrix.shape[1]):
                                feature_dict[f"{feature}--{field1}-{field2}--dim{dim}"] = feature_matrix[:, dim]
                        else:
                            feature_dict[f"{feature}--{field1}-{field2}"] = feature_matrix
                        print(f'Created feature: {feature}--{field1}-{field2}')
                    else:
                        raise ValueError(f"Feature method '{feature}' not found in FeatureEngineer.")

        # Add index, score, and main_metric
        feature_dict['index'] = self.df['index']
        feature_dict['score'] = self.df['score']
        feature_dict['main_metric'] = self.df['main_metric']

        # Create the DataFrame at once
        df_new = pd.DataFrame(feature_dict)
        return df_new

class DataGenerator:
    def __init__(self, df, test_fraction, fold_k, metric_column):
        """
        Initialize the DataGenerator with a DataFrame, test fraction, number of folds, and metric column.

        :param df: Input DataFrame
        :param test_fraction: Fraction of data to be used for testing
        :param fold_k: Number of folds for cross-validation
        :param metric_column: Column name representing the metric
        """
        self.df = df
        self.test_fraction = test_fraction
        self.fold_k = fold_k
        self.metric_column = metric_column

    def generate_splits(self):
        """
        Generate test and cross-validation splits while maintaining per-metric proportions.

        :return: A dictionary containing test data and cross-validation splits
        """
        # Check for unique values in the metric column
        value_counts = self.df[self.metric_column].value_counts()
        singular_values = value_counts[value_counts == 1].index.tolist()

        if singular_values:
            print(f"Dropping rows with singular values in '{self.metric_column}': {singular_values}")
            self.df = self.df[~self.df[self.metric_column].isin(singular_values)]

        # Split the data into train and test sets while stratifying by the metric column
        train_data, test_data = train_test_split(
            self.df, test_size=self.test_fraction, stratify=self.df[self.metric_column], random_state=42
        )
        if self.fold_k == 1:
            return {
                "test_data": test_data,
                "train_data": train_data
            }

        # Initialize StratifiedKFold for cross-validation
        skf = StratifiedKFold(n_splits=self.fold_k, shuffle=True, random_state=42)

        # Generate cross-validation splits
        cv_splits = []
        for train_idx, val_idx in skf.split(train_data, train_data[self.metric_column]):
            train_split = train_data.iloc[train_idx]
            val_split = train_data.iloc[val_idx]
            cv_splits.append((train_split, val_split))

        return {
            "test_data": test_data,
            "cv_splits": cv_splits
        }
    
class Embedder:
    def __init__(self, data_file, output_file, chunk_size=500, batch_size=16):
        """
        Initialize the Embedder class.

        :param data_file: Path to the input JSON file (train_data.json)
        :param output_file: Path to the output JSON file
        :param chunk_size: Number of entries to process per chunk
        :param batch_size: Batch size for embedding
        """
        self.data_file = data_file
        self.output_file = output_file
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer("google/embeddinggemma-300m", device=self.device, trust_remote_code=True)

    def modify_initial(self):
        """
        Add indices and the main metric field to each entry in the JSON file.
        """

        with open(self.data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Expected JSON file to contain a list of entries")

        for i, entry in enumerate(data):
            if isinstance(entry, dict):
                # Add index
                entry['index'] = i

                # Add main metric
                metric_name = entry.get('metric_name', '')
                entry['main_metric'] = metric_name.split('/')[0] if metric_name else ''
                
        #drop index 3766
        data = [entry for entry in data if entry.get('index') != 3766]
        # Write back to the file atomically
        dirpath = os.path.dirname(self.data_file) or '.'
        temp_fp = None
        try:
            with tempfile.NamedTemporaryFile('w', delete=False, dir=dirpath, encoding='utf-8') as tf:
                json.dump(data, tf, ensure_ascii=False, indent=2)
                temp_fp = tf.name
            os.replace(temp_fp, self.data_file)
            temp_fp = None
        except Exception:
            if temp_fp and os.path.exists(temp_fp):
                os.remove(temp_fp)
            raise

    def embed_fields(self):
        """
        Embed specified fields using the SentenceTransformer model with GPU-enabled batch processing.
        """
        with open(self.data_file, 'r') as f:
            data = json.load(f)

        # Process data in chunks
        with open(self.output_file, 'w') as out_file:
            for i in tqdm(range(0, len(data), self.chunk_size), desc="Processing chunks"):
                chunk = data[i:i + self.chunk_size]

                # Extract fields to be embedded
                metric_names = [entry['metric_name'] for entry in chunk]
                user_prompts = [entry['user_prompt'] for entry in chunk]
                responses = [entry['response'] for entry in chunk]
                system_prompts = [entry['system_prompt'] if entry['system_prompt'] else '' for entry in chunk]
                scores = [float(entry['score']) for entry in chunk]
                index = [int(entry['index']) for entry in chunk]
                main_metrics = [entry['main_metric'] for entry in chunk]

                # Batch encode fields
                metric_name_embeddings = self.model.encode(metric_names, batch_size=self.batch_size, show_progress_bar=False)
                user_prompt_embeddings = self.model.encode(user_prompts, batch_size=self.batch_size, show_progress_bar=False)
                response_embeddings = self.model.encode(responses, batch_size=self.batch_size, show_progress_bar=False)
                system_prompt_embeddings = self.model.encode(system_prompts, batch_size=self.batch_size, show_progress_bar=False)

                # Combine embeddings and scores into a chunk of data
                transformed_chunk = [
                    {
                        'index': idx,
                        'metric_name': metric.tolist(),
                        'user_prompt': user.tolist(),
                        'response': resp.tolist(),
                        'system_prompt': sys.tolist(),
                        'score': score,
                        'main_metric': main_metric
                    }
                    for idx, metric, user, resp, sys, score, main_metric in zip(
                        index, metric_name_embeddings, user_prompt_embeddings, response_embeddings, system_prompt_embeddings, scores, main_metrics
                    )
                ]

                # Write the chunk to the output file
                for entry in transformed_chunk:
                    out_file.write(json.dumps(entry) + '\n')

    def process(self):
        """
        Run the full embedding process: modify initial data and embed fields.
        """
        self.modify_initial()
        self.embed_fields()


def prepare_final(df):
    y = df['score']
    X = df.drop(['index','score','main_metric'], axis =1)

    return X, y

def LBGMCrossValidate(dataset, fold_k, model_type='regressor', n_estimators=1000, learning_rate=0.05, random_state=42):
    """
    Perform cross-validation using LightGBM.

    :param dataset: Dictionary containing 'cv_splits' (list of train-validation splits)
    :param fold_k: Number of folds for cross-validation
    :param model_type: Type of model ('regressor' or 'classifier')
    :param n_estimators: Number of estimators for LightGBM
    :param learning_rate: Learning rate for LightGBM
    :param random_state: Random state for reproducibility
    :return: Dictionary containing average metrics and per-fold metrics
    """
 
    metrics = []
    device  = 'cpu'
    print(f"Using device: {device}")

    for fold in range(fold_k):
        print(f"--- FOLD {fold + 1}/{fold_k} ---")

        # 1. Create the training and validation sets for this fold
        X_train, y_train = prepare_final(dataset['cv_splits'][fold][0])
        X_val, y_val = prepare_final(dataset['cv_splits'][fold][1])

        # 2. Initialize the model
        if model_type == 'regressor':
            model = lgb.LGBMRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=random_state,
                device=device
            )
            eval_metric = 'rmse'
        elif model_type == 'classifier':
            model = lgb.LGBMClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=random_state,
                device=device
            )
            eval_metric = 'logloss'
        else:
            raise ValueError("Invalid model_type. Choose 'regressor' or 'classifier'.")

        # 3. Train the model
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=eval_metric,
            callbacks=[lgb.early_stopping(10, verbose=True)]
        )

        # 4. Get predictions and calculate metrics
        preds = model.predict(X_val)
        if model_type == 'regressor':
            fold_metric = root_mean_squared_error(y_val, preds)  # RMSE
            print(f"Fold {fold + 1} RMSE: {fold_metric}")
        elif model_type == 'classifier':
            preds = (preds > 0.5).astype(int)  # Threshold for binary classification
            fold_metric = {
                "accuracy": accuracy_score(y_val, preds),
                "precision": precision_score(y_val, preds, average='weighted', zero_division=0),
                "recall": recall_score(y_val, preds, average='weighted', zero_division=0),
                "f1": f1_score(y_val, preds, average='weighted', zero_division=0)
            }
            print(f"Fold {fold + 1} Metrics: {fold_metric}")

        metrics.append(fold_metric)

    # 5. Calculate average metrics
    if model_type == 'regressor':
        avg_metric = np.mean(metrics)
        print("\n--- Cross-Validation Summary ---")
        print(f"All RMSE Scores: {metrics}")
        print(f"Average RMSE: {avg_metric}")
        return {"average_rmse": avg_metric, "fold_metrics": metrics}
    elif model_type == 'classifier':
        avg_metrics = {
            "accuracy": np.mean([m["accuracy"] for m in metrics]),
            "precision": np.mean([m["precision"] for m in metrics]),
            "recall": np.mean([m["recall"] for m in metrics]),
            "f1": np.mean([m["f1"] for m in metrics])
        }
        print("\n--- Cross-Validation Summary ---")
        print(f"Average Metrics: {avg_metrics}")
        return {"average_metrics": avg_metrics, "fold_metrics": metrics}

def train_neural_network(model, train_data, test_data, learning_rate=0.01, epochs=20, batch_size=32):
    """
    Train and evaluate a neural network model.

    :param model: PyTorch model to train
    :param train_data: Tuple (X_train, y_train) with training features and labels
    :param test_data: Tuple (X_test, y_test) with testing features and labels
    :param learning_rate: Learning rate for the optimizer
    :param epochs: Number of training epochs
    :param batch_size: Batch size for training
    :return: Trained model and evaluation metrics
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Unpack training and testing data
    X_train, y_train = prepare_final(train_data)
    X_test, y_test = prepare_final(test_data)

    # Convert data to tensors, cast to Float, and move to device
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).to(device)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).to(device)

    # Create DataLoader for batching
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Move model to device
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_X, batch_y in train_dataloader:
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_X)

            # Compute the loss
            loss = criterion(outputs, batch_y)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Print epoch loss
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_dataloader):.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        #print train and test accuracy
        train_predictions = model(X_train_tensor).argmax(dim=1)
        train_accuracy = (train_predictions == y_train_tensor).float().mean().item()
        print(f"Train Accuracy: {train_accuracy * 100:.2f}%")

        predictions = model(X_test_tensor).argmax(dim=1)
        accuracy = (predictions == y_test_tensor).float().mean().item()
        print(f"Test Accuracy: {accuracy * 100:.2f}%")

    return model
