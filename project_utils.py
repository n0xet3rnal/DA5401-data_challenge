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
    def __init__(self, data_file, output_file, chunk_size=500, batch_size=8, is_test=False):
        """
        Initialize the Embedder class.

        :param data_file: Path to the input JSON file (train_data.json or test_data.json)
        :param output_file: Path to the output JSON file
        :param chunk_size: Number of entries to process per chunk
        :param batch_size: Batch size for embedding
        :param is_test: Whether this is test data (no score field expected)
        """
        self.data_file = data_file
        self.output_file = output_file
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.is_test = is_test
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer("google/embeddinggemma-300m", device=self.device, trust_remote_code=True)

    def modify_initial(self):
        """
        Add indices, main metric field, and metric ID to each entry in the JSON file.
        """

        with open(self.data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Expected JSON file to contain a list of entries")

        # For test mode, try to load existing metric mapping from training
        if self.is_test:
            # Try to find existing metric mapping from training data
            train_mapping_file = '/home/jerryjose/DA5401/DA5401-data_challenge/data/train_data_metric_mapping.json'
            if os.path.exists(train_mapping_file):
                with open(train_mapping_file, 'r', encoding='utf-8') as f:
                    metric_to_id = json.load(f)
                print(f"Loaded existing metric mapping from: {train_mapping_file}")
            else:
                # Fallback: create mapping from test data
                unique_metrics = sorted(list(set(entry.get('metric_name', '') for entry in data if isinstance(entry, dict))))
                metric_to_id = {metric: idx for idx, metric in enumerate(unique_metrics)}
                print("Created new metric mapping from test data (no training mapping found)")
        else:
            # Training mode: create new mapping
            unique_metrics = sorted(list(set(entry.get('metric_name', '') for entry in data if isinstance(entry, dict))))
            metric_to_id = {metric: idx for idx, metric in enumerate(unique_metrics)}
            
            # Save the mapping for reference
            mapping_file = self.data_file.replace('.json', '_metric_mapping.json')
            with open(mapping_file, 'w', encoding='utf-8') as f:
                json.dump(metric_to_id, f, ensure_ascii=False, indent=2)
            print(f"Metric ID mapping saved to: {mapping_file}")

        # Only drop index 3766 for training data (this is a known bad training sample)
        if not self.is_test:
            data = [entry for i, entry in enumerate(data) if i != 3766]
        
        for i, entry in enumerate(data):
            if isinstance(entry, dict):
                # Add index
                entry['index'] = i

                # Add main metric
                metric_name = entry.get('metric_name', '')
                entry['main_metric'] = metric_name.split('/')[0] if metric_name else ''
                
                # Add metric ID
                entry['metric_id'] = metric_to_id.get(metric_name, -1)
        
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
                index = [int(entry['index']) for entry in chunk]
                main_metrics = [entry['main_metric'] for entry in chunk]
                metric_ids = [int(entry['metric_id']) for entry in chunk]
                
                # Handle scores - only extract if not test mode
                if not self.is_test:
                    scores = [float(entry['score']) for entry in chunk]

                # Batch encode fields
                metric_name_embeddings = self.model.encode(metric_names, batch_size=self.batch_size, show_progress_bar=False)
                user_prompt_embeddings = self.model.encode(user_prompts, batch_size=self.batch_size, show_progress_bar=False)
                response_embeddings = self.model.encode(responses, batch_size=self.batch_size, show_progress_bar=False)
                system_prompt_embeddings = self.model.encode(system_prompts, batch_size=self.batch_size, show_progress_bar=False)
                
                # Create transformed chunk based on mode
                if self.is_test:
                    # Test mode - no score field
                    transformed_chunk = [
                        {
                            'index': idx,
                            'metric_name': metric.tolist(),
                            'user_prompt': user.tolist(),
                            'response': resp.tolist(),
                            'system_prompt': sys.tolist(),
                            'main_metric': main_metric,
                            'metric_id': metric_id
                        }
                        for idx, metric, user, resp, sys, main_metric, metric_id in zip(
                            index, metric_name_embeddings, user_prompt_embeddings, response_embeddings, system_prompt_embeddings, main_metrics, metric_ids
                        )
                    ]
                else:
                    # Training mode - include score field
                    transformed_chunk = [
                        {
                            'index': idx,
                            'metric_name': metric.tolist(),
                            'user_prompt': user.tolist(),
                            'response': resp.tolist(),
                            'system_prompt': sys.tolist(),
                            'score': score,
                            'main_metric': main_metric,
                            'metric_id': metric_id
                        }
                        for idx, metric, user, resp, sys, score, main_metric, metric_id in zip(
                            index, metric_name_embeddings, user_prompt_embeddings, response_embeddings, system_prompt_embeddings, scores, main_metrics, metric_ids
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

