import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

# --- STAGE 1 DATASET ---

class RankedTripletDataset(Dataset):
    """
    Dataset for Stage 1 (Metric Learner) using ranked triplet loss.
    Returns anchor (rule), positive (better event), negative (worse event), and dynamic margin.
    """
    def __init__(self, a_pca_data, bcd_pca_data, df_train):
        self.a_pca = a_pca_data
        self.bcd_pca = bcd_pca_data
        self.scores = df_train['score'].values
        self.metric_ids = df_train['metric_id'].values
        
        # Pre-compute lookup dictionary mapping each metric to all its indices
        self.metric_to_indices = {}
        for i, metric_id in enumerate(self.metric_ids):
            if metric_id not in self.metric_to_indices:
                self.metric_to_indices[metric_id] = []
            self.metric_to_indices[metric_id].append(i)
        
    def __len__(self):
        return len(self.scores)
    
    def __getitem__(self, idx):
        # Step 1: Get the Anchor (Rule) and Positive (Better Event)
        anchor_a = self.a_pca[idx]
        positive_bcd = self.bcd_pca[idx]
        positive_score = self.scores[idx]
        metric_id = self.metric_ids[idx]
        
        # Step 2: Find a "Worse" Negative
        possible_indices = self.metric_to_indices[metric_id]
        valid_worse_indices = [i for i in possible_indices if self.scores[i] < positive_score]
        
        # Step 3: Handle the two cases
        if len(valid_worse_indices) > 0:
            # Case 1: We found valid "worse" samples
            worse_idx = random.choice(valid_worse_indices)
            worse_bcd = self.bcd_pca[worse_idx]
            worse_score = self.scores[worse_idx]
        else:
            # Case 2: This sample is the lowest score for its metric
            # Create a "zero-loss" triplet by setting worse to be same as positive
            worse_bcd = positive_bcd
            worse_score = positive_score
        
        # Step 4: Calculate the dynamic margin
        margin = positive_score - worse_score
        
        # Step 5: Return all four items as tensors
        return (
            torch.tensor(anchor_a, dtype=torch.float32),
            torch.tensor(positive_bcd, dtype=torch.float32), 
            torch.tensor(worse_bcd, dtype=torch.float32),
            torch.tensor(margin, dtype=torch.float32)
        )

# --- STAGE 2 DATASET ---

class RegressorFeatureDataset(Dataset):
    """
    Dataset for Stage 2 (Score Calibrator).
    
    This dataset *uses* the frozen Stage 1 model to create
    the final 384-dim feature vector on-the-fly.
    """
    def __init__(self, a_pca_data, bcd_pca_data, scores, frozen_model, device):
        self.a_pca_data = torch.tensor(a_pca_data, dtype=torch.float32)
        self.bcd_pca_data = torch.tensor(bcd_pca_data, dtype=torch.float32)
        self.scores = torch.tensor(scores, dtype=torch.float32)
        
        self.frozen_model = frozen_model.to(device)
        self.frozen_model.eval() # Set to evaluation mode
        self.device = device
        
        self.features = self._precompute_features()

    def _precompute_features(self):
        """
        Pre-computes all features to make training much faster.
        We pass all data through the frozen model once.
        """
        print("Stage 2: Pre-computing features with frozen model...")
        all_features = []
        
        # Create a temp loader to process data in batches
        temp_loader = DataLoader(
            list(zip(self.a_pca_data, self.bcd_pca_data)), 
            batch_size=256, 
            shuffle=False
        )
        
        with torch.no_grad():
            for a_batch, bcd_batch in temp_loader:
                a_batch = a_batch.to(self.device)
                bcd_batch = bcd_batch.to(self.device)
                
                # Get embeddings from the frozen Stage 1 model
                embed_rule, embed_event = self.frozen_model(a_batch, bcd_batch)
                
                # Create the 384-dim relationship vector
                relationship_vector = torch.cat([
                    embed_rule,
                    embed_event,
                    embed_rule - embed_event # The "gap" in the new metric space
                ], dim=1)
                
                all_features.append(relationship_vector.cpu())
        
        print("Stage 2: Feature pre-computation complete.")
        return torch.cat(all_features, dim=0)

    def __len__(self):
        return len(self.scores)
    
    def __getitem__(self, idx):
        # Convert tensor indices to appropriate format
        if torch.is_tensor(idx):
            if idx.dim() == 0:  # single element tensor
                idx = idx.item()
            else:  # batch of indices - shouldn't happen in normal DataLoader usage
                idx = idx.tolist()
        
        # Return the pre-computed feature vector and the score
        return self.features[idx], self.scores[idx]