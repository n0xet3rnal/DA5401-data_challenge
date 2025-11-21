import torch
import torch.nn as nn
import torch.nn.functional as F

class FinalSpecialistNN(nn.Module):
    """
    Stage 1: The 2-Tower "Specialist" Metric Learner.
    
    This model's job is to learn a 128-dim "metric space" where
    distance = quality. It has two separate "towers" to process
    the "Rule" and the "Event" vectors independently.
    """
    def __init__(self, rule_dims=87, event_dims=827, latent_dim=128):
        super(FinalSpecialistNN, self).__init__()
        self.dropout_1 = nn.Dropout(0.641243742834063)
        self.dropout_2 = nn.Dropout(0.22203393374815192)
        # Tower 1: "Rule Expert" (compresses 87 -> 128)
        self.tower1 = nn.Sequential(
            nn.Linear(rule_dims, 128),
            nn.ReLU(),
            self.dropout_1,
            nn.Linear(128, latent_dim)
        )
        
        # Tower 2: "Event Expert" (compresses 827 -> 128)
        self.tower2 = nn.Sequential(
            nn.Linear(event_dims, 512),
            nn.ReLU(),
            self.dropout_2,
            nn.Linear(512, 256),
            nn.ReLU(),
            self.dropout_2,
            nn.Linear(256, latent_dim)
        )

    def forward(self, a_pca, bcd_pca):
        """
        Passes each input through its dedicated tower.
        """
        embed_rule = self.tower1(a_pca)
        embed_event = self.tower2(bcd_pca)
        return embed_rule, embed_event

class SimpleRegressorNN(nn.Module):
    """
    Stage 2: The "Score Calibrator".
    
    This model's job is to take the *relationship* features 
    from the frozen Stage 1 model and map them to a 0-10 score.
    """
    def __init__(self, input_size=384, output_size=1):
        super(SimpleRegressorNN, self).__init__()
        # Input size is 384 (embed_rule + embed_event + (embed_rule - embed_event))
        self.dropout = nn.Dropout(0.43728960253816973)
        self.layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            self.dropout,
            nn.Linear(256, 128),
            nn.ReLU(),
            self.dropout,
            nn.Linear(128,output_size)
            )

    def forward(self, x):
        return self.layers(x).squeeze(-1) # Squeeze from [batch, 1] to [batch]
    


    import torch
import torch.nn as nn
import torch.nn.functional as F

