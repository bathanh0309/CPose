import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    """Triplet Loss for ReID metric learning research."""
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()

def compute_cosine_distance(feat1, feat2):
    """Common distance metric in ReID research."""
    feat1 = F.normalize(feat1, p=2, dim=1)
    feat2 = F.normalize(feat2, p=2, dim=1)
    return 1 - torch.mm(feat1, feat2.t())

class MetricLearningTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.criterion = TripletLoss()
    
    def train_step(self, a, p, n):
        self.optimizer.zero_grad()
        feat_a = self.model(a)
        feat_p = self.model(p)
        feat_n = self.model(n)
        loss = self.criterion(feat_a, feat_p, feat_n)
        loss.backward()
        self.optimizer.step()
        return loss.item()
