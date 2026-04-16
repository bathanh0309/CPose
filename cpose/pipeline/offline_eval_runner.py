import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

class OfflineEvalRunner:
    """
    Runner for offline evaluation of ADL models on benchmark datasets.
    Computes standard research metrics.
    """
    def __init__(self, model, dataloader, device='cuda'):
        self.model = model
        self.dataloader = dataloader
        self.device = device

    def evaluate(self):
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in self.dataloader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        return self.compute_metrics(all_labels, all_preds)

    def compute_metrics(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist()
        }
