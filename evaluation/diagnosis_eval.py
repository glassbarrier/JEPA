# TODO: Implement logic here

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class DiagnosisEvaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    def evaluate(self, dataloader):
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for imgs, labels in dataloader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass through diagnosis head
                logits = self.model.forward_diagnosis(imgs)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        acc = accuracy_score(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)
        
        return {
            'accuracy': acc,
            'report': report,
            'confusion_matrix': cm
        }