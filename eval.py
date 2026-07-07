# TODO: Implement logic here
# 基于 le-wm-test/eval.py 简化
# 简化：移除规划相关代码，只保留诊断评估

import os
import yaml
import torch
import argparse
from models.jepa_combined import JEPACombinedModel
from data.contact_net_dataset import ContactNetDataset
from data.transforms import make_contact_net_transforms
from evaluation.diagnosis_eval import DiagnosisEvaluator
from torch.utils.data import DataLoader

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/eval/diagnosis.yaml')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup Model
    model = JEPACombinedModel(config['model'])
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    
    # Setup Data
    transform = make_contact_net_transforms(crop_size=config['data']['crop_size'])
    # Assuming eval dataset structure is similar
    dataset = ContactNetDataset(root_dir=config['data']['root_path'], transform=transform, is_train=False)
    dataloader = DataLoader(dataset, batch_size=config['data']['batch_size'], shuffle=False, num_workers=4)
    
    # Evaluate
    evaluator = DiagnosisEvaluator(model, device)
    results = evaluator.evaluate(dataloader)
    
    print(f"Evaluation Accuracy: {results['accuracy']:.4f}")
    print("Classification Report:")
    print(results['report'])

if __name__ == '__main__':
    main()