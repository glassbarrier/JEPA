# TODO: Implement logic here
# 基于 le-wm-test/train.py 重构
# 重构：使用Hydra配置系统，调用新的trainer和dataset


import os
import yaml
import torch
import argparse
from models.jepa_combined import JEPACombinedModel
from data.contact_net_dataset import ContactNetDataset
from data.transforms import make_contact_net_transforms
from training.trainer import Trainer
from torch.utils.data import DataLoader

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/train/default.yaml')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup Model
    model = JEPACombinedModel(config['model'])
    model.to(device)
    
    # Setup Data
    transform = make_contact_net_transforms(crop_size=config['data']['crop_size'])
    dataset = ContactNetDataset(root_dir=config['data']['root_path'], transform=transform)
    dataloader = DataLoader(dataset, batch_size=config['data']['batch_size'], shuffle=True, num_workers=4)
    
    # Setup Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['optimization']['lr'], weight_decay=config['optimization']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['optimization']['epochs'])
    
    # Setup Trainer
    trainer = Trainer(model, optimizer, scheduler, device, config)
    
    # Training Loop
    for epoch in range(config['optimization']['epochs']):
        avg_loss = trainer.train_epoch(dataloader, epoch)
        print(f"Epoch {epoch}, Loss: {avg_loss}")
        
        # Save Checkpoint
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(config['logging']['folder'], f'model_ep{epoch}.pth'))

if __name__ == '__main__':
    main()
