import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

class QuickFeatureExtractor:
    def __init__(self, model_type='dinov2_small', device='cuda'):
        self.device = device
        self.model_type = model_type
        self.model = self._load_model()
        self.transform = self._get_transform()
        
    def _load_model(self):
        if self.model_type == 'dinov2_small':
            return self._load_dinov2('small')
        elif self.model_type == 'dinov2_base':
            return self._load_dinov2('base')
        elif self.model_type == 'dinov2_large':
            return self._load_dinov2('large')
        elif self.model_type == 'resnet50':
            return self._load_resnet50()
        elif self.model_type == 'clip_vit':
            return self._load_clip()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _load_dinov2(self, size):
        size_map = {
            'small': 'facebook/dinov2-small',
            'base': 'facebook/dinov2-base',
            'large': 'facebook/dinov2-large'
        }
        model_name = size_map[size]
        
        print(f"Loading DINOv2 {size} model from HuggingFace...")
        try:
            from transformers import AutoModel, AutoImageProcessor
            model = AutoModel.from_pretrained(model_name)
            processor = AutoImageProcessor.from_pretrained(model_name)
            model = model.to(self.device).eval()
            self.processor = processor
            print(f"Successfully loaded DINOv2 {size}")
            return model
        except Exception as e:
            print(f"Error loading DINOv2: {e}")
            print("Falling back to torchvision implementation...")
            return self._load_fallback_dinov2(size)
    
    def _load_fallback_dinov2(self, size):
        if size == 'small':
            embed_dim = 384
        elif size == 'base':
            embed_dim = 768
        elif size == 'large':
            embed_dim = 1024
        else:
            embed_dim = 384
        
        model = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{size}14')
        model = model.to(self.device).eval()
        print(f"Loaded DINOv2 {size} from torch hub")
        return model
    
    def _load_resnet50(self):
        print("Loading ResNet50 pretrained model...")
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1])  # Remove final FC layer
        model = model.to(self.device).eval()
        print("Successfully loaded ResNet50")
        return model
    
    def _load_clip(self):
        print("Loading CLIP model...")
        try:
            import clip
            model, preprocess = clip.load("ViT-B/32", device=self.device)
            model = model.visual
            self.clip_preprocess = preprocess
            print("Successfully loaded CLIP")
            return model
        except ImportError:
            print("CLIP not installed. Install with: pip install git+https://github.com/openai/CLIP.git")
            raise
    
    def _get_transform(self):
        if hasattr(self, 'processor'):
            return None  # Use processor
        elif hasattr(self, 'clip_preprocess'):
            return None  # Use clip_preprocess
        else:
            return transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def extract_features(self, image_path):
        image = Image.open(image_path).convert('RGB')
        
        if hasattr(self, 'processor'):
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                features = self.model(**inputs)
                features = features.last_hidden_state.mean(dim=1)  # Global average pooling
        elif hasattr(self, 'clip_preprocess'):
            image = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.model(image)
                features = features / features.norm(dim=-1, keepdim=True)
        else:
            image = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.model(image)
                features = features.squeeze()
        
        return features.cpu().numpy()
    
    def extract_features_batch(self, image_paths, batch_size=32):
        all_features = []
        all_paths = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting features"):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            
            for path in batch_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    if hasattr(self, 'processor'):
                        inputs = self.processor(images=image, return_tensors="pt")
                        batch_images.append(inputs)
                    elif hasattr(self, 'clip_preprocess'):
                        image = self.clip_preprocess(image)
                        batch_images.append(image)
                    else:
                        image = self.transform(image)
                        batch_images.append(image)
                    all_paths.append(path)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    continue
            
            if not batch_images:
                continue
            
            if hasattr(self, 'processor'):
                batch_inputs = {}
                for key in batch_images[0].keys():
                    batch_inputs[key] = torch.cat([item[key] for item in batch_images]).to(self.device)
                with torch.no_grad():
                    features = self.model(**batch_inputs)
                    features = features.last_hidden_state.mean(dim=1)
            elif hasattr(self, 'clip_preprocess'):
                batch_images = torch.stack(batch_images).to(self.device)
                with torch.no_grad():
                    features = self.model(batch_images)
                    features = features / features.norm(dim=-1, keepdim=True)
            else:
                batch_images = torch.stack(batch_images).to(self.device)
                with torch.no_grad():
                    features = self.model(batch_images)
                    features = features.squeeze()
            
            all_features.extend(features.cpu().numpy())
        
        return np.array(all_features), all_paths


def process_dataset(data_root, output_dir, model_type='dinov2_small', split='train'):
    extractor = QuickFeatureExtractor(model_type=model_type)
    
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if split == 'train':
        image_dir = data_root / 'train' / 'good'
        split_name = 'train_normal'
    elif split == 'test':
        image_dir = data_root / 'test' / 'good'
        split_name = 'test_normal'
    else:
        # Process all test categories
        test_dir = data_root / 'test'
        all_features = []
        all_paths = []
        all_labels = []
        
        for category_dir in sorted(test_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            
            category_name = category_dir.name
            label = 0 if category_name == 'good' else 1
            
            print(f"Processing {category_name}...")
            image_paths = list(category_dir.glob('*.png')) + list(category_dir.glob('*.jpg'))
            
            if not image_paths:
                print(f"No images found in {category_dir}")
                continue
            
            features, paths = extractor.extract_features_batch(image_paths)
            all_features.extend(features)
            all_paths.extend(paths)
            all_labels.extend([label] * len(features))
        
        # Save all test features
        features_array = np.array(all_features)
        labels_array = np.array(all_labels)
        
        np.save(output_dir / 'test_features.npy', features_array)
        np.save(output_dir / 'test_labels.npy', labels_array)
        np.save(output_dir / 'test_paths.npy', np.array(all_paths, dtype=object))
        
        print(f"Saved {len(features_array)} test features to {output_dir}")
        return
    
    if split == 'train' or split == 'test':
        if not image_dir.exists():
            print(f"Image directory {image_dir} does not exist")
            return
        
        image_paths = list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg'))
        print(f"Found {len(image_paths)} images in {image_dir}")
        
        if not image_paths:
            print("No images found!")
            return
        
        features, paths = extractor.extract_features_batch(image_paths)
        
        np.save(output_dir / f'{split_name}_features.npy', features)
        np.save(output_dir / f'{split_name}_paths.npy', np.array(paths, dtype=object))
        
        print(f"Saved {len(features)} features to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Quick feature extraction for anomaly detection')
    parser.add_argument('--data_root', type=str, default='./data', help='Data root directory')
    parser.add_argument('--output_dir', type=str, default='./features', help='Output directory for features')
    parser.add_argument('--model_type', type=str, default='dinov2_small', 
                       choices=['dinov2_small', 'dinov2_base', 'dinov2_large', 'resnet50', 'clip_vit'],
                       help='Model type for feature extraction')
    parser.add_argument('--split', type=str, default='all', 
                       choices=['train', 'test', 'all'],
                       help='Data split to process')
    
    args = parser.parse_args()
    
    if args.split == 'all':
        print("Processing all splits...")
        process_dataset(args.data_root, args.output_dir, args.model_type, 'train')
        process_dataset(args.data_root, args.output_dir, args.model_type, 'test')
    else:
        process_dataset(args.data_root, args.output_dir, args.model_type, args.split)


if __name__ == '__main__':
    main()