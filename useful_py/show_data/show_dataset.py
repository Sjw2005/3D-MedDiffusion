import torchio as tio
import matplotlib.pyplot as plt
import numpy as np
import json
import glob
import os
 
def visualize_dataset(config_path, num_samples=3):
    """
    Simple 3D dataset visualization
    """
    # Load config
    with open(config_path) as f:
        dataroots = json.load(f)
    
    # Collect files
    all_files = []
    for class_name, data_path in dataroots.items():
        if isinstance(data_path, list):
            for path in data_path:
                files = glob.glob(os.path.join(path, '*.nii.gz'))
                all_files.extend([(class_name, f) for f in files])
        else:
            files = glob.glob(os.path.join(data_path, '*.nii.gz'))
            all_files.extend([(class_name, f) for f in files])
    
    print(f"Found {len(all_files)} files")
    
    # Show samples
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(min(num_samples, len(all_files))):
        class_name, file_path = all_files[i]
        
        # Load image
        img = tio.ScalarImage(file_path)
        data = img.data.squeeze().numpy()
        
        print(f"Sample {i+1}: {os.path.basename(file_path)}")
        print(f"  Shape: {data.shape}, Range: [{data.min():.3f}, {data.max():.3f}]")
        
        # Show 3 slices
        mid_x, mid_y, mid_z = data.shape[0]//2, data.shape[1]//2, data.shape[2]//2
        
        axes[i, 0].imshow(data[mid_x, :, :], cmap='gray')
        axes[i, 0].set_title(f'Axial - {class_name}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(data[:, mid_y, :], cmap='gray')
        axes[i, 1].set_title('Coronal')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(data[:, :, mid_z], cmap='gray')
        axes[i, 2].set_title('Sagittal')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
 
def test_preprocessing(config_path):
    """
    Test the actual preprocessing pipeline
    """
    import sys
    sys.path.append('.')
    from dataset.vqgan import VQGANDataset
    from torch.utils.data import DataLoader
    
    # Test different stages
    for stage in [1, 2]:
        print(f"\nTesting Stage {stage} preprocessing:")
        
        dataset = VQGANDataset(
            root_dir=config_path,
            augmentation=True,
            split='train',
            stage=stage,
            patch_size=64
        )
        
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        batch = next(iter(dataloader))
        
        data = batch['data'].squeeze().numpy()
        print(f"  Output shape: {data.shape}")
        print(f"  Value range: [{data.min():.3f}, {data.max():.3f}]")
        
        # Show one slice
        plt.figure(figsize=(8, 3))
        
        plt.subplot(131)
        plt.imshow(data[data.shape[0]//2, :, :], cmap='gray')
        plt.title(f'Stage {stage} - Axial')
        plt.axis('off')
        
        plt.subplot(132)
        plt.imshow(data[:, data.shape[1]//2, :], cmap='gray')
        plt.title('Coronal')
        plt.axis('off')
        
        plt.subplot(133)
        plt.imshow(data[:, :, data.shape[2]//2], cmap='gray')
        plt.title('Sagittal')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
 
if __name__ == "__main__":
    config_path = 'config/PatchVolume_data.json'
    
    if os.path.exists(config_path):
        # Basic visualization
        print("=== Dataset Overview ===")
        visualize_dataset(config_path, num_samples=2)
        
        # Test preprocessing
        print("\n=== Preprocessing Test ===")
        test_preprocessing(config_path)
    else:
        print(f"Config file not found: {config_path}")