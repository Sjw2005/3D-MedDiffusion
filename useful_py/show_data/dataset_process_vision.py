#!/usr/bin/env python3
"""
独立的数据集预处理测试脚本
可以单独运行，查看预处理效果
"""
 
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import glob
import torchio as tio
from torch.utils.data import Dataset, DataLoader
import random
 
# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)
 
# 直接复制项目的VQGANDataset代码，便于独立运行
class TestVQGANDataset(Dataset):
    def __init__(self, root_dir=None, augmentation=False, split='train', stage=1, patch_size=64):
        randnum = 216
        self.file_names = []
        self.stage = stage
        print(f"Loading dataset from: {root_dir}")
        
        if root_dir.endswith('json'):
            with open(root_dir) as json_file:
                dataroots = json.load(json_file)
            for key, value in dataroots.items():
                if type(value) == list:
                    for path in value:
                        self.file_names += (glob.glob(os.path.join(path, '*.nii.gz'), recursive=True))
                else:
                    self.file_names += (glob.glob(os.path.join(value, '*.nii.gz'), recursive=True))
        else:
            self.root_dir = root_dir
            self.file_names = glob.glob(os.path.join(root_dir, '*.nii.gz'), recursive=True)
        
        if not self.file_names:
            print("警告: 没有找到.nii.gz文件，请检查路径配置")
            return
            
        random.seed(randnum)
        random.shuffle(self.file_names)
 
        self.split = split
        self.augmentation = augmentation
        if split == 'train':
            self.file_names = self.file_names[:-40] if len(self.file_names) > 40 else self.file_names
        elif split == 'val':
            self.file_names = self.file_names[-40:] if len(self.file_names) > 40 else self.file_names[:5]
            self.augmentation = False 
            
        # 预处理组件
        self.patch_sampler = tio.data.UniformSampler(patch_size)
        self.patch_sampler_192 = tio.data.UniformSampler((192, 192, 192))
        self.patch_sampler_256 = tio.data.UniformSampler((256, 256, 256))
        self.randomflip = tio.RandomFlip(axes=(0, 1), flip_probability=0.5)
        
        print(f'Dataset loaded: {len(self.file_names)} files, patch size: {patch_size}')
        print(f'Stage: {stage}, Split: {split}, Augmentation: {augmentation}')
 
    def __len__(self):
        return len(self.file_names)
 
    def __getitem__(self, index):
        path = self.file_names[index]
        print(f"Processing: {os.path.basename(path)}")
        
        # 1. 加载原始数据
        whole_img = tio.ScalarImage(path)
        original_shape = whole_img.shape
        original_data_range = (whole_img.data.min().item(), whole_img.data.max().item())
        
        # 2. 分块采样策略
        if self.stage == 1 and self.split == 'train':
            img = None
            attempts = 0
            while (img is None or img.data.sum() == 0) and attempts < 10:
                img = next(self.patch_sampler(tio.Subject(image=whole_img)))['image']
                attempts += 1
            if attempts >= 10:
                print(f"警告: 无法找到非空patch，使用原图")
                img = whole_img
        elif self.stage == 2 and self.split == 'train':
            img = whole_img
            if img.shape[1] * img.shape[2] * img.shape[3] > 256 * 256 * 128:
                img = next(self.patch_sampler_192(tio.Subject(image=img)))['image']
        elif self.split == 'val':
            img = whole_img
            if img.shape[1] * img.shape[2] * img.shape[3] > 256 * 256 * 256:
                img = next(self.patch_sampler_256(tio.Subject(image=img)))['image']
        
        sampled_shape = img.shape
        
        # 3. 数据增强
        augmentation_applied = []
        if self.augmentation:
            # 随机翻转
            if hasattr(self, 'randomflip'):
                img_before_flip = img.data.clone()
                img = self.randomflip(img)
                if not torch.equal(img_before_flip, img.data):
                    augmentation_applied.append("flip")
        
        # 4. 最终预处理
        imageout = img.data
        
        if self.augmentation and random.random() > 0.5:
            imageout = torch.rot90(imageout, dims=(1, 2))
            augmentation_applied.append("rotation")
        
        # 数值归一化: [0,1] -> [-1,1]
        pre_norm_range = (imageout.min().item(), imageout.max().item())
        imageout = imageout * 2 - 1
        post_norm_range = (imageout.min().item(), imageout.max().item())
        
        # 维度变换
        pre_transpose_shape = imageout.shape
        imageout = imageout.transpose(1, 3).transpose(2, 3)
        final_shape = imageout.shape
        imageout = imageout.type(torch.float32)
 
        # 返回处理信息
        processing_info = {
            'file_path': path,
            'original_shape': original_shape,
            'original_data_range': original_data_range,
            'sampled_shape': sampled_shape,
            'augmentation_applied': augmentation_applied,
            'pre_norm_range': pre_norm_range,
            'post_norm_range': post_norm_range,
            'pre_transpose_shape': pre_transpose_shape,
            'final_shape': final_shape
        }
 
        if self.split == 'val':
            return {
                'data': imageout, 
                'affine': img.affine, 
                'path': path,
                'processing_info': processing_info
            }
        else:
            return {
                'data': imageout,
                'processing_info': processing_info
            }
 
 
def visualize_preprocessing_steps(config_path, num_samples=3, stage=1, augmentation=True):
    """
    可视化预处理的每个步骤
    """
    print("=" * 60)
    print("数据集预处理步骤可视化")
    print("=" * 60)
    
    # 创建数据集
    dataset = TestVQGANDataset(
        root_dir=config_path,
        augmentation=augmentation,
        split='train',
        stage=stage,
        patch_size=64
    )
    
    if len(dataset) == 0:
        print("数据集为空，请检查配置文件路径")
        return
    
    # 创建dataloader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, batch in enumerate(dataloader):
        if i >= num_samples:
            break
            
        data = batch['data'].squeeze().numpy()
        info = batch['processing_info']
        
        print(f"\n样本 {i+1} 处理信息:")
        print(f"  文件: {os.path.basename(info['file_path'][0])}")
        print(f"  原始形状: {info['original_shape'][0]}")
        print(f"  原始数值范围: [{info['original_data_range'][0][0]:.3f}, {info['original_data_range'][0][1]:.3f}]")
        print(f"  采样后形状: {info['sampled_shape'][0]}")
        print(f"  应用的增强: {info['augmentation_applied'][0]}")
        print(f"  归一化前范围: [{info['pre_norm_range'][0][0]:.3f}, {info['pre_norm_range'][0][1]:.3f}]")
        print(f"  归一化后范围: [{info['post_norm_range'][0][0]:.3f}, {info['post_norm_range'][0][1]:.3f}]")
        print(f"  最终形状: {info['final_shape'][0]}")
        
        # 数据已经在[-1,1]范围，转换到[0,1]用于显示
        display_data = (data + 1) / 2
        
        # 显示三个正交切片
        mid_dims = [s//2 for s in display_data.shape]
        
        axes[i, 0].imshow(display_data[mid_dims[0], :, :], cmap='gray')
        axes[i, 0].set_title(f'样本{i+1} - Axial切片')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(display_data[:, mid_dims[1], :], cmap='gray')
        axes[i, 1].set_title(f'Coronal切片')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(display_data[:, :, mid_dims[2]], cmap='gray')
        axes[i, 2].set_title(f'Sagittal切片')
        axes[i, 2].axis('off')
        
        # 显示数据分布直方图
        axes[i, 3].hist(data.flatten(), bins=50, alpha=0.7, color='blue')
        axes[i, 3].set_title(f'像素值分布\n范围: [{data.min():.2f}, {data.max():.2f}]')
        axes[i, 3].set_xlabel('像素值')
        axes[i, 3].set_ylabel('频次')
    
    plt.tight_layout()
    plt.show()
 
 
def compare_preprocessing_stages():
    """
    比较不同训练阶段的预处理差异
    """
    config_path = 'config/PatchVolume_data.json'
    
    print("\n" + "=" * 60)
    print("比较不同训练阶段的预处理差异")
    print("=" * 60)
    
    stages = [1, 2]
    stage_names = ['阶段1(Patch训练)', '阶段2(全图微调)']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for stage_idx, stage in enumerate(stages):
        print(f"\n{stage_names[stage_idx]}:")
        
        dataset = TestVQGANDataset(
            root_dir=config_path,
            augmentation=False,  # 关闭增强以便比较
            split='train',
            stage=stage,
            patch_size=64
        )
        
        if len(dataset) > 0:
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            batch = next(iter(dataloader))
            
            data = batch['data'].squeeze().numpy()
            info = batch['processing_info']
            
            print(f"  处理后形状: {info['final_shape'][0]}")
            print(f"  数值范围: [{info['post_norm_range'][0][0]:.3f}, {info['post_norm_range'][0][1]:.3f}]")
            
            # 显示
            display_data = (data + 1) / 2
            mid_dims = [s//2 for s in display_data.shape]
            
            axes[stage_idx, 0].imshow(display_data[mid_dims[0], :, :], cmap='gray')
            axes[stage_idx, 0].set_title(f'{stage_names[stage_idx]} - Axial')
            axes[stage_idx, 0].axis('off')
            
            axes[stage_idx, 1].imshow(display_data[:, mid_dims[1], :], cmap='gray')
            axes[stage_idx, 1].set_title(f'Coronal')
            axes[stage_idx, 1].axis('off')
            
            axes[stage_idx, 2].imshow(display_data[:, :, mid_dims[2]], cmap='gray')
            axes[stage_idx, 2].set_title(f'Sagittal')
            axes[stage_idx, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
 
 
if __name__ == "__main__":
    # 使用示例
    config_path = 'config/PatchVolume_data.json'
    
    # 检查配置文件是否存在
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        print("请创建配置文件或修改路径")
        exit(1)
    
    try:
        # 1. 基本预处理可视化
        print("1. 基本预处理步骤可视化 (带增强)")
        visualize_preprocessing_steps(config_path, num_samples=2, stage=1, augmentation=True)
        
        # 2. 无增强对比
        print("\n2. 无增强预处理可视化")
        visualize_preprocessing_steps(config_path, num_samples=2, stage=1, augmentation=False)
        
        # 3. 不同阶段对比
        compare_preprocessing_stages()
        
    except Exception as e:
        print(f"运行出错: {e}")
        print("请检查:")
        print("1. 配置文件路径是否正确")
        print("2. 数据文件是否存在")
        print("3. 是否安装了必要的依赖包 (torchio, matplotlib等)")