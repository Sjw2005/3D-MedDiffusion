#!/usr/bin/env python3
"""
DICOM to NIfTI 高性能批量转换器 (修复版)
解决DICOM方向矩阵和倾斜扫描问题
"""
 
# ================================ 配置参数 ================================
# 📁 输入输出路径
INPUT_DIR = "CQ-500/CQ500CT0 CQ500CT0"          # DICOM文件根目录
OUTPUT_DIR = "dataproout"                        # NIfTI输出目录
 
# 🔧 转换参数
MIN_DICOM_FILES = 10        # 文件夹最少DICOM文件数才进行转换
OVERWRITE_EXISTING = True   # 是否覆盖已存在的文件
REORIENT_NIFTI = False     # 关闭自动重定向，避免方向问题
ADD_TIMESTAMP = False      # 文件名是否添加时间戳
 
# 📊 质量检查参数  
QUALITY_THRESHOLD = 60     # 质量得分阈值 (0-100)
MAX_ZERO_RATIO = 0.9      # 最大零值占比阈值
ENABLE_QUALITY_CHECK = True # 是否启用质量检查
 
# 🏷️ 文件命名策略 ('folder_name', 'folder_path', 'dicom_metadata')
NAMING_STRATEGY = 'folder_name'
 
# 🛠️ 高级参数
USE_SITK_FALLBACK = True   # 使用SimpleITK作为备用转换方案
FORCE_RESAMPLING = True    # 强制重采样为正交方向
TARGET_SPACING = None      # 目标体素间距 [x,y,z] mm，None为保持原始
IGNORE_ORIENTATION = True  # 忽略方向信息，强制转换
 
# 📝 输出控制
VERBOSE = True             # 是否显示详细输出
SAVE_FAILED_LIST = True    # 是否保存失败文件列表
# =========================================================================
 
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings("ignore")
 
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from tqdm import tqdm
 
# 尝试导入dicom2nifti，如果失败就用SimpleITK
try:
    import dicom2nifti
    HAS_DICOM2NIFTI = True
except ImportError:
    HAS_DICOM2NIFTI = False
    print("⚠️ dicom2nifti未安装，将使用SimpleITK转换")
 
class RobustDicomConverter:
    def __init__(self):
        self.stats = {
            'total': 0, 'converted': 0, 'failed': 0, 'skipped': 0,
            'failed_files': [], 'low_quality_files': [], 'conversion_methods': {}
        }
        
    def find_dicom_folders(self) -> List[Path]:
        """快速查找DICOM文件夹"""
        dicom_folders = []
        root_path = Path(INPUT_DIR)
        
        if VERBOSE:
            print(f"🔍 扫描DICOM文件夹: {INPUT_DIR}")
        
        # 递归查找包含DICOM文件的文件夹
        for folder in root_path.rglob('*'):
            if folder.is_dir():
                dicom_files = []
                for ext in ['*.dcm', '*.DCM', '*.dicom', '*.DICOM', '*.ima', '*.IMA']:
                    dicom_files.extend(list(folder.glob(ext)))
                
                # 也检查没有扩展名的DICOM文件
                for file in folder.iterdir():
                    if file.is_file() and not file.suffix:
                        try:
                            # 尝试读取DICOM header
                            sitk.ReadImage(str(file))
                            dicom_files.append(file)
                        except:
                            continue
                
                if len(dicom_files) >= MIN_DICOM_FILES:
                    dicom_folders.append(folder)
        
        if VERBOSE:
            print(f"📁 找到 {len(dicom_folders)} 个符合条件的DICOM文件夹")
        return dicom_folders
    
    def generate_filename(self, dicom_folder: Path) -> str:
        """生成输出文件名"""
        if NAMING_STRATEGY == 'folder_name':
            base_name = dicom_folder.name
        elif NAMING_STRATEGY == 'folder_path':
            relative_path = dicom_folder.relative_to(Path(INPUT_DIR))
            base_name = str(relative_path).replace(os.sep, '_')
        elif NAMING_STRATEGY == 'dicom_metadata':
            base_name = self._extract_metadata_name(dicom_folder)
        else:
            base_name = dicom_folder.name
            
        # 清理文件名，保留安全字符
        safe_name = "".join(c for c in base_name if c.isalnum() or c in ('-', '_', '.'))
        
        # 添加时间戳
        if ADD_TIMESTAMP:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = f"{safe_name}_{timestamp}"
            
        return f"{safe_name}.nii.gz"
    
    def _extract_metadata_name(self, dicom_folder: Path) -> str:
        """从DICOM元数据提取患者信息"""
        try:
            dicom_files = []
            for ext in ['*.dcm', '*.DCM', '*.dicom', '*.DICOM']:
                dicom_files.extend(list(dicom_folder.glob(ext)))
            
            if not dicom_files:
                # 尝试没有扩展名的文件
                for file in dicom_folder.iterdir():
                    if file.is_file() and not file.suffix:
                        dicom_files.append(file)
            
            if not dicom_files:
                return dicom_folder.name
                
            reader = sitk.ImageFileReader()
            reader.SetFileName(str(dicom_files[0]))
            reader.LoadPrivateTagsOn()
            reader.ReadImageInformation()
            
            patient_id = reader.GetMetaData("0010|0020") if reader.HasMetaDataKey("0010|0020") else "unknown"
            series_desc = reader.GetMetaData("0008|103e") if reader.HasMetaDataKey("0008|103e") else "unknown"
            
            # 清理元数据
            patient_id = "".join(c for c in patient_id if c.isalnum() or c in ('-', '_'))[:20]
            series_desc = "".join(c for c in series_desc if c.isalnum() or c in ('-', '_'))[:30]
            
            return f"{patient_id}_{series_desc}"
            
        except Exception:
            return dicom_folder.name
    
    def convert_with_dicom2nifti(self, dicom_folder: Path, output_file: Path) -> Tuple[bool, str]:
        """使用dicom2nifti转换"""
        try:
            dicom2nifti.dicom_series_to_nifti(
                str(dicom_folder),
                str(output_file),
                reorient_nifti=REORIENT_NIFTI
            )
            return True, "dicom2nifti转换成功"
        except Exception as e:
            return False, f"dicom2nifti转换失败: {str(e)}"
    
    def convert_with_sitk(self, dicom_folder: Path, output_file: Path) -> Tuple[bool, str]:
        """使用SimpleITK转换（更强的兼容性）"""
        try:
            # 读取DICOM序列
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_folder))
            
            if not dicom_names:
                return False, "未找到DICOM文件"
            
            reader.SetFileNames(dicom_names)
            reader.MetaDataDictionaryArrayUpdateOn()
            reader.LoadPrivateTagsOn()
            
            # 读取图像
            image = reader.Execute()
            
            # 处理方向问题
            if IGNORE_ORIENTATION:
                # 重置为标准方向矩阵
                image.SetDirection([1,0,0, 0,1,0, 0,0,1])
            
            # 强制重采样（如果需要）
            if FORCE_RESAMPLING:
                image = self._resample_to_orthogonal(image)
            
            # 目标体素间距重采样
            if TARGET_SPACING:
                image = self._resample_spacing(image, TARGET_SPACING)
            
            # 写入NIfTI
            sitk.WriteImage(image, str(output_file))
            return True, "SimpleITK转换成功"
            
        except Exception as e:
            return False, f"SimpleITK转换失败: {str(e)}"
    
    def _resample_to_orthogonal(self, image):
        """重采样为正交方向"""
        try:
            # 获取原始信息
            original_spacing = image.GetSpacing()
            original_size = image.GetSize()
            original_origin = image.GetOrigin()
            
            # 设置标准方向矩阵
            new_direction = [1,0,0, 0,1,0, 0,0,1]
            
            # 创建重采样器
            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputDirection(new_direction)
            resampler.SetOutputOrigin(original_origin)
            resampler.SetOutputSpacing(original_spacing)
            resampler.SetSize(original_size)
            resampler.SetInterpolator(sitk.sitkLinear)
            
            return resampler.Execute(image)
        except:
            return image
    
    def _resample_spacing(self, image, target_spacing):
        """重采样到目标体素间距"""
        try:
            original_spacing = image.GetSpacing()
            original_size = image.GetSize()
            
            # 计算新尺寸
            new_size = [
                int(round(original_size[i] * original_spacing[i] / target_spacing[i]))
                for i in range(3)
            ]
            
            # 重采样
            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputSpacing(target_spacing)
            resampler.SetSize(new_size)
            resampler.SetOutputDirection(image.GetDirection())
            resampler.SetOutputOrigin(image.GetOrigin())
            resampler.SetInterpolator(sitk.sitkLinear)
            
            return resampler.Execute(image)
        except:
            return image
    
    def convert_single_folder(self, dicom_folder: Path, output_dir: Path) -> Tuple[bool, str, Dict]:
        """转换单个DICOM文件夹（多方法尝试）"""
        output_filename = self.generate_filename(dicom_folder)
        output_file = output_dir / output_filename
        
        # 检查是否跳过已存在文件
        if output_file.exists() and not OVERWRITE_EXISTING:
            return False, "已存在", {'skipped': True}
        
        conversion_methods = []
        
        # 方法1: 尝试dicom2nifti（如果可用且启用）
        if HAS_DICOM2NIFTI and not IGNORE_ORIENTATION:
            success, message = self.convert_with_dicom2nifti(dicom_folder, output_file)
            conversion_methods.append(('dicom2nifti', success, message))
            if success:
                return True, message, {
                    'output_file': str(output_file),
                    'method': 'dicom2nifti',
                    'quality': self.quick_quality_check(output_file) if ENABLE_QUALITY_CHECK else {}
                }
        
        # 方法2: 使用SimpleITK（更强兼容性）
        if USE_SITK_FALLBACK:
            success, message = self.convert_with_sitk(dicom_folder, output_file)
            conversion_methods.append(('SimpleITK', success, message))
            if success:
                return True, message, {
                    'output_file': str(output_file),
                    'method': 'SimpleITK',
                    'quality': self.quick_quality_check(output_file) if ENABLE_QUALITY_CHECK else {}
                }
        
        # 所有方法都失败
        error_details = "; ".join([f"{method}: {msg}" for method, success, msg in conversion_methods if not success])
        return False, f"所有转换方法失败: {error_details}", {'conversion_attempts': conversion_methods}
    
    def quick_quality_check(self, nifti_file: Path) -> Dict:
        """快速质量检查"""
        try:
            img = nib.load(str(nifti_file))
            data = img.get_fdata()
            
            # 基本信息
            file_size_mb = nifti_file.stat().st_size / (1024 * 1024)
            shape = data.shape
            data_range = [float(data.min()), float(data.max())]
            zero_ratio = float(np.sum(data == 0) / data.size)
            
            # 问题检测
            issues = []
            if np.isnan(data).any():
                issues.append("含NaN值")
            if np.isinf(data).any():
                issues.append("含无穷值")
            if zero_ratio > MAX_ZERO_RATIO:
                issues.append(f"零值占比过高({zero_ratio:.1%})")
            if len(shape) != 3:
                issues.append("非3D数据")
            if file_size_mb < 0.1:
                issues.append("文件过小")
            
            # 计算质量得分
            quality_score = 100.0
            if issues:
                quality_score -= len(issues) * 20
            quality_score = max(0, quality_score)
            
            return {
                'file_size_mb': round(file_size_mb, 2),
                'shape': list(shape),
                'data_range': [round(data_range[0], 2), round(data_range[1], 2)],
                'zero_ratio': round(zero_ratio, 3),
                'issues': issues,
                'quality_score': quality_score,
                'is_good_quality': quality_score >= QUALITY_THRESHOLD
            }
            
        except Exception as e:
            return {'error': str(e), 'quality_score': 0, 'is_good_quality': False}
    
    def batch_convert(self):
        """批量转换主函数"""
        # 验证输入目录
        if not Path(INPUT_DIR).exists():
            print(f"❌ 输入目录不存在: {INPUT_DIR}")
            return
        
        # 创建输出目录
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 显示配置信息
        if VERBOSE:
            print(f"🚀 开始批量转换 DICOM → NIfTI (修复版)")
            print(f"📂 输入目录: {INPUT_DIR}")
            print(f"📁 输出目录: {OUTPUT_DIR}")
            print(f"🔧 最少文件数: {MIN_DICOM_FILES}")
            print(f"🏷️  命名策略: {NAMING_STRATEGY}")
            print(f"📊 质量检查: {'启用' if ENABLE_QUALITY_CHECK else '禁用'}")
            print(f"🛠️  转换策略: SimpleITK主导，忽略方向={'是' if IGNORE_ORIENTATION else '否'}")
            print("-" * 60)
        
        # 查找DICOM文件夹
        dicom_folders = self.find_dicom_folders()
        self.stats['total'] = len(dicom_folders)
        
        if not dicom_folders:
            print("❌ 未找到符合条件的DICOM文件夹")
            return
        
        # 批量转换
        print(f"🔄 开始转换 {len(dicom_folders)} 个文件夹...")
        
        progress_bar = tqdm(dicom_folders, desc="转换进度", 
                           disable=not VERBOSE, unit="folder")
        
        for dicom_folder in progress_bar:
            success, message, info = self.convert_single_folder(dicom_folder, output_dir)
            
            # 更新进度条描述
            if VERBOSE:
                folder_name = dicom_folder.name[:20] + "..." if len(dicom_folder.name) > 20 else dicom_folder.name
                progress_bar.set_postfix_str(f"当前: {folder_name}")
            
            # 统计结果
            if success:
                self.stats['converted'] += 1
                method = info.get('method', 'unknown')
                self.stats['conversion_methods'][method] = self.stats['conversion_methods'].get(method, 0) + 1
                
                # 检查质量
                if ENABLE_QUALITY_CHECK and info.get('quality'):
                    if not info['quality'].get('is_good_quality', True):
                        self.stats['low_quality_files'].append(info['output_file'])
            elif info.get('skipped'):
                self.stats['skipped'] += 1
            else:
                self.stats['failed'] += 1
                self.stats['failed_files'].append(str(dicom_folder))
        
        # 显示结果
        self.show_results()
        
        # 保存失败列表
        if SAVE_FAILED_LIST and self.stats['failed_files']:
            self.save_failed_list(output_dir)
    
    def show_results(self):
        """显示转换结果"""
        print(f"\n{'='*60}")
        print(f"📈 转换完成统计:")
        print(f"   总文件夹数: {self.stats['total']}")
        print(f"   转换成功: {self.stats['converted']} ✅")
        print(f"   转换失败: {self.stats['failed']} ❌")
        print(f"   跳过文件: {self.stats['skipped']} ⏭️")
        
        if self.stats['conversion_methods']:
            print(f"   转换方法统计: {dict(self.stats['conversion_methods'])}")
        
        if ENABLE_QUALITY_CHECK and self.stats['low_quality_files']:
            print(f"   低质量文件: {len(self.stats['low_quality_files'])} ⚠️")
        
        success_rate = (self.stats['converted'] / self.stats['total'] * 100) if self.stats['total'] > 0 else 0
        print(f"   成功率: {success_rate:.1f}%")
        
        # 显示部分失败文件
        if self.stats['failed'] > 0:
            print(f"\n❌ 失败的文件夹 (显示前5个):")
            for i, failed_file in enumerate(self.stats['failed_files'][:5]):
                print(f"   {i+1}. {Path(failed_file).name}")
            if len(self.stats['failed_files']) > 5:
                print(f"   ... 还有 {len(self.stats['failed_files']) - 5} 个失败")
        
        print(f"{'='*60}")
        print(f"✅ 转换完成！输出目录: {OUTPUT_DIR}")
    
    def save_failed_list(self, output_dir: Path):
        """保存失败文件列表"""
        failed_file = output_dir / f"failed_conversions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(failed_file, 'w', encoding='utf-8') as f:
            f.write(f"转换失败的DICOM文件夹列表\n")
            f.write(f"生成时间: {datetime.now()}\n")
            f.write(f"{'='*50}\n\n")
            
            for failed_folder in self.stats['failed_files']:
                f.write(f"{failed_folder}\n")
        
        if VERBOSE:
            print(f"📝 失败列表已保存: {failed_file}")
 
def check_dependencies():
    """检查依赖包"""
    required_packages = ['nibabel', 'SimpleITK', 'tqdm', 'numpy']
    optional_packages = ['dicom2nifti']
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_required.append(package)
    
    for package in optional_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_optional.append(package)
    
    if missing_required:
        print(f"❌ 缺少必需依赖包: {', '.join(missing_required)}")
        print(f"请运行: pip install {' '.join(missing_required)}")
        return False
    
    if missing_optional:
        print(f"⚠️ 缺少可选依赖包: {', '.join(missing_optional)}")
        print(f"建议安装: pip install {' '.join(missing_optional)}")
    
    return True
 
def main():
    """主函数"""
    print("🔬 DICOM to NIfTI 高性能批量转换器 (修复版)")
    print("=" * 60)
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 执行转换
    converter = RobustDicomConverter()
    converter.batch_convert()
 
if __name__ == "__main__":
    main()