"""
COCO数据集下载和解压脚本
下载COCO 2014训练集、验证集和标注文件

注意：annotations_trainval2014.zip 包含以下标注信息：
- instances_train2014.json / instances_val2014.json (物体实例)
- captions_train2014.json / captions_val2014.json (图像描述)
- person_keypoints_train2014.json / person_keypoints_val2014.json (人体关键点)
"""

import os
import urllib.request
import zipfile
from pathlib import Path
import sys


def download_file(url, filepath, backup_urls=None):
    """下载文件并显示进度，支持备用链接"""
    urls_to_try = [url] + (backup_urls or [])
    
    for i, current_url in enumerate(urls_to_try):
        if i > 0:
            print(f"尝试备用链接 {i}: {current_url}")
        else:
            print(f"正在下载: {current_url}")
        print(f"保存到: {filepath}")
        
        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) / total_size)
                sys.stdout.write(f"\r下载进度: {percent:.1f}%")
                sys.stdout.flush()
        
        try:
            urllib.request.urlretrieve(current_url, filepath, reporthook=progress_hook)
            print(f"\n✓ 下载完成: {filepath}")
            return True
        except Exception as e:
            print(f"\n✗ 下载失败: {e}")
            if i < len(urls_to_try) - 1:
                print("尝试下一个链接...")
            continue
    
    return False


def unzip_file(zip_path, extract_to):
    """解压文件"""
    print(f"正在解压: {zip_path}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✓ 解压完成: {zip_path}")
        return True
    except Exception as e:
        print(f"✗ 解压失败: {e}")
        return False


def main():
    """主函数"""
    # 创建data目录
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    print(f"✓ 创建目录: {data_dir}")
    
    # 定义下载文件列表（包含备用链接）
    downloads = [
        {
            "url": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
            "filename": "annotations_trainval2014.zip",
            "backup_urls": [
                "https://github.com/cocodataset/cocoapi/releases/download/v1.0/annotations_trainval2014.zip"
            ]
        },
        {
            "url": "http://images.cocodataset.org/zips/train2014.zip", 
            "filename": "train2014.zip",
            "backup_urls": []
        },
        {
            "url": "http://images.cocodataset.org/zips/val2014.zip",
            "filename": "val2014.zip",
            "backup_urls": []
        }
    ]
    
    # 下载所有文件
    for item in downloads:
        filepath = data_dir / item["filename"]
        
        # 如果文件已存在，询问是否重新下载
        if filepath.exists():
            response = input(f"文件 {filepath} 已存在，是否重新下载? (y/n): ").lower()
            if response != 'y':
                print(f"跳过下载: {filepath}")
                continue
        
        # 下载文件（使用备用链接）
        success = download_file(item["url"], filepath, item.get("backup_urls"))
        if not success:
            print(f"所有链接都下载失败，跳过文件: {item['filename']}")
            continue
    
    print("\n" + "="*50)
    print("开始解压文件...")
    print("="*50)
    
    # 解压所有文件
    zip_files = [
        "annotations_trainval2014.zip",
        "train2014.zip", 
        "val2014.zip"
    ]
    
    for zip_filename in zip_files:
        zip_path = data_dir / zip_filename
        
        if zip_path.exists():
            # 解压文件
            success = unzip_file(zip_path, data_dir)
            
            if success:
                # 删除压缩文件
                try:
                    zip_path.unlink()
                    print(f"✓ 删除压缩文件: {zip_path}")
                except Exception as e:
                    print(f"✗ 删除失败: {e}")
            else:
                print(f"解压失败，保留压缩文件: {zip_path}")
        else:
            print(f"✗ 文件不存在，跳过解压: {zip_path}")
    
    print("\n" + "="*50)
    print("处理完成！")
    print("="*50)
    
    # 显示最终的目录结构
    print("\n数据集目录结构:")
    for item in sorted(data_dir.rglob("*")):
        if item.is_dir():
            print(f"📁 {item.relative_to(data_dir)}/")
        else:
            # 只显示前几个文件，避免输出太长
            relative_path = item.relative_to(data_dir)
            if len(str(relative_path).split('/')) <= 2:  # 只显示前两级
                size_mb = item.stat().st_size / (1024 * 1024)
                if size_mb > 1:  # 只显示大于1MB的文件
                    print(f"📄 {relative_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断下载")
        sys.exit(1)
    except Exception as e:
        print(f"\n发生错误: {e}")
        sys.exit(1)
