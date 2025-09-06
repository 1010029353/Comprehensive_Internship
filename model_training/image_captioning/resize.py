"""
图像尺寸调整工具

该模块用于批量调整图像尺寸，主要用于深度学习模型的数据预处理。
主要功能包括：
1. 单张图像尺寸调整
2. 批量图像处理
3. 保持图像质量的高质量缩放
4. 进度显示和状态监控

适用于COCO数据集等大规模图像数据的预处理工作。
"""

import argparse
import os
from PIL import Image


def resize_single_image(image, target_size):
    """
    调整单张图像的尺寸
    
    使用高质量的LANCZOS算法进行图像缩放，能够保持较好的图像质量。
    
    Args:
        image (PIL.Image): 待调整的PIL图像对象
        target_size (tuple): 目标尺寸，格式为(width, height)
        
    Returns:
        PIL.Image: 调整尺寸后的图像对象
    """
    return image.resize(target_size, Image.LANCZOS)


def validate_directories(image_dir, output_dir):
    """
    验证输入和输出目录
    
    Args:
        image_dir (str): 输入图像目录路径
        output_dir (str): 输出图像目录路径
        
    Raises:
        FileNotFoundError: 当输入目录不存在时抛出异常
    """
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"输入图像目录不存在: {image_dir}")
    
    if not os.path.exists(output_dir):
        print(f"创建输出目录: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)


def get_image_files(image_dir):
    """
    获取目录中的所有图像文件
    
    Args:
        image_dir (str): 图像目录路径
        
    Returns:
        list: 图像文件名列表
    """
    # 支持的图像格式
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    all_files = os.listdir(image_dir)
    image_files = [
        f for f in all_files 
        if os.path.splitext(f.lower())[1] in supported_formats
    ]
    
    print(f"在目录 '{image_dir}' 中找到 {len(image_files)} 张图像")
    return image_files


def process_single_image(image_path, output_path, target_size):
    """
    处理单张图像
    
    Args:
        image_path (str): 输入图像路径
        output_path (str): 输出图像路径
        target_size (tuple): 目标尺寸
        
    Returns:
        bool: 处理成功返回True，失败返回False
    """
    try:
        with open(image_path, 'r+b') as f:
            with Image.open(f) as img:
                # 调整图像尺寸
                resized_img = resize_single_image(img, target_size)
                
                # 保存调整后的图像
                resized_img.save(output_path, img.format)
                
        return True
    except Exception as e:
        print(f"处理图像失败 '{image_path}': {str(e)}")
        return False


def batch_resize_images(image_dir, output_dir, target_size):
    """
    批量调整图像尺寸
    
    遍历输入目录中的所有图像文件，调整其尺寸后保存到输出目录。
    每处理100张图像会显示一次进度。
    
    Args:
        image_dir (str): 输入图像目录路径
        output_dir (str): 输出图像目录路径
        target_size (tuple): 目标尺寸，格式为(width, height)
    """
    # 验证目录
    validate_directories(image_dir, output_dir)
    
    # 获取图像文件列表
    image_files = get_image_files(image_dir)
    
    if not image_files:
        print("未找到任何图像文件！")
        return
    
    print(f"开始批量调整图像尺寸到 {target_size[0]}x{target_size[1]}")
    print("-" * 50)
    
    # 统计处理结果
    success_count = 0
    total_count = len(image_files)
    
    # 批量处理图像
    for i, image_filename in enumerate(image_files):
        input_path = os.path.join(image_dir, image_filename)
        output_path = os.path.join(output_dir, image_filename)
        
        # 处理单张图像
        if process_single_image(input_path, output_path, target_size):
            success_count += 1
        
        # 每处理100张图像显示一次进度
        if (i + 1) % 100 == 0:
            print(f"已处理 [{i+1}/{total_count}] 张图像，保存到 '{output_dir}'")
    
    # 显示最终结果
    print("-" * 50)
    print(f"批量处理完成！")
    print(f"成功处理: {success_count}/{total_count} 张图像")
    print(f"输出目录: {output_dir}")


def parse_arguments():
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(
        description='批量调整图像尺寸工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python resize.py --image_dir ./data/train2014/ --output_dir ./data/resized2014/ --image_size 256
  python resize.py --image_dir ./images/ --output_dir ./resized_images/ --image_size 512
        """
    )
    
    parser.add_argument(
        '--image_dir', 
        type=str, 
        default='./data/train2014/',
        help='输入图像目录路径 (默认: ./data/train2014/)'
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='./data/resized2014/',
        help='输出图像目录路径 (默认: ./data/resized2014/)'
    )
    
    parser.add_argument(
        '--image_size', 
        type=int, 
        default=256,
        help='调整后的图像尺寸（正方形） (默认: 256)'
    )
    
    return parser.parse_args()


def main(args):
    """
    主函数：执行图像尺寸调整的完整流程
    
    Args:
        args: 命令行参数对象
    """
    print("=" * 60)
    print("图像尺寸批量调整工具")
    print("=" * 60)
    
    # 准备参数
    image_dir = args.image_dir
    output_dir = args.output_dir
    target_size = (args.image_size, args.image_size)  # 创建正方形尺寸
    
    print(f"输入目录: {image_dir}")
    print(f"输出目录: {output_dir}")
    print(f"目标尺寸: {target_size[0]}x{target_size[1]}")
    print()
    
    # 执行批量调整
    batch_resize_images(image_dir, output_dir, target_size)
    
    print("=" * 60)
    print("图像尺寸调整完成！")
    print("=" * 60)


if __name__ == '__main__':
    # 解析命令行参数
    args = parse_arguments()
    
    # 执行主程序
    main(args)