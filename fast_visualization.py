"""
PhreeqcRM 快速可视化工具
解决Pillow/ImageMagick性能问题的替代方案
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
from PIL import Image

def create_fast_gif_from_data(ph_history, nx, ny, output_filename='ph_diffusion_fast.gif'):
    """
    使用PIL直接创建GIF，绕过matplotlib动画的性能问题
    """
    print("正在使用快速GIF生成方法...")
    
    # 创建自定义颜色映射
    colors = ['red', 'yellow', 'green', 'blue', 'purple']
    cmap = LinearSegmentedColormap.from_list('custom_hot', colors, N=256)
    
    # 获取全局数据范围
    all_data = np.concatenate([data.flatten() for data in ph_history])
    vmin, vmax = np.min(all_data), np.max(all_data)
    
    # 生成图像帧
    frames = []
    print(f"正在生成 {len(ph_history)} 帧图像...")
    
    for i, ph_data in enumerate(ph_history):
        # 创建高分辨率图像
        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        
        # 绘制热力图
        im = ax.imshow(ph_data, cmap=cmap, vmin=vmin, vmax=vmax,
                      interpolation='nearest', aspect='equal')
        
        # 添加标题和标签
        ax.set_title(f'时间步: {i*10} (pH扩散)', fontsize=12, pad=15)
        ax.set_xlabel('X 坐标', fontsize=10)
        ax.set_ylabel('Y 坐标', fontsize=10)
        
        # 简化颜色条
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('pH 值', fontsize=10)
        
        # 设置刻度
        ax.set_xticks(range(0, nx, 2))
        ax.set_yticks(range(0, ny, 2))
        
        plt.tight_layout()
        
        # 保存到内存
        fig.canvas.draw()
        image_array = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)
        
        # 转换为PIL图像
        pil_image = Image.fromarray(image_array)
        # 转换为RGB（去除alpha通道）
        pil_image = pil_image.convert('RGB')
        frames.append(pil_image)
        
        if i % 20 == 0:
            print(f"  已处理 {i}/{len(ph_history)} 帧")
    
    # 保存GIF
    print("正在保存GIF文件...")
    if frames:
        frames[0].save(
            output_filename,
            save_all=True,
            append_images=frames[1:],
            duration=100,  # 100ms per frame = 10fps
            loop=0,  # 无限循环
            optimize=True
        )
        print(f"✓ 成功保存为 '{output_filename}'")
        print(f"  帧数: {len(frames)}, 尺寸: {frames[0].size}")
    else:
        print("✗ 没有生成任何帧")

def create_mp4_from_data(ph_history, nx, ny, output_filename='ph_diffusion.mp4'):
    """
    使用opencv创建MP4视频（如果可用）
    """
    try:
        import cv2
        print("正在使用OpenCV创建MP4视频...")
        
        # 创建自定义颜色映射
        colors = ['red', 'yellow', 'green', 'blue', 'purple']
        cmap = LinearSegmentedColormap.from_list('custom_hot', colors, N=256)
        
        # 获取全局数据范围
        all_data = np.concatenate([data.flatten() for data in ph_history])
        vmin, vmax = np.min(all_data), np.max(all_data)
        
        # 视频参数
        width, height = 800, 600
        fps = 15
        
        # 创建视频编写器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
        
        print(f"正在生成 {len(ph_history)} 帧视频...")
        
        for i, ph_data in enumerate(ph_history):
            # 创建matplotlib图像
            fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
            
            im = ax.imshow(ph_data, cmap=cmap, vmin=vmin, vmax=vmax,
                          interpolation='nearest', aspect='equal')
            ax.set_title(f'时间步: {i*10} (pH扩散)', fontsize=12)
            ax.set_xlabel('X 坐标')
            ax.set_ylabel('Y 坐标')
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            
            # 转换为OpenCV格式
            fig.canvas.draw()
            image_array = np.array(fig.canvas.renderer.buffer_rgba())
            plt.close(fig)
            
            # 转换RGBA到BGR
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
            
            # 调整大小
            image_resized = cv2.resize(image_bgr, (width, height))
            
            # 写入视频
            video_writer.write(image_resized)
            
            if i % 20 == 0:
                print(f"  已处理 {i}/{len(ph_history)} 帧")
        
        video_writer.release()
        print(f"✓ 成功保存为 '{output_filename}'")
        
    except ImportError:
        print("OpenCV未安装，跳过MP4生成")
        print("安装命令: pip install opencv-python")

def load_and_visualize_simulation_data():
    """
    加载模拟数据并创建可视化
    """
    print("=" * 50)
    print("PhreeqcRM 快速可视化工具")
    print("=" * 50)
    
    # 这里应该加载你的模拟数据
    # 作为示例，我创建一些测试数据
    print("生成测试数据...")
    nx, ny = 10, 10
    n_frames = 50
    
    # 生成测试的pH数据序列
    ph_history = []
    for frame in range(n_frames):
        # 创建随时间变化的pH分布
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y)
        
        # 随时间扩散的高斯分布
        center_x = 0.3 + 0.4 * (frame / n_frames)
        center_y = 0.3 + 0.4 * (frame / n_frames)
        sigma = 0.2 + 0.3 * (frame / n_frames)
        
        ph_field = 7.0 + 4.0 * np.exp(-((X - center_x)**2 + (Y - center_y)**2) / (2 * sigma**2))
        ph_history.append(ph_field)
    
    print(f"生成了 {len(ph_history)} 帧测试数据")
    
    # 创建可视化
    create_fast_gif_from_data(ph_history, nx, ny)
    create_mp4_from_data(ph_history, nx, ny)

if __name__ == "__main__":
    load_and_visualize_simulation_data()