"""
PhreeqcRM 可视化模块
包含2D动画和3D交互式可视化的实现
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Slider, Button
import matplotlib
import threading
import time as time_module

def create_ph_diffusion_animation(ph_history, nx, ny):
    """
    创建pH值扩散的热点图动画 - 性能优化版本
    提供多种保存选项以解决Pillow/ImageMagick性能问题
    
    Args:
        ph_history (list): pH历史数据列表
        nx (int): X方向网格数
        ny (int): Y方向网格数
    """
    print("\n正在创建可视化动画...")

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建图形和轴
    fig, ax = plt.subplots(figsize=(10, 8))

    # 创建自定义颜色映射（Ph试纸红到黄到紫）
    colors = ['red', 'yellow', 'green', 'blue', 'purple']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('custom_hot', colors, N=n_bins)

    # 获取数据范围 (pH通常在0-14范围内)
    all_data = np.concatenate([data.flatten() for data in ph_history])
    vmin, vmax = np.min(all_data), np.max(all_data)

    # 初始化热力图
    im = ax.imshow(ph_history[0], cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation='nearest', aspect='equal')

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('pH 值', fontsize=12)

    # 添加数值标注
    text_elements = []
    for i in range(ny):
        for j in range(nx):
            text = ax.text(j, i, f'{ph_history[0][i, j]:.1f}',
                          ha="center", va="center", color="white", fontweight='bold', fontsize=6)
            text_elements.append(text)

    # 设置标题和标签
    title = ax.set_title('时间步: 0 (初始状态 - pH分布)', fontsize=14, pad=20)
    ax.set_xlabel('X 坐标', fontsize=12)
    ax.set_ylabel('Y 坐标', fontsize=12)

    # 设置刻度
    ax.set_xticks(range(nx))
    ax.set_yticks(range(ny))
    ax.set_xticklabels([f'{i}' for i in range(nx)])
    ax.set_yticklabels([f'{i}' for i in range(ny)])

    # 添加网格
    ax.set_xticks(np.arange(-.5, nx, 1), minor=True)
    ax.set_yticks(np.arange(-.5, ny, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=1)

    def animate(frame):
        """动画更新函数"""
        # 获取当前帧的数据范围
        current_data = ph_history[frame]
        current_vmin, current_vmax = np.min(current_data), np.max(current_data)

        # 重新设置颜色映射范围
        im.set_clim(vmin=current_vmin, vmax=current_vmax)
        im.set_array(current_data)

        # 更新数值标注
        for i, text_elem in enumerate(text_elements):
            row = i // nx
            col = i % nx
            text_elem.set_text(f'{current_data[row, col]:.1f}')

            # 根据当前帧的背景颜色调整文字颜色以提高可读性
            bg_value = current_data[row, col]
            normalized_value = (bg_value - current_vmin) / (current_vmax - current_vmin)
            if normalized_value > 0.7:  # 背景较亮时用黑色字
                text_elem.set_color('black')
            else:  # 背景较暗时用白色字
                text_elem.set_color('white')

        # 更新标题
        if frame == 0:
            title.set_text('时间步: 0 (初始状态 - pH分布)')
        else:
            title.set_text(f'时间步: {frame} (pH扩散)')

        return [im, title] + text_elements

    # 创建动画对象
    print("正在生成动画对象...")
    anim = animation.FuncAnimation(
        fig, animate, frames=len(ph_history),
        interval=50, blit=True, repeat=True
    )

    # 性能优化的保存选项
    print("正在保存动画文件...")
    
    # 方案1: 使用pillow但降低质量以提高速度
    try:
        print("尝试使用Pillow保存 (低质量快速模式)...")
        # anim.save('ph_diffusion_fast.gif', writer='pillow', fps=10, dpi=50)
        print("✓ 成功保存为 'ph_diffusion_fast.gif' (快速版本)")
    except Exception as e:
        print(f"✗ Pillow保存失败: {e}")
    
    # 方案2: 保存为MP4视频格式 (推荐!)
    try:
        print("尝试保存为MP4视频...")
        anim.save('ph_diffusion.mp4', writer='ffmpeg', fps=15, dpi=100)
        print("✓ 成功保存为 'ph_diffusion.mp4' (高质量视频)")
    except Exception as e:
        print(f"✗ MP4保存失败: {e}")
        print("  提示: 请安装ffmpeg: conda install ffmpeg")
    
    # 方案3: 逐帧保存为PNG图片序列
    try:
        print("正在保存PNG图片序列...")
        import os
        if not os.path.exists('ph_frames'):
            os.makedirs('ph_frames')
        
        # 为PNG序列创建专用的figure，避免影响主动画
        fig_png, ax_png = plt.subplots(figsize=(8, 6))
        
        for i, frame_data in enumerate(ph_history):
            ax_png.clear()
            im_png = ax_png.imshow(frame_data, cmap=cmap, vmin=vmin, vmax=vmax,
                          interpolation='nearest', aspect='equal')
            ax_png.set_title(f'时间步: {i*10} (pH扩散)', fontsize=12)
            ax_png.set_xlabel('X 坐标')
            ax_png.set_ylabel('Y 坐标')
            
            # 只在第一帧添加colorbar，避免重复添加
            if i == 0:
                plt.colorbar(im_png, ax=ax_png)
            
            plt.tight_layout()
            plt.savefig(f'ph_frames/frame_{i:04d}.png', dpi=80, bbox_inches='tight')
            if i % 20 == 0:
                print(f"  已保存 {i}/{len(ph_history)} 帧")
        
        plt.close(fig_png)  # 关闭专用figure
        print("✓ 成功保存PNG序列到 'ph_frames/' 目录")
        print("  可使用以下命令合并为GIF: magick ph_frames/*.png -delay 5 -loop 0 ph_diffusion_magick.gif")
    except Exception as e:
        print(f"✗ PNG序列保存失败: {e}")

    # 方案4: 如果ImageMagick可用，使用它
    try:
        print("尝试使用ImageMagick保存...")
        # anim.save('ph_diffusion_magick.gif', writer='imagemagick', fps=12, dpi=80)
        print("✓ 成功保存为 'ph_diffusion_magick.gif' (ImageMagick版本)")
    except Exception as e:
        print(f"✗ ImageMagick保存失败: {e}")
        print("  提示: 请安装ImageMagick并确保其在PATH中")
    
    # 显示动画
    print("显示动画窗口...")
    plt.tight_layout()
    plt.show()

def create_interactive_3d_visualization(ph_history, nx, ny, nz):
    """
    创建交互式3D剖面图可视化界面
    支持通过滑块控制Z轴切片位置
    
    Args:
        ph_history (list): pH历史数据列表
        nx (int): X方向网格数
        ny (int): Y方向网格数
        nz (int): Z方向网格数
    """
    print("\n正在创建交互式3D可视化界面...")
    
    # 确保使用正确的matplotlib后端
    matplotlib.use('TkAgg')  # 使用TkAgg后端，更适合交互式GUI
    import matplotlib.pyplot as plt
    print("Matplotlib后端设置为:", matplotlib.get_backend())
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图形和子图
    fig = plt.figure(figsize=(12, 8))
    fig.canvas.manager.set_window_title('PhreeqcRM 3D pH扩散可视化')
    print("图形窗口已创建")
    
    # 主显示区域（XY平面切片）
    ax_main = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
    
    # XZ剖面图
    ax_xz = plt.subplot2grid((3, 3), (0, 2))
    
    # YZ剖面图
    ax_yz = plt.subplot2grid((3, 3), (1, 2))
    
    # 控制面板
    ax_slider = plt.subplot2grid((3, 3), (2, 0), colspan=3)
    
    # 创建自定义颜色映射
    colors = ['red', 'yellow', 'green', 'blue', 'purple']
    cmap = LinearSegmentedColormap.from_list('custom_hot', colors, N=256)
    
    # 获取全局数据范围
    all_data = np.concatenate([data.flatten() for data in ph_history])
    vmin, vmax = np.min(all_data), np.max(all_data)
    
    # 初始化显示（中间层）
    initial_z = nz // 2
    initial_frame = 0
    
    # XY平面切片（主视图）
    xy_slice = ph_history[initial_frame][initial_z, :, :]
    im_xy = ax_main.imshow(xy_slice, cmap=cmap, vmin=vmin, vmax=vmax,
                          interpolation='nearest', aspect='equal')
    ax_main.set_title(f'XY平面切片 (Z={initial_z})', fontsize=12)
    ax_main.set_xlabel('X 坐标')
    ax_main.set_ylabel('Y 坐标')
    
    # 添加颜色条
    cbar = plt.colorbar(im_xy, ax=ax_main, shrink=0.8)
    cbar.set_label('pH 值')
    
    # XZ剖面图
    xz_slice = ph_history[initial_frame][:, initial_z//2, :]
    im_xz = ax_xz.imshow(xz_slice, cmap=cmap, vmin=vmin, vmax=vmax,
                         interpolation='nearest', aspect='auto')
    ax_xz.set_title(f'XZ剖面 (Y={initial_z//2})')
    ax_xz.set_xlabel('X')
    ax_xz.set_ylabel('Z')
    
    # YZ剖面图
    yz_slice = ph_history[initial_frame][:, :, initial_z//2]
    im_yz = ax_yz.imshow(yz_slice, cmap=cmap, vmin=vmin, vmax=vmax,
                         interpolation='nearest', aspect='auto')
    ax_yz.set_title(f'YZ剖面 (X={initial_z//2})')
    ax_yz.set_xlabel('Y')
    ax_yz.set_ylabel('Z')
    
    # 时间滑块 (底部水平)
    ax_time = plt.axes([0.1, 0.02, 0.5, 0.03])
    time_slider = Slider(ax_time, '时间步', 0, len(ph_history)-1, 
                        valinit=0, valfmt='%d')
    
    # Z轴切片滑块 (底部水平，紧接时间滑块)
    z_slider_ax = plt.axes([0.1, 0.06, 0.5, 0.03])
    z_slider = Slider(z_slider_ax, 'Z层', 0, nz-1, 
                     valinit=initial_z, valfmt='%d')
    
    # Y轴切片滑块 (右侧垂直)
    y_slider_ax = plt.axes([0.85, 0.2, 0.03, 0.4])
    y_slider = Slider(y_slider_ax, 'Y层', 0, ny-1, 
                     valinit=initial_z//2, valfmt='%d', orientation='vertical')
    
    # X轴切片滑块 (顶部水平)
    x_slider_ax = plt.axes([0.2, 0.85, 0.4, 0.03])
    x_slider = Slider(x_slider_ax, 'X层', 0, nx-1, 
                     valinit=initial_z//2, valfmt='%d')
    
    def update_display(val):
        """更新所有显示"""
        frame_idx = int(time_slider.val)
        z_idx = int(z_slider.val)
        y_idx = int(y_slider.val)
        x_idx = int(x_slider.val)
        
        # 获取当前帧数据
        current_data = ph_history[frame_idx]
        
        # 更新XY切片
        xy_slice = current_data[z_idx, :, :]
        im_xy.set_array(xy_slice)
        ax_main.set_title(f'XY平面切片 (Z={z_idx}, 时间步={frame_idx})', fontsize=12)
        
        # 更新XZ切片
        xz_slice = current_data[:, y_idx, :]
        im_xz.set_array(xz_slice)
        ax_xz.set_title(f'XZ剖面 (Y={y_idx})')
        
        # 更新YZ切片
        yz_slice = current_data[:, :, x_idx]
        im_yz.set_array(yz_slice)
        ax_yz.set_title(f'YZ剖面 (X={x_idx})')
        
        # 刷新图形
        fig.canvas.draw_idle()
    
    # 连接滑块事件
    time_slider.on_changed(update_display)
    z_slider.on_changed(update_display)
    y_slider.on_changed(update_display)
    x_slider.on_changed(update_display)
    
    # 添加播放按钮功能
    play_ax = plt.axes([0.85, 0.02, 0.1, 0.04])
    play_button = Button(play_ax, '播放')
    
    playing = False
    
    def toggle_play(event):
        nonlocal playing
        playing = not playing
        play_button.label.set_text('暂停' if playing else '播放')
        
        if playing:
            def auto_play():
                current_frame = int(time_slider.val)
                while playing and current_frame < len(ph_history) - 1:
                    current_frame += 1
                    time_slider.set_val(current_frame)
                    time_module.sleep(0.1)  # 控制播放速度
                
            thread = threading.Thread(target=auto_play)
            thread.daemon = True
            thread.start()
    
    play_button.on_clicked(toggle_play)
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, right=0.82, top=0.82, left=0.1)
    
    print("交互式3D可视化界面已准备就绪！")
    print("使用说明：")
    print("- 上方主图为XY平面切片")
    print("- 右侧为XZ和YZ剖面图")
    print("- 底部滑块控制时间和Z轴切片位置")
    print("- 右侧垂直滑块控制Y轴切片位置")
    print("- 顶部滑块控制X轴切片位置")
    print("- 底部按钮可以自动播放动画")
    print("\n注意：如果窗口没有显示，请检查：")
    print("1. PyCharm的运行配置是否阻止了GUI显示")
    print("2. 是否需要在终端中直接运行程序")
    print("3. matplotlib后端是否正确设置")
    
    print("正在显示图形窗口...")
    plt.ion()  # 开启交互模式
    plt.show(block=True)  # 阻塞模式显示
    print("图形窗口已关闭")