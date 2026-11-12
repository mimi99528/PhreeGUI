# 深度纹理保存功能使用说明

## 功能概述

该功能允许在SSFR渲染过程中自动保存深度纹理数据，支持多种保存格式和灵活的配置选项。

## 配置参数

在 `modular_mesh_system/utils/config.py` 中可以配置以下参数：

```python
# 深度纹理保存参数
ENABLE_DEPTH_TEXTURE_SAVE = True      # 是否启用深度纹理保存
DEPTH_SAVE_INTERVAL = 1.0             # 保存间隔（秒）
DEPTH_SAVE_PATH = "./depth_textures"  # 保存目录路径
DEPTH_SAVE_FORMAT = "png"             # 保存格式：png 或 npy
DEPTH_NORMALIZE_RANGE = True          # 是否将深度值归一化到[0,1]范围
```

## 使用方法

### 1. 基本使用

深度纹理保存功能会在SSFR渲染过程中自动触发，无需额外调用。

```python
# 在渲染循环中，系统会自动按配置间隔保存深度纹理
renderer.render(width, height, screen_fbo)
```

### 2. 手动触发保存

如果需要手动保存深度纹理：

```python
# 手动调用保存方法
renderer._save_depth_texture()
```

### 3. 修改保存配置

```python
from modular_mesh_system.utils.config import CONFIG

# 临时修改配置
CONFIG.ENABLE_DEPTH_TEXTURE_SAVE = True
CONFIG.DEPTH_SAVE_INTERVAL = 0.5  # 每0.5秒保存一次
CONFIG.DEPTH_SAVE_FORMAT = "npy"  # 保存为NumPy格式
```

## 保存的文件格式

### PNG格式
- 文件名格式：`depth_frame_000001_20260222_185224.png`
- 8位灰度图像
- 适合可视化查看

### NPY格式  
- 文件名格式：`depth_frame_000001_20260222_185224.npy`
- NumPy数组格式
- 保持原始浮点精度，适合后续处理

## 文件命名规则

```
depth_frame_{帧号:06d}_{时间戳}.{格式}
```

例如：
- `depth_frame_000001_20260222_185224.png`
- `depth_frame_000123_20260222_190015.npy`

## 数据处理

### 归一化模式（推荐）
当 `DEPTH_NORMALIZE_RANGE = True` 时：
- 自动检测有效深度值范围
- 将数据线性映射到 [0,1] 区间
- 提供更好的视觉对比度

### 原始模式
当 `DEPTH_NORMALIZE_RANGE = False` 时：
- 直接将深度值乘以255进行截断
- 可能丢失精度但保持绝对数值关系

## 注意事项

1. **性能影响**：频繁保存会影响渲染性能，建议根据需要调整保存间隔
2. **磁盘空间**：PNG格式每帧约几十KB，NPY格式更大，请监控存储空间
3. **目录权限**：确保程序有写入指定目录的权限
4. **OpenGL上下文**：保存功能需要有效的OpenGL上下文才能正常工作

## 故障排除

### 常见问题

1. **保存目录不存在**
   ```
   解决：检查 DEPTH_SAVE_PATH 配置，确保目录可写
   ```

2. **保存文件为空**
   ```
   解决：确认渲染过程中确实产生了深度数据
   ```

3. **保存速度慢**
   ```
   解决：增大 DEPTH_SAVE_INTERVAL 值，减少保存频率
   ```

### 日志信息

成功保存时会显示：
```
✓ 深度纹理已保存: ./depth_textures/depth_frame_000001_20260222_185224.png
```

失败时会显示错误信息：
```
✗ 深度纹理保存失败: {具体错误信息}
```

## 示例代码

```python
# 完整的使用示例
from modular_mesh_system.renderer.opengl_renderer import OpenGLRenderer
from modular_mesh_system.utils.config import CONFIG

# 配置保存参数
CONFIG.ENABLE_DEPTH_TEXTURE_SAVE = True
CONFIG.DEPTH_SAVE_INTERVAL = 2.0  # 每2秒保存一次
CONFIG.DEPTH_SAVE_FORMAT = "png"
CONFIG.DEPTH_NORMALIZE_RANGE = True

# 创建渲染器（会自动创建保存目录）
renderer = OpenGLRenderer(mesh_system)

# 在渲染循环中正常使用即可
# 深度纹理会按配置自动保存
```

这个功能特别适用于：
- 流体仿真过程记录
- 深度学习训练数据收集
- 渲染调试和分析
- 科学可视化数据导出