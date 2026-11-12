# SSFR厚度感知渲染升级总结

## 概述
本次升级将原有的单通道SSFR系统升级为双通道渲染系统，引入厚度(Thickness)概念来显著改善水体渲染的视觉质量。

## 主要改进

### 1. 双通道架构 (Depth + Thickness)
- **Depth通道**: 维持原有的深度场信息用于法线重建
- **Thickness通道**: 新增厚度信息用于Beer-Lambert吸收计算
- 使用MRT (Multiple Render Targets) 在单次粒子pass中同时写入两个通道

### 2. 着色器升级

#### 粒子Fragment Shader
```glsl
layout(location=0) out float out_depth;  // 深度场: 1.0(背景) -> 0.55(流体)
layout(location=1) out float out_thick;  // 厚度场: 0.0(稀疏) -> 1.0(密集)

// 深度场范围扩大: 1.0 - z * 0.45 (原为0.5)
// 厚度使用球帽z²强调中心，减少环状伪影
```

#### Composite Shader
```glsl
// 引入Beer-Lambert吸收模型
float trans = exp(-u_sigma_a * thick);   // 透射率
float scatter = (1.0 - trans) * u_sigma_s; // 散射增强

// 改进的Alpha混合
alpha = u_alpha_base + (1.0 - trans) * 0.85 + fresnel * 0.25;
```

### 3. 参数优化
- **粒子半径**: 6.0 → 7.5 pixels (增加粒子覆盖)
- **模糊σ值**: 2.0 → 2.3 (更强的平滑效果)
- **吸收系数**: u_sigma_a = 2.2 (真实感吸收)
- **散射系数**: u_sigma_s = 0.55 (前向散射提亮)

### 4. 渲染流程
```
Pass 1: 粒子 → MRT (depth0, thick0)
Pass 2: 水平模糊 depth0→depth1, thick0→thick1  
Pass 3: 垂直模糊 depth1→depth0, thick1→thick0
Pass 4: 复合渲染 → 屏幕输出
```

## 视觉效果提升

### 解决的问题
- ✅ **颗粒感过重** - 厚度模糊消除粒子离散感
- ✅ **过于黑暗** - Beer-Lambert模型提供自然光照
- ✅ **缺乏体积感** - 厚度信息增强水体密度表现
- ✅ **法线噪点** - 扩大的深度范围改善法线质量

### 新增特性
- 🌊 **真实的水体吸收** - 厚地方更暗，薄地方更亮
- ✨ **边缘高光增强** - 菲涅尔效应更加明显
- 💧 **密度感知透明度** - 厚度影响alpha通道
- 🎨 **更好的颜色层次** - 散射提亮内部区域

## 性能考虑
- 使用R16F纹理格式平衡质量和带宽
- 厚度模糊使用较小半径(3)降低开销
- 保持原有3-pass架构，仅增加一个纹理通道

## 使用建议
1. 启用SSFR渲染模式查看效果
2. 可根据需要调整:
   - `ssfr_particle_radius_px`: 控制粒子大小(7.5推荐)
   - `u_sigma_a`: 吸收强度(2.2推荐)
   - `u_sigma_s`: 散射强度(0.55推荐)

## 测试结果
系统已在NVIDIA RTX 4060上验证通过，14079粒子场景运行流畅，视觉质量显著提升。