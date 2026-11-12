# SSFR渲染器核心Bug修复说明

## 问题描述

在 `_render_ssfr` 方法中存在一个关键的逻辑错误，导致双边滤波结果可能无法正确传递到最终合成阶段。

### 核心问题

1. **乒乓渲染逻辑混乱**: 原始代码中的乒乓缓冲逻辑在迭代次数为奇数时会导致最终有效数据存储在错误的纹理中
2. **9x9精炼阶段硬编码**: 9x9核精炼总是从 `depth_tex1` 读取并写入 `depth_fbo0`，这可能覆盖前面迭代的正确结果
3. **合成阶段固定使用**: 合成阶段始终使用 `depth_tex0`，但该纹理可能包含未滤波或中间结果

### 具体表现

当 `ssfr_filter_intervals` 长度为奇数时：
- 最终滤波结果存储在 `depth_tex1` 中
- 但合成阶段使用 `depth_tex0`（可能包含旧数据或未处理数据）
- 导致渲染质量下降或出现视觉错误

## 修复方案

### 1. 重构滤波逻辑

创建了独立的 `_filter_depth_passes()` 方法，实现清晰的乒乓渲染逻辑：

```python
def _filter_depth_passes(self):
    """执行双边滤波，返回最终的深度纹理"""
    # 初始化乒乓缓冲
    src_tex = self.depth_tex0
    dst_fbo = self.depth_fbo1
    
    # 执行5x5核的滤波迭代
    for i, interval in enumerate(self.ssfr_filter_intervals):
        # 渲染到目标FBO
        dst_fbo.use()
        # ... 设置参数和渲染 ...
        
        # 乒乓交换
        src_tex, dst_fbo = (
            self.depth_tex1 if src_tex == self.depth_tex0 else self.depth_tex0,
            self.depth_fbo0 if dst_fbo == self.depth_fbo1 else self.depth_fbo1
        )
    
    # 9x9核精炼
    # ... 处理 ...
    
    # 确保最终结果在depth_tex0中
    if len(self.ssfr_filter_intervals) % 2 == 1:
        # 奇数次迭代，需要拷贝
        # ... 执行1x1核拷贝 ...
    
    return final_tex
```

### 2. 关键改进点

#### a) 清晰的乒乓逻辑
- 使用明确的状态变量跟踪当前源纹理和目标FBO
- 每次迭代后正确交换状态
- 避免基于迭代索引的模运算判断

#### b) 最终结果保证
- 检查迭代次数奇偶性
- 当为奇数时，执行额外的拷贝操作确保结果在 `depth_tex0`
- 使用1x1核的双边滤波作为高效拷贝手段

#### c) 正确的9x9精炼
- 根据实际的当前源纹理选择输入
- 避免硬编码的纹理引用

### 3. 调用方式变更

原代码：
```python
# 原始的内联滤波逻辑
for i, interval in enumerate(self.ssfr_filter_intervals):
    # ... 复杂的乒乓逻辑 ...
# 9x9精炼硬编码使用depth_tex1
# 合成阶段固定使用depth_tex0
```

修复后：
```python
# 调用封装的方法
final_depth_tex = self._filter_depth_passes()
# 合成阶段使用返回的正确纹理
final_depth_tex.use(location=0)
```

## 测试验证

创建了 `test_ssfr_fix.py` 测试脚本，验证了以下场景：

1. **奇数次迭代**: [1, 2, 1] - 需要额外拷贝步骤
2. **偶数次迭代**: [1, 2] - 自然在正确位置
3. **边界情况**: 空列表、单次迭代等

测试结果显示修复后的逻辑能够正确处理所有情况。

## 性能影响

- **正面影响**: 消除了因使用错误纹理导致的渲染质量问题
- **轻微开销**: 奇数次迭代时增加一次1x1核的拷贝操作
- **总体收益**: 渲染质量和一致性显著提升

## 兼容性

此修复保持了原有API接口不变，仅重构了内部实现逻辑，对现有配置和使用方式完全兼容。

## 总结

这次修复解决了SSFR渲染器的核心逻辑错误，确保了无论滤波迭代次数如何，都能获得正确的滤波结果用于最终合成，从而提升了整体渲染质量和系统稳定性。