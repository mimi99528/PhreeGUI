# PhreeGUI
> 打造最真实的科普级化学模拟GUI系统。


[🇨🇳 中文](README.md) | [🇬🇧 English](README_en.md)

## 📖 简介
致力于解决当前化学模拟软件普遍：简单好上手-真实性-付费的不可能三角，当前开发了一个demo，可供体验。

![Demo Image](./assets/demo.png)
> 
## ✨ 特性

- 🚀 **高真实性**：以Phreeqcrm为化学核心，splishsplash提供的dfsph为物理核心，利用ssfr进行渲染，做到了准科研级的真实模拟
- 🛠 **易用性**：使用该项目只需要三步：克隆-安装-使用
- 🔌 **扩展性**：预留了丰富的开发接口

## 📦 安装

确保你安装了 [依赖环境]。

```bash
pip install -r requirements.txt
```

## 🚀 快速开始
打开launch_coupled_simulation.py，点击开始模拟，你可以在右边的3d显示窗口看到模拟状态，使用左键来创建粒子，右键移动摄像机。

### 物理后端切换（CPU / Taichi GPU）

默认使用 SPlisHSPlasH CPU 版 DFSPH。启用 Taichi GPU 后端：

```bash
# 需要先安装：pip install "taichi>=1.7"
PHREEGUI_PHYSICS_BACKEND=taichi PHREEGUI_TAICHI_ARCH=cuda python launch_coupled_simulation.py
```

| 变量 | 取值 | 说明 |
|---|---|---|
| `PHREEGUI_PHYSICS_BACKEND` | `cpu`（默认）/ `taichi` | 物理求解器实现 |
| `PHREEGUI_TAICHI_ARCH` | `cuda`（默认）/ `cpu` | Taichi 执行后端 |

注意：
- 两个后端边界模型不同（CPU=Bender2019 体积图，Taichi=Akinci2012 边界粒子），数值不会逐位一致；化学耦合只依赖 (位置, 稳定粒子 id)，契约相同。
- 后端选择失败会直接报错，不会静默降级（Taichi 未安装 / CUDA 不可用都会显式抛错）。

## 🤝 贡献
我们欢迎各种形式的贡献！

## 🙏 致谢

本项目在开发过程中采用了 **Vibe Coding** 工作流，极大地提升了迭代效率。特别感谢以下 AI 工具提供的智能辅助：

- 🤖 [GitHub Copilot](https://github.com/features/copilot)
- 🧠 [通义千问 (Qwen)](https://tongyi.aliyun.com/)
- 🌕 [Kimi Ai](https://kimi.moonshot.cn/)

感谢这些强大的 AI 结对程序员，让创意能更快地转化为代码！

## 📄 许可证

本项目采用 GPL v3 许可证 - 查看 [LICENSE](./LICENSE) 文件了解详情。

---
*Made with ❤️ and 🤖 by mimi*

## 📬 联系

- 作者：mimi - @email mimi99258@outlook.com
