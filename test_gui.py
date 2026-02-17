"""
简单的GUI测试脚本，用于验证matplotlib GUI能否正常显示
"""

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

print("Matplotlib后端:", matplotlib.get_backend())

# 创建简单测试图形
fig, ax = plt.subplots(figsize=(8, 6))
x = np.linspace(0, 10, 100)
y = np.sin(x)
ax.plot(x, y)
ax.set_title('测试图形 - 如果能看到这个窗口，说明GUI正常工作')
ax.set_xlabel('X轴')
ax.set_ylabel('Y轴')

print("正在显示测试窗口...")
plt.show()
print("测试窗口已关闭")