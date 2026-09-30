import matplotlib.pyplot as plt


# 三组实际实验结果
depths = [2, 6, 10]

train_accuracy = [
    74.55,
    77.48,
    85.17
]

test_accuracy = [
    74.60,
    75.10,
    73.20
]


# 创建图表
plt.figure(figsize=(8, 5))

plt.plot(
    depths,
    train_accuracy,
    marker="o",
    label="Train Accuracy"
)

plt.plot(
    depths,
    test_accuracy,
    marker="o",
    label="Test Accuracy"
)


plt.xlabel("Tree Depth")
plt.ylabel("Accuracy (%)")
plt.title("Effect of Tree Depth on CatBoost Performance")

plt.xticks(depths)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()


# 保存图片
plt.savefig(
    "results/depth_comparison.png",
    dpi=200
)

plt.show()