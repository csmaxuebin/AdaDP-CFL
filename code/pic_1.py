import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# 实验数据
# x = [0,5,10,20,30,40,50,60,70,80,90,100]  # x轴数据
# y1 = [0,29.060]
# y2 = []
# y3 = []
# y4 = []
# 设置 x 轴坐标范围为 0 到 10
plt.plot(x, y1, color='red', marker='^', linestyle='--', label='no-prox')
plt.plot(x, y2, color='blue', marker='x', linestyle='--', label='L=1')
plt.plot(x, y3, color='green', marker='+', linestyle='--', label='L=0.1')
plt.plot(x, y4, color='green', marker='-', linestyle='--', label='L=0.01')
plt.xlim(0, 100)
plt.ylim(0, 100)
# 设置x轴的主刻度，例如，每5个轮次一个主刻度
plt.xticks(range(0, 100 + 1, 10))
plt.yticks(range(0, 100 + 1, 10))
# 添加图例
plt.legend(loc='center right', fontsize='large')

# 添加图标题和轴标签
# plt.title('Historical weights of client1', fontsize='large')
plt.xlabel('Round', fontsize='large')
plt.ylabel('Accuracy', fontsize='large')

# 保存图片

plt.savefig(fname="e=10,cifar10.svg" , format="svg")

# 显示图形
plt.show()

