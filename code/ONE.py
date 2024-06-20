import matplotlib.pyplot as plt

# 示例数据
#
data_statistics ={0: {0: 6000, 6: 2000}, 1: {1: 6000, 4: 3000}, 2: {2: 6000, 3: 1500}, 3: {3: 1500, 5: 3000}, 4: {4: 3000, 9: 3000}, 5: {3: 1500, 5: 3000}, 6: {6: 2000, 8: 2000}, 7: {3: 1500, 7: 6000}, 8: {6: 2000, 8: 2000}, 9: {8: 2000, 9: 3000}}

# 初始化绘图数据
client_ids = []
class_labels = []
sample_sizes = []

# 遍历数据统计
for client_id, classes in data_statistics.items():
    for class_label, sample_size in classes.items():
        client_ids.append(client_id)
        class_labels.append(class_label)
        sample_sizes.append(sample_size)  # 样本数用来表示点的大小

# 将样本数转化为可视化的点大小
point_sizes = [size / max(sample_sizes) * 1000 for size in sample_sizes]

# 创建颜色映射，根据样本大小映射颜色
colors = sample_sizes

# 绘图
plt.figure(figsize=(8, 8))
scatter = plt.scatter(client_ids, class_labels, s=point_sizes, c=colors, cmap='coolwarm')
plt.xlabel('Client IDs', fontsize=22, fontname='Times New Roman')
plt.ylabel('Class labels', fontsize=22, fontname='Times New Roman')

# 设置网格线为虚线
plt.grid(True, linestyle='--')

# 设置横纵刻度范围从1到10，注意：刻度标签是从0开始的，所以设置的是0到9
plt.xticks(range(10), fontsize=18, fontname='Times New Roman')  # 设置横坐标字体大小和样式
plt.yticks(range(10), fontsize=18, fontname='Times New Roman')

# 添加颜色条
cbar = plt.colorbar(scatter)
cbar.set_label('Sample Size', fontsize=22, fontname='Times New Roman')
cbar.ax.tick_params(labelsize=18)
for label in cbar.ax.get_yticklabels():
    label.set_fontname('Times New Roman')

# 保存为 SVG 格式的图片
plt.savefig(fname="lable2111.jpg", format="jpg")

# 显示图形
plt.show()