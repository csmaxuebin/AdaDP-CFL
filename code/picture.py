import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# 手动输入的数据
data_client_80 = []
data_client_22 = []

data_client_6 = []

# 解析数据的函数
def parse_dp_clips(dp_clip_data):
    rounds, values = [], []
    for item in dp_clip_data:
        round_num, value = item.split()
        rounds.append(int(round_num))
        values.append(float(value))
    return rounds, values

# 解析每个客户端的dp_clip数据
rounds_80, dp_clips_80 = parse_dp_clips(data_client_80)
rounds_22, dp_clips_22 = parse_dp_clips(data_client_22)
rounds_6, dp_clips_6 = parse_dp_clips(data_client_6)

# 绘制主图
fig, ax = plt.subplots(figsize=(10, 8))  # 指定主图的大小
ax.plot(rounds_80, dp_clips_80, label='Client 80', marker='o', linestyle='-',linewidth=3, markersize=10)
ax.plot(rounds_6, dp_clips_6, label='Client 6', marker='o', linestyle='-')
ax.plot(rounds_22, dp_clips_22, label='Client 22', marker='o', linestyle='-')
ax.legend()
ax.set_xlabel('Round Number')
ax.set_ylabel('Dp_clip Value')
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))  # X轴主刻度每隔10显示一次
ax.yaxis.set_major_locator(ticker.MultipleLocator(2))  # Y轴主刻度每隔2显示一次
ax.grid(True)

ax.legend().set_visible(False)
plt.savefig(fname="clip_svhn.svg" , format="svg")
plt.show()