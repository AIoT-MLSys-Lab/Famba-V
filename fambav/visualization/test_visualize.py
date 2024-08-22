import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

def visualize_cosine_similarity_from_npz(file_path):
    print("Start to visualize cosine similarity from .npz file")
    
    # 加载 .npz 文件
    loaded_data = np.load(file_path)
    
    # 遍历所有的层
    for key in loaded_data.files:
        if key.startswith('layer_'):
            hidden_states = loaded_data[key]
            
            if isinstance(hidden_states, np.ndarray):
                hidden_states = torch.from_numpy(hidden_states)
            
            # 如果数据在GPU上，确保将其移动到CPU上
            if hidden_states.device.type != 'cpu':
                hidden_states = hidden_states.cpu()
            
            # 转换为 NumPy 数组（如果尚未转换）
            hidden_states = hidden_states.numpy()
            
            # 计算余弦相似度矩阵
            cosine_sim_matrix = cosine_similarity(hidden_states)
            
            print(f"cosine_sim_matrix for {key} success")
            
            plt.figure(figsize=(12, 10))  # 增大图像尺寸
            sns.heatmap(cosine_sim_matrix, cmap='Blues')

            # 设置标题和标签，增大字体大小
            plt.suptitle(f"Cosine Similarity Matrix of Hidden States for {key}", fontsize=26, y=0.98)
            plt.xlabel("Hidden State Index", fontsize=26)
            plt.ylabel("Hidden State Index", fontsize=26)

            # 计算新的刻度位置和标签
            n = cosine_sim_matrix.shape[0]  # 假设矩阵是方阵
            ticks = np.arange(0, n, 8)
            tick_labels = ticks

            # 设置新的刻度
            plt.xticks(ticks, tick_labels, fontsize=16)
            plt.yticks(ticks, tick_labels, fontsize=16)

            # 调整colorbar的字体大小
            cbar = plt.gcf().axes[-1]
            cbar.tick_params(labelsize=12)

            plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
            plt.savefig(f"./cosine_similarity_{key}.png", dpi=600)  # 增加DPI以提高图像质量
            print(f"Heatmap for {key} saved.")
    
# 示例 .npz 文件路径
file_path = './hidden_states_20240714_113251.npz'

# 调用函数
visualize_cosine_similarity_from_npz(file_path)