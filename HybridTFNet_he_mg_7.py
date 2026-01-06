# 由3版本而来，画t-SNE、监督UMAP可视化图,重在Umap图形，结果：镁合金、铜合金、钢的监督UMAP图形都可以
import os
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import SpectralClustering
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.pipeline import make_pipeline
from torch import pdist
from torch.linalg import LinAlgError
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    roc_curve, silhouette_score
from sklearn.metrics import auc
from numpy import interp
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve  # ==== 新增校准曲线 ====
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # ==== 新增3D支持 ====
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.model_selection import ParameterGrid  # 自动化参数搜索
import pacmap  # 新增PaCMAP库
from matplotlib import colormaps


# 固定随机种子，确保实验结果可重复
def set_seed(seed):
    random.seed(seed)  # Python 随机数种子
    np.random.seed(seed)  # Numpy 随机数种子
    torch.manual_seed(seed)  # PyTorch 随机数种子
    torch.cuda.manual_seed_all(seed)  # CUDA 随机数种子
    torch.backends.cudnn.deterministic = True  # 确保 CUDNN 使用确定性算法
    torch.backends.cudnn.benchmark = False  # 禁用 CUDNN 自动优化


# 自定义数据集类，用于加载光谱数据
class SpectralDataset(Dataset):
    def __init__(self, data_dir):
        self.spectral_data = []  # 存储光谱特征
        self.labels = []  # 存储标签

        # 遍历数据目录，加载每个文件的数据和标签
        for label in os.listdir(data_dir):
            label_dir = os.path.join(data_dir, label)
            if not os.path.isdir(label_dir):  # 跳过非目录项
                continue

            for file in os.listdir(label_dir):
                file_path = os.path.join(label_dir, file)
                spectral_data = self.process_data(file_path)  # 处理光谱数据
                self.spectral_data.append(spectral_data)
                self.labels.append(int(label))  # 保存标签

        self.labels = torch.tensor(self.labels, dtype=torch.long)  # 转为张量

    @staticmethod
    def process_data(file_path):
        # 加载光谱数据文件，提取波长和强度
        data = pd.read_csv(file_path, header=None).values
        wavelengths = data[:, 0]  # 波长
        intensities = data[:, 1]  # 强度

        # 数据归一化，将波长和强度映射到 [0, 1]
        # 统一归一化
        min_val = min(wavelengths.min(), intensities.min())
        max_val = max(wavelengths.max(), intensities.max())
        wavelengths = (wavelengths - min_val) / (max_val - min_val)
        intensities = (intensities - min_val) / (max_val - min_val)

        # 提取峰值特征
        peaks, _ = find_peaks(intensities, height=0.8 * intensities.max())  # 检测峰值
        num_peaks = len(peaks)  # 峰值数量
        peak_intensity_mean = intensities[peaks].mean() if num_peaks > 0 else 0  # 峰值强度均值

        # 计算统计特征（均值和标准差）
        mean_intensity = intensities.mean()
        std_intensity = intensities.std()

        # 组合特征向量
        features = np.concatenate(
            [wavelengths + intensities, [num_peaks, peak_intensity_mean, mean_intensity, std_intensity]])
        return torch.tensor(features, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)  # 返回数据集大小

    def __getitem__(self, idx):
        return self.spectral_data[idx], self.labels[idx]  # 返回特定索引的数据和标签


# 模糊逻辑层，用于模糊化输入特征
class FuzzyLayer(nn.Module):
    def __init__(self, input_dim, num_rules):
        super(FuzzyLayer, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_rules, input_dim))  # 模糊规则的中心
        self.widths = nn.Parameter(torch.ones(num_rules, input_dim))  # 模糊规则的宽度

    def forward(self, x):
        x = x.unsqueeze(1)  # 扩展维度以适配规则
        centers = self.centers.unsqueeze(0)  # 扩展中心维度
        diff = x - centers  # 计算输入与规则中心的差异
        exp = -torch.pow(diff / (self.widths.unsqueeze(0) + 1e-6), 2)  # 高斯模糊函数
        membership = torch.exp(exp.sum(dim=-1))  # 计算模糊隶属度
        return membership


# Transformer 特征提取器，用于提取深层语义特征
class SpectrumTransformerExtractor(nn.Module):
    def __init__(self, input_dim, embed_dim=128, num_heads=4, num_layers=2, dropout=0.1):
        super(SpectrumTransformerExtractor, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)  # 将输入映射到 embed_dim  # 特征嵌入
        self.positional_encoding = nn.Parameter(torch.randn(1, 1000, embed_dim))  # 位置编码
        encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)  # Transformer 编码器   # 自注意力
        self.output_fc = nn.Linear(embed_dim, embed_dim)  # 输出层

    def forward(self, x):
        if len(x.shape) == 2:  # 如果输入是 (batch_size, input_dim)，增加序列维度
            x = x.unsqueeze(1)
        seq_len = x.size(1)
        x = self.embedding(x)  # 嵌入
        x = x + self.positional_encoding[:, :seq_len, :]  # 加上位置编码  嵌入+位置编码
        x = self.transformer_encoder(x)  # Transformer 编码   # 自注意力计算
        x = x.mean(dim=1)  # 对序列维度求平均     # 全局特征
        return self.output_fc(x)  # 输出结果


# HybridTFNet 模型
class HybridTFNet(nn.Module):
    '''
    Hybrid 表示结合了模糊逻辑（Fuzzy Logic） 和 CNN, TF表示以Transformer为核心, 故叫 HybridTFNet
    '''

    def __init__(self, spectrum_dim, num_classes):
        super(HybridTFNet, self).__init__()
        self.fuzzy_spectrum = FuzzyLayer(input_dim=spectrum_dim, num_rules=200)  # 模糊逻辑层
        self.spectrum_extractor = SpectrumTransformerExtractor(input_dim=200)  # Transformer 特征提取
        self.conv_feature_extractor = nn.Sequential(  # CNN 特征提取
            nn.Conv1d(1, 64, kernel_size=3, padding=1),  # 卷积层
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),  # 卷积层
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2)  # 最大池化层
        )

        dummy_spectrum = torch.randn(1, spectrum_dim).unsqueeze(1)
        with torch.no_grad():
            conv_output = self.conv_feature_extractor(dummy_spectrum)
            conv_output_dim = torch.flatten(conv_output, start_dim=1).size(1)  # 计算 CNN 输出维度

        dummy_fuzzy_spectrum = self.fuzzy_spectrum(torch.randn(1, spectrum_dim))
        deep_spectrum_dim = self.spectrum_extractor(dummy_fuzzy_spectrum.unsqueeze(1)).shape[1]  # 计算 Transformer 输出维度

        combined_dim = deep_spectrum_dim + conv_output_dim  # 特征融合维度

        self.fc = nn.Sequential(  # 全连接分类层
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )

    def forward(self, spectrum):
        fuzzy_spectrum = self.fuzzy_spectrum(spectrum)  # 模糊逻辑提取
        spectrum_features = self.spectrum_extractor(fuzzy_spectrum.unsqueeze(1))  # Transformer 提取
        spectrum = spectrum.unsqueeze(1)  # 增加通道维度
        conv_features = self.conv_feature_extractor(spectrum)  # CNN 提取
        conv_features = torch.flatten(conv_features, start_dim=1)  # 展平 CNN 输出
        combined_features = torch.cat([spectrum_features, conv_features], dim=1)  # 特征拼接
        return self.fc(combined_features)  # 分类结果

    def extract_features(self, spectrum):
        """特征提取方法"""
        fuzzy_spectrum = self.fuzzy_spectrum(spectrum)
        spectrum_features = self.spectrum_extractor(fuzzy_spectrum.unsqueeze(1))
        spectrum = spectrum.unsqueeze(1)
        conv_features = self.conv_feature_extractor(spectrum)
        conv_features = torch.flatten(conv_features, start_dim=1)

        return torch.cat([spectrum_features, conv_features], dim=1)


from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE  # 在文件开头添加导入


# 修改后的 plot_confusion_matrix（添加labels_length参数）
def plot_confusion_matrix(cm, labels_length):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.figure(figsize=(6, 6))
    cell_size = 0.5  # 每个单元格的尺寸（英寸）
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                annot_kws={"size": 16},  # 注释字体放大
                xticklabels=range(labels_length),
                yticklabels=range(labels_length),
                square=True,  # 关键参数：单元格显示为正方形
                )
    plt.xlabel("Predicted labels", fontsize=18)
    plt.xticks(fontsize=16, fontname='Times New Roman')
    plt.yticks(fontsize=16, fontname='Times New Roman')
    plt.ylabel("True labels", fontsize=18)
    # plt.title("Confusion Matrix", fontsize=16)
    plt.show()


def plot_bar(y_true, y_true_cal, y_pred_cal):
    bar_width = 0.4
    unique_y_true = np.unique(y_true)
    bar_positions_shop1 = [i for i in range(len(unique_y_true))]
    bar_positions_shop2 = [i + bar_width for i in bar_positions_shop1]
    print(bar_positions_shop1)
    # 计算相对误差（示例计算方式，你可按需替换为实际正确的计算逻辑）
    # relative_errors = np.abs((np.array(y_true_cal) - np.array(y_pred_cal)) / np.array(y_true_cal))

    # 绘制柱状图Shop 1
    plt.bar(bar_positions_shop1, y_pred_cal, width=bar_width, label='pred',color='deeppink')#color='deeppink'品红色，#FFC0CB粉红
    # 在Shop 1的柱子上添加数值标签
    for i in range(len(bar_positions_shop1)):
        plt.text(bar_positions_shop1[i], y_pred_cal[i], str(y_pred_cal[i]), ha='center', va='bottom',)

    # 绘制柱状图Shop 2
    plt.bar(bar_positions_shop2, y_true_cal, width=bar_width, label='true',color='orange')#color='paleturquoise'淡青色，#FF9800橙色
    # 在Shop 2的柱子上添加数值标签
    for i in range(len(bar_positions_shop2)):
        plt.text(bar_positions_shop2[i], y_true_cal[i], str(y_true_cal[i]), ha='center', va='bottom')

    max_value = max(max(y_pred_cal), max(y_true_cal))
    # # 在柱子之间添加相对误差标签（位置和样式可根据实际调整优化）
    # for i in range(len(bar_positions_shop1)):
    #     x_pos = (bar_positions_shop1[i] + bar_positions_shop2[i]) / 2  # 取两个柱子中间位置
    #     y_pos = max_value + 10  # 将相对误差标签统一放置在整体数据最大值上方一定距离处
    #     plt.text(x_pos, y_pos, f"{relative_errors[i]:.2f}", ha='center', va='bottom')

    plt.xlabel('Class', fontsize=18)  # 这里修改为Categories更通用些，可根据实际改回Fruits等合适名称
    plt.ylabel('Values', fontsize=18)
    # plt.title('True vs Predicted Class Distribution')
    plt.xticks([i + bar_width / 2 for i in bar_positions_shop1], unique_y_true)
    plt.legend()
    plt.show()

#

###
# # 在原有代码基础上新增Umap和t-SNE聚类图（保持原有模型架构不变）
#
# --------------------- 修正版特征增强模块 ---------------------
from sklearn.base import BaseEstimator, TransformerMixin


class DynamicSVD(TransformerMixin, BaseEstimator):
    """完全兼容scikit-learn API的降维组件"""

    def __init__(self):
        self.n_components_ = None
        self.svd_ = None

    def fit(self, X, y=None):
        self.n_components_ = min(150, X.shape[1])
        if self.n_components_ < 2:
            self.n_components_ = 2
        self.svd_ = TruncatedSVD(n_components=self.n_components_)
        self.svd_.fit(X)
        return self

    def transform(self, X):
        return self.svd_.transform(X)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class FeatureEnhancerPro:
    def __init__(self):
        self.scaler = make_pipeline(
            QuantileTransformer(n_quantiles=3500, output_distribution='normal'), #镁和铜用n_quantiles=1500，钢用n_quantiles=3500
            PCA(n_components=0.98, whiten=True),
            DynamicSVD()  # 使用完全兼容的组件
        )

    def enhance(self, features, labels=None):
        # 数据清洗强化
        features = np.nan_to_num(features, nan=0.0)
        features = np.clip(features, -1e5, 1e5)

        # 维度保护
        if features.shape[1] < 2:
            features = np.hstack([features, np.zeros((features.shape[0], 2))])

        return self.scaler.fit_transform(features)


# --------------------- 绝对稳定可视化模块 ---------------------
def plot_supervised_umap_ultimate(features, labels):
    """工业级稳定UMAP可视化
    减小 n_neighbors 可以让数据更局部化，而增大 min_dist 可以让类别之间的距离更明显。
    """
    # 参数自动安全设置
    config = {
        'n_neighbors':max(3, min(50, int(len(features) * 0.05))),
        'min_dist':0.5,# 0.5,#0.1,#max(0.05, 1.0 / np.sqrt(len(features))),
        'metric': 'manhattan',
        'target_weight': 0.98,
        #让同类别的点分布的更散
        'spread' :1,  # 增大点分布范围,spread 与 min_dist 类似，但更直接地影响点之间的分布。
        'local_connectivity' : 1, # 增大局部连接性，local_connectivity 控制了局部连接性，增大这个值可以让点之间的连接更松散。

        'random_state': 42
    }

    # 降维安全执行
    try:
        reducer = UMAP(**config)
        embeddings = reducer.fit_transform(features, labels)
    except Exception as e:
        print(f"UMAP Error: {str(e)}, Using PCA as fallback")
        embeddings = PCA(n_components=2).fit_transform(features)

    # 可视化安全渲染
    plt.figure(figsize=(6, 5))
    unique_labels = np.unique(labels)

    for label in unique_labels:
        mask = labels == label
        pts = embeddings[mask]
        if len(pts) == 0:
            continue

        # 密度安全计算
        try:
            kde = gaussian_kde(pts.T)
            density = kde(pts.T)
        except:
            density = np.ones(len(pts))

        plt.scatter(pts[:, 0], pts[:, 1],
                    s=50,  # 点大小
                    # s=30 + 100 * (density / density.max()),
                    alpha=0.7,
                    edgecolors='w',
                    linewidth=0.3,
                    zorder=2,
                    label=f'Class {label}')

        # plt.scatter(pts[:, 0], pts[:, 1],
        #             s=30 + 100 * (density / density.max()),
        #             alpha=0.7,
        #             label=f'Class {label}')
    # 坐标轴标签
    plt.xlabel('Umap 1', fontsize=16)
    plt.ylabel('Umap 2', fontsize=16)

    plt.legend(loc='upper left',  # 图例位置：左下角
               fontsize=6,  # 图例字体大小
               frameon=False,  # 去掉图例边框
               markerscale=0.7)  # 缩小图例标记的大小)
    plt.show()


# ==== 优化后的2D t-SNE可视化函数 ====
def plot_tsne(features, labels, title="t-SNE Visualization",
              perplexity=30, n_iter=5000, random_state=42,
              early_exaggeration=32,  # 新增参数
              learning_rate='auto',  # 确保参数存在
              angle=0.3):  # 新增角度参数

    """
    专用于2D t-SNE可视化
    参数：
        features: 特征矩阵 (n_samples, n_features)   t-SNE/UMAP 可视化所用的特征来源于模型训练后提取的融合特征extract_features()
        labels: 类别标签 (n_samples,)
        title: 图表标题
        perplexity: 困惑度（建议根据数据量调整）
        perplexity 可理解为 "有效邻居数量"，其物理意义是：
        低值（如5-10）：强调局部结构，能看到更细粒度的聚类
        高值（如30-50）：关注全局结构，能展现大类之间的拓扑关系
        n_iter: 迭代次数
    """
    plt.rcParams["font.family"] = "Times New Roman"

    # 数据标准化
    # 增加PCA预处理
    # 确保n_components不超过数据的最小维度
    n_components_pca = min(50, features.shape[0], features.shape[1])
    n_components_pca = max(n_components_pca, 2)  # 至少保留2个成分
    pca = PCA(n_components=n_components_pca)
    features = pca.fit_transform(features)

    # 创建t-SNE模型
    tsne = TSNE(n_components=2,
                perplexity=150,  # max(5, int(len(features)*0.05)),  # 自适应设置,
                random_state=random_state,
                init='pca',
                learning_rate='auto',
                early_exaggeration=early_exaggeration,  # 增强初始放大效应
                metric='cosine',  # 改用余弦距离
                angle=0.3,  # 加速计算同时保持精度
                n_jobs=-1,

                )  # 自动学习率

    # 执行降维

    embeddings = tsne.fit_transform(features)

    # 可视化设置
    plt.figure(figsize=(6, 5))
    cmap = plt.get_cmap('tab10')  # 使用10色循环
    unique_labels = np.unique(labels)

    # ===== 新增维度校验 =====
    assert len(features) == len(labels), \
        f"特征与标签维度不匹配！特征数：{len(features)}，标签数：{len(labels)}"

    # 绘制每个类别的散点
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(embeddings[mask, 0], embeddings[mask, 1],
                    color=cmap(i % 20),  # 直接指定颜色
                    label=f'Class{label}',
                    s=50,
                    alpha=0.8,
                    edgecolors='w',
                    linewidth=0.5)

    # 坐标轴标签
    plt.xlabel('t-SNE 1', fontsize=18)
    plt.ylabel('t-SNE 2', fontsize=18)

    # 标题和图例
    # plt.title(title, fontsize=14, pad=15)
    plt.legend(loc='upper right',
               # bbox_to_anchor=(1.25, 1),  # 图例放在右侧
               fontsize=8,
               frameon=False,
               )  # title='Categories'

    # 网格线和边框
    plt.grid(alpha=0.2, linestyle='--')
    plt.tight_layout()
    plt.show()


# ==== 新增类条件概率分布图 ====
def plot_class_distributions(y_true, y_prob, class_names=None):
    """
    绘制每个类别的预测概率分布密度图
    参数:
        y_true: 真实标签数组 (n_samples,)
        y_prob: 预测概率矩阵 (n_samples, n_classes)
        class_names: 类别名称列表 (可选)
    """
    plt.rcParams["font.family"] = "Times New Roman"
    unique_labels = np.unique(y_true)
    n_classes = len(unique_labels)

    class_names = class_names or [f'Class {i}' for i in unique_labels]

    rows = int(np.ceil(n_classes / 3))
    cols = min(n_classes, 3)
    plt.figure(figsize=(cols * 5, rows * 4))

    for i, label in enumerate(unique_labels):
        ax = plt.subplot(rows, cols, i + 1)
        class_probs = y_prob[y_true == label, i]

        if len(class_probs) > 0:  # 防止空类别
            sns.kdeplot(class_probs,
                        fill=True,
                        color=plt.cm.tab20(i % 20),
                        linewidth=2,
                        alpha=0.6)

            # 添加统计信息
            mean_prob = np.mean(class_probs)
            plt.axvline(mean_prob, color='darkred', linestyle='--',
                        label=f'Mean: {mean_prob:.2f}')

            plt.plot(np.linspace(0, 1, 100), np.ones(100), 'k:')

            plt.title(f'{class_names[i]} (n={len(class_probs)})', fontsize=14)
            plt.xlabel('Probability', fontsize=12)
            plt.grid(True, alpha=0.3)

            if i == 0:
                plt.legend(loc='upper left')

    plt.tight_layout()
    plt.show()


# ==== 新增学习率曲线 ====
# 优化过程监控：展示训练过程中学习率的动态变化，反映优化器的调度策略（如余弦退火、步长衰减）。
# 训练稳定性分析：平稳的学习率变化表明优化过程可控；剧烈波动可能暗示超参数设置不当（如初始学习率过高）
def plot_lr_curve(lr_history):
    plt.figure(figsize=(7, 5))
    plt.plot(lr_history, marker='o', color='darkorange')
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Learning Rate', fontsize=18)
    # plt.title('Learning Rate Schedule', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


# ==== 新增校准曲线 ====
# 概率可靠性评估：检查模型预测概率是否反映真实置信度。例如，预测概率0.8的样本应有约80%的实际正确率。
# 校准误差识别：若曲线高于对角线，模型过于自信（预测概率高于实际概率）；若低于对角线，则不够自信。

def plot_calibration_curve(y_true, y_prob):
    plt.figure(figsize=(8, 5))
    for i in range(y_prob.shape[1]):
        true_prob, pred_prob = calibration_curve((y_true == i).astype(int), y_prob[:, i], n_bins=10)
        plt.plot(pred_prob, true_prob, marker='o', linestyle='--', label=f'Class {i}')

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)
    plt.xlabel('Predicted Probability', fontsize=18)
    plt.ylabel('True Probability', fontsize=18)
    plt.title('Calibration Curve', fontsize=16)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.show()


def calculate_number(y, unique_labels):
    # 根据已知的唯一标签初始化字典
    count_dict = {label: 0 for label in unique_labels}
    for element in y:
        count_dict[element] += 1
    # 返回按unique_labels顺序排列的计数列表
    return [count_dict[label] for label in unique_labels]


def plot_train_val_curve(train_losses, val_losses, train_accuracies, val_accuracies):
    """
    绘制训练和验证集的准确率、损失随训练轮次变化曲线
    """
    epochs_range = range(1, len(train_losses) + 1)  # 动态生成横坐标
    plt.figure(figsize=(12, 5))
    # plt.figure(figsize=(16, 6))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Training Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.title('Training and Validation Loss')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label='Training Accuracy')
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()


# 在训练函数train_model中记录每轮的损失和准确率
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []


def plot_roc_curve(y_true, y_score):
    assert isinstance(y_score, np.ndarray), "y_score must be numpy array"
    y_test_binarized = label_binarize(y_true,
                                      classes=np.unique(y_true))  # np.unique 去重并且按照顺序输出0,1,2,3,4,5,6,7； 输出结构最终为0，1
    # print(np.unique(y_test))
    print(len(y_test_binarized))
    fpr_micro, tpr_micro, _ = roc_curve(y_test_binarized.ravel(),
                                        y_score.ravel())  # ravel将数组维度拉为一维数组fpr：False positive rate。tpr：True positive rate。
    # print(fpr_micro)
    roc_auc_micro = auc(fpr_micro, tpr_micro)  # 输出面积的zhi
    plt.figure(figsize=(6, 5))  # (8, 5)
    plt.plot(fpr_micro, tpr_micro,
             label='Micro-average ROC curve (area = {0:0.4f})'.format(roc_auc_micro),
             color='deeppink', linestyle=':', linewidth=4)
    fpr = dict()  # 创建一个新的字典，/空{}
    # print(fpr)
    tpr = dict()
    roc_auc = dict()
    n_classes = len(np.unique(y_true))
    print(n_classes)
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], lw=2, label='ROC curve of class %d (area = %0.4f)' % (i, roc_auc[i]))
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)  # 复制矩阵并且全写0
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])  # 基于离散数组fpr[i]和tpr[i]的映射关系，对新的输入数组x预测对应的输出y(线性插值)
    mean_tpr /= n_classes  ##怎么除出来的
    roc_auc_macro = auc(all_fpr, mean_tpr)

    plt.plot(all_fpr, mean_tpr, label='Macro-average ROC curve (area = {0:0.4f})'.format(roc_auc_macro),
             color='navy', linestyle=':', linewidth=4)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    plt.gca().set_aspect(0.7)  # 横轴为纵轴的1.4倍
    plt.xlim(-0.05, 1.05)  # 横轴多留5%空白
    plt.ylim(0, 1.05)
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    # plt.title('ROC Curve for multi-class',fontsize=16)
    plt.legend(loc="lower right",
               prop={'size': 6},  # 将图例的字体大小设置为 8。
               handlelength=1,  # 将图例中线条的长度设置为 1。
               handletextpad=0.3,  # 将图例中线条与文字之间的间距设置为 0.5。
               # labelspacing=0.2,  #将图例中不同条目之间的垂直间距设置为 0.2。
               # borderpad=0.3  # 将图例边框与内容之间的间距设置为 0.3。
               )
    plt.show()


# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=20):
    global epoch
    lr_history = []  # ==== 新增学习率记录 ====
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0
        for spectra, labels in train_loader:
            spectra, labels = spectra.float(), labels.long()
            optimizer.zero_grad()
            outputs = model(spectra)
            loss = criterion(outputs, labels)  ## 计算预测值和真实值的损失
            loss.backward()  ### 反向传播计算梯度，loss.backward() 会反向传播误差并累积梯度。
            optimizer.step()  ## 根据梯度更新模型参数
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_losses.append(total_loss / len(train_loader))
        train_accuracies.append(correct_train / total_train)
        lr_history.append(optimizer.param_groups[0]['lr'])  # 记录学习率

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

        val_loss, val_accuracy, val_macro_f1 = evaluate_model(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Macro F1: {val_macro_f1:.4f}")

        # 每个epoch后调用 scheduler.step()
        scheduler.step()

        if epoch == epochs - 1:  # 只在最后一次绘图 (epoch + 1) % 10 == 0:  # 每10个epoch绘制一次
            plot_train_val_curve(train_losses, val_losses, train_accuracies, val_accuracies)
            plot_lr_curve(lr_history)  # 训练结束后绘制学习率曲线


# 评估函数
def evaluate_model(model, loader, criterion=None):
    model.eval()
    total_loss = 0
    y_true, y_pred = [], []
    y_score = []  # 改为列表存储每个batch的数组
    features = []  # ==== 新增特征收集 ====

    with torch.no_grad():
        for spectra, labels in loader:
            spectra, labels = spectra.float(), labels.long()
            outputs = model(spectra)
            if criterion:
                loss = criterion(outputs, labels)
                total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            # 新增特征重要性
            batch_features = model.extract_features(spectra).numpy()
            features.append(batch_features)

            # 收集数据
            y_true.extend(labels.tolist())
            y_pred.extend(predictions.tolist())
            probs = torch.softmax(outputs, dim=1).numpy()  # 获取每个类别的概率值
            y_score.extend(probs)  # 修改为 extend 并直接存储概率数组
            # y_score.append(probs)  # 追加每个batch的二维数组

    # 处理最终数据
    y_score = np.vstack(y_score)  # 形状为 (总样本数, 类别数)
    cm = confusion_matrix(y_true, y_pred)
    unique_labels = sorted(np.unique(y_true + y_pred))
    y_true_cal = calculate_number(y_true, unique_labels)
    y_pred_cal = calculate_number(y_pred, unique_labels)
    print(cm)

    # ==== 修改：清理特征数据 ====
    features = np.vstack(features)
    features = np.nan_to_num(features)  # 替换 NaN/Inf
    features = features.astype(np.float32)  # 降低精度节省内存

    # 绘图部分保持标签一致性
    if epoch == epochs - 1:
        plot_confusion_matrix(cm, len(unique_labels))
        plot_bar(unique_labels, y_true_cal, y_pred_cal)
        plot_roc_curve(y_true, y_score)
        plot_calibration_curve(np.array(y_true), y_score)
        plot_class_distributions(
            np.array(y_true),
            y_score,
            class_names=[str(i) for i in unique_labels]  # 可根据实际数据修改类别名称
        )
        # 全量数据可视化
        # plot_tsne(features, np.array(y_true), title="t-SNE Clustering")  #, perplexity=30
        # plot_supervised_umap(features, np.array(y_true), title="Supervised UMAP")  # 新增
        # 特征增强
        enhancer = FeatureEnhancerPro()
        try:
            enhanced_features = enhancer.enhance(features)
        except Exception as e:
            print(f"Feature Enhancement Error: {str(e)}")
            enhanced_features = PCA(n_components=2).fit_transform(features)

        # 安全可视化
        plot_tsne(enhanced_features, np.array(y_true))  # , perplexity=30
        plot_supervised_umap_ultimate(enhanced_features, np.array(y_true))
        # 若数据量过大进行采样
        if len(features) > 5000:
            idx = np.random.choice(len(features), 2000, replace=False)
            sampled_labels = np.array(y_true)[idx]
            plot_tsne(enhanced_features[idx], sampled_labels, title="Sampled t-SNE", perplexity=40)
            # plot_supervised_umap(features[idx], sampled_labels, n_neighbors=10)  # 新增
            # ==== 强化可视化 ====
            plot_supervised_umap_ultimate(enhanced_features[idx], np.array(y_true))

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    if criterion:
        return total_loss / len(loader), acc, macro_f1
    return acc, macro_f1


# 主函数
if __name__ == "__main__":
    # 超参数
    seed = 2024
    num_class_map = {'mg_discharge_1_3kv_0_8ma': 5, 'copper_alloy': 5, 'aluminium_alloy_discharge_1_160kv': 11,
                     'steel_alloy_discharge_2_5kv': 9}
    epochs =200  # 50#200#100#200
    learning_rate = 1e-4
    batch_size = 1024

    # 数据集名称
    data_name = "mg_discharge_1_3kv_0_8ma"
    # 数据路径
    # 数据集第一列表示光谱波长, 第二列表示是信号强度
    root_dir = "data/He/target/"
    train_dir = root_dir + data_name + '/train/'
    valid_dir = root_dir + data_name + '/valid/'
    test_dirs = [root_dir + data_name + f'/test{i}/' for i in range(4)]

    print(f'参数: {data_name},{num_class_map[data_name]},{seed},{epochs},{learning_rate},{batch_size}')

    # 设置随机种子
    set_seed(seed)

    # 数据加载
    train_dataset = SpectralDataset(data_dir=train_dir)
    val_dataset = SpectralDataset(data_dir=valid_dir)
    test_datasets = [SpectralDataset(test_dir) for test_dir in test_dirs]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loaders = [DataLoader(test_dataset, batch_size=batch_size, shuffle=False) for test_dataset in test_datasets]

    # 模型训练
    spectrum_dim = train_dataset[0][0].shape[0]
    model = HybridTFNet(spectrum_dim=spectrum_dim, num_classes=num_class_map[data_name])
    criterion = nn.CrossEntropyLoss()  ## 交叉熵损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print("Training model...")
    train_model(model, train_loader, val_loader, criterion, optimizer, epochs=epochs)

    # ==== 新增特征重要性图 ====
    # plot_feature_importance(model)

    print("Evaluating model on Test Set...")
    for i, test_loader in enumerate(test_loaders):
        test_loss, test_accuracy, test_macro_f1 = evaluate_model(model, test_loader, criterion)
        print(f"Test{i} Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, Macro F1: {test_macro_f1:.4f}")
