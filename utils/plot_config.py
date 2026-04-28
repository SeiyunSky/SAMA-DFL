"""
Matplotlib配置 - 解决中文显示问题
在所有绘图脚本开头导入此模块
"""
import matplotlib
import matplotlib.pyplot as plt


def setup_matplotlib():
    """
    配置matplotlib
    优先使用中文字体，如果不可用则使用英文
    """
    # 尝试中文字体
    try:
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        # 中文字体不可用，使用默认英文字体
        plt.rcParams['font.family'] = 'DejaVu Sans'

    # 设置图表样式
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'


# 英文标签映射（用于无中文字体环境）
LABELS_EN = {
    '训练轮次': 'Training Round',
    '测试损失': 'Test Loss',
    '测试准确率': 'Test Accuracy (%)',
    '共识误差': 'Consensus Error',
    '鲁棒常数': 'Robustness Constant',
    'κ值': 'Kappa Value',
    '相对误差': 'Relative Error',
    '频数': 'Frequency',
    '直径': 'Diameter',
    '最大成对距离': 'Max Pairwise Distance',
}


def get_label(chinese, use_english=True):
    """
    获取标签（自动回退到英文）

    参数:
        chinese: 中文标签
        use_english: 是否强制使用英文

    返回:
        str - 标签文本
    """
    if use_english and chinese in LABELS_EN:
        return LABELS_EN[chinese]
    return chinese


# 自动配置
setup_matplotlib()
