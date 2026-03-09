# AI彩票量化研究系统 - Python版本

> **⚠️ 重要声明**: 本系统**仅供教育和学术研究**使用。彩票结果完全随机，任何预测模型都无法改变随机性。切勿用于实际投注。

## 📋 目录

- [系统概述](#系统概述)
- [核心功能](#核心功能)
- [安装步骤](#安装步骤)
- [使用方法](#使用方法)
- [技术架构](#技术架构)
- [算法说明](#算法说明)
- [文件结构](#文件结构)
- [常见问题](#常见问题)

## 🎯 系统概述

这是一个完全基于Python实现的AI彩票量化研究系统，包含：

- ✅ **8种高级特征工程**
- ✅ **5个机器学习模型**
- ✅ **Transformer深度学习模型**
- ✅ **概率融合引擎**
- ✅ **历史回测系统**
- ✅ **Streamlit Web界面**
- ✅ **交互式可视化**

## ✨ 核心功能

### 1. 数据处理
- Excel文件自动读取 (.xlsx, .xlsm, .xls)
- 数据清洗与验证
- 派生特征计算（大小、单双、波色等）
- 统计分析

### 2. 8维特征工程

#### 特征1: 深度统计特征
```python
- 均值、标准差
- 偏度 (Skewness)
- 峰度 (Kurtosis)
- 变异系数 (CV)
- 四分位距 (IQR)
- 平均绝对偏差 (MAD)
```

#### 特征2: 频率与概率特征
```python
- 信息熵 (Entropy)
- 归一化熵
- 吉尼不纯度 (Gini)
- 热号/冷号识别
```

#### 特征3: 时间序列特征
```python
- 自相关函数 (ACF)
- 趋势强度
- R²拟合优度
- 平稳性检验
```

#### 特征4: 波动性与动量特征
```python
- 多窗口波动率
- RSI相对强弱指标
- 动量计算
```

#### 特征5: 模式识别特征
```python
- 连续递增/递减检测
- Z字形模式
- 重复数字
- 等差数列识别
```

#### 特征6: 空间分布特征
```python
- 区域平衡度
- 波色分布
- 单双比例
- 优势区域/波色
```

#### 特征7: 遗漏特征
```python
- 最大遗漏
- 平均遗漏
- 当前遗漏分布
- 热号/冷号列表
```

#### 特征8: 组合特征
```python
- 大小单双组合
- 和值分析
- 组合频率
```

### 3. 机器学习模型

#### 模型1: 改进朴素贝叶斯
```python
class NaiveBayes:
    - 基础频率计算
    - 热号加权 (×1.3)
    - 冷号降权 (×0.7)
    - 趋势自适应调整
```

#### 模型2: 加权K近邻
```python
class WeightedKNN:
    - 多特征距离计算
    - 反距离加权
    - 自适应K值
```

#### 模型3: 决策树分类器
```python
class DecisionTree:
    规则1: 大小趋势判断
    规则2: 波色主导性
    规则3: 最大遗漏补偿
    规则4: 周期性模式
```

#### 模型4: 随机森林
```python
class RandomForest:
    - 集成4个基础学习器
    - 投票加权融合
```

#### 模型5: 梯度提升
```python
class GradientBoosting:
    - 多轮迭代优化
    - 残差拟合
    - 学习率控制
```

### 4. Transformer深度学习

```python
class Transformer:
    输入层: 序列编码 (30期历史)
      ↓
    多头注意力层 (4个头)
      ↓
    位置编码层 (Sin/Cos)
      ↓
    前馈神经网络
      ↓
    输出层: 49维概率分布
```

**核心技术**:
- 多头注意力机制
- Query-Key-Value注意力
- 位置编码
- 时间衰减
- 置信度评估

### 5. 概率融合

```python
# 简单加权融合
fused = Σ(model_prob[i] × weight[i])

# 堆叠集成
stacked = meta_learning(base_models, features)
```

### 6. 历史回测

```python
支持策略:
- TOP1: 预测单个号码
- TOP3: 预测3个号码范围
- TOP5: 预测5个号码范围

输出指标:
- 总测试数
- 命中次数
- 准确率
- 详细记录
```

## 🚀 安装步骤

### 环境要求
- Python 3.8+
- pip 或 conda

### 1. 克隆或下载代码

```bash
# 下载所有Python文件到本地目录
lottery_system/
├── lottery_core.py      # 核心算法模块
├── lottery_app.py       # Streamlit界面
├── requirements.txt     # 依赖文件
└── README.md           # 本文档
```

### 2. 安装依赖

```bash
# 使用pip安装
pip install -r requirements.txt

# 或使用conda
conda install --file requirements.txt
```

### 3. 验证安装

```bash
python -c "import pandas, numpy, scipy, plotly, streamlit; print('✓ 所有依赖安装成功')"
```

## 📖 使用方法

### 方法1: Web界面（推荐）

```bash
# 启动Streamlit Web应用
streamlit run lottery_app.py
```

浏览器会自动打开 `http://localhost:8501`

**使用步骤**:
1. 上传Excel数据文件
2. 调整预测参数（可选）
3. 点击「运行AI预测」
4. 在各标签页查看结果

### 方法2: 命令行使用

```python
# 导入模块
from lottery_core import DataProcessor, PredictionEngine

# 加载数据
df = DataProcessor.load_excel('澳门六合彩数据导入器.xlsm')
data = DataProcessor.parse_data(df)

# 运行预测
engine = PredictionEngine(data)
predictions = engine.run_prediction(top_k=10, transformer_top_k=10)

# 打印结果
engine.print_predictions()
```

### 方法3: Jupyter Notebook

```python
# 在Jupyter中使用
import pandas as pd
from lottery_core import *

# 加载数据
df = pd.read_excel('澳门六合彩数据导入器.xlsm', sheet_name='六合彩数据')
data = DataProcessor.parse_data(df)

# 提取特征
features = FeatureEngineering.extract_all_features(data)

# 运行单个模型
nb_probs = MLModels.naive_bayes(data, features)
top_10 = EnsembleFusion.get_top_predictions(nb_probs, 10)

# 显示结果
pd.DataFrame(top_10)
```

## 🏗️ 技术架构

```
┌─────────────────────────────────────────────┐
│           Streamlit Web界面                  │
│     (lottery_app.py)                        │
│  - 数据上传                                  │
│  - 参数配置                                  │
│  - 结果展示                                  │
│  - 可视化图表                                │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│          核心算法模块                        │
│     (lottery_core.py)                       │
│                                             │
│  ┌─────────────────────────────────────┐   │
│  │  DataProcessor                      │   │
│  │  - Excel读取                        │   │
│  │  - 数据清洗                         │   │
│  │  - 统计分析                         │   │
│  └─────────────────────────────────────┘   │
│                    ↓                        │
│  ┌─────────────────────────────────────┐   │
│  │  FeatureEngineering                 │   │
│  │  - 8种特征提取                      │   │
│  │  - 统计/频率/时序/波动...           │   │
│  └─────────────────────────────────────┘   │
│                    ↓                        │
│  ┌─────────────────────────────────────┐   │
│  │  MLModels                           │   │
│  │  - 朴素贝叶斯                       │   │
│  │  - K近邻                            │   │
│  │  - 决策树                           │   │
│  │  - 随机森林                         │   │
│  │  - 梯度提升                         │   │
│  └─────────────────────────────────────┘   │
│                    ↓                        │
│  ┌─────────────────────────────────────┐   │
│  │  TransformerModel                   │   │
│  │  - 多头注意力                       │   │
│  │  - 位置编码                         │   │
│  │  - 前馈网络                         │   │
│  └─────────────────────────────────────┘   │
│                    ↓                        │
│  ┌─────────────────────────────────────┐   │
│  │  EnsembleFusion                     │   │
│  │  - 简单融合                         │   │
│  │  - 堆叠集成                         │   │
│  └─────────────────────────────────────┘   │
│                    ↓                        │
│  ┌─────────────────────────────────────┐   │
│  │  BacktestEngine                     │   │
│  │  - 历史回测                         │   │
│  │  - 性能评估                         │   │
│  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│              数据层                          │
│  - Excel文件                                │
│  - pandas DataFrame                         │
│  - numpy数组                                │
└─────────────────────────────────────────────┘
```

## 🔬 算法说明

### 特征工程数学公式

#### 1. 统计特征
```
偏度 (Skewness):
S = E[(X-μ)³] / σ³

峰度 (Kurtosis):
K = E[(X-μ)⁴] / σ⁴ - 3

变异系数:
CV = σ / μ
```

#### 2. 频率特征
```
信息熵:
H(X) = -Σ p(x) log₂ p(x)

吉尼不纯度:
Gini = 1 - Σ p(x)²
```

#### 3. 时间序列
```
自相关函数:
ACF(k) = Σ(xᵢ-μ)(xᵢ₊ₖ-μ) / Σ(xᵢ-μ)²

趋势强度:
slope, r² = linear_regression(t, X)
```

#### 4. 波动率
```
波动率:
σ = √(Σ(rᵢ - r̄)² / n)

RSI:
RSI = 100 - 100/(1 + RS)
RS = avg_gain / avg_loss
```

### Transformer注意力机制

```python
# 注意力权重计算
Q, K, V = Query, Key, Value

# 缩放点积注意力
Attention(Q, K, V) = softmax(QK^T / √d_k) × V

# 多头注意力
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ)W^O
where headᵢ = Attention(QWᵢ^Q, KWᵢ^K, VWᵢ^V)

# 位置编码
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

## 📁 文件结构

```
lottery_system/
│
├── lottery_core.py          # 核心算法模块 (1000+行)
│   ├── DataProcessor        # 数据处理类
│   ├── FeatureEngineering   # 特征工程类（8种）
│   ├── MLModels            # 机器学习模型类（5个）
│   ├── TransformerModel    # Transformer深度学习类
│   ├── EnsembleFusion      # 概率融合类
│   ├── BacktestEngine      # 回测引擎类
│   └── PredictionEngine    # 主预测引擎类
│
├── lottery_app.py           # Streamlit Web界面 (700+行)
│   ├── 页面配置
│   ├── 免责声明
│   ├── 侧边栏配置
│   ├── 主标签页
│   │   ├── 预测分析
│   │   ├── 特征工程
│   │   ├── 模型评估
│   │   ├── 历史回测
│   │   ├── 统计图表
│   │   └── 使用说明
│   └── 可视化组件
│
├── requirements.txt         # Python依赖列表
│
└── README.md               # 本文档
```

## ❓ 常见问题

### Q1: 如何运行系统？

**A**: 最简单的方法是使用Web界面：
```bash
streamlit run lottery_app.py
```

### Q2: 需要什么Python版本？

**A**: Python 3.8 或更高版本。推荐使用 3.9-3.11。

### Q3: Excel文件格式要求？

**A**: 文件必须包含以下列：
- 期号
- 开奖时间
- 号码1-6
- 特码

工作表名称应为「六合彩数据」。

### Q4: 预测准确率如何？

**A**: 
- 理论随机准确率: TOP1 ≈ 2%, TOP3 ≈ 6%, TOP5 ≈ 10%
- 模型准确率可能略高于随机，但**这不代表实际预测能力**
- 彩票结果完全随机，任何模型都无法改变随机性

### Q5: 可以用于实际投注吗？

**A**: **绝对不可以！**
- 本系统仅供教育研究
- 彩票完全随机，不可预测
- 任何投注都可能造成经济损失
- 请理性娱乐，远离赌博

### Q6: 如何调整预测数量？

**A**: 在Web界面左侧侧边栏：
- 融合预测数量: 5-20（默认10）
- Transformer预测数量: 5-20（默认10）

### Q7: 回测准确率有参考价值吗？

**A**: 
- 回测只反映**历史数据的统计特征**
- **不代表未来预测能力**
- 高回测准确率可能是过拟合
- 仅供学习算法评估方法

### Q8: 为什么不同模型准确率不同？

**A**: 
- 不同模型使用不同的特征和算法
- 某些模型可能更适合某些数据模式
- 但在真正的随机数据上，所有模型都无效

### Q9: 可以修改算法吗？

**A**: 完全可以！
- 所有代码开源
- 可以修改特征工程
- 可以调整模型参数
- 可以添加新的模型

### Q10: 如何贡献代码？

**A**: 
- Fork项目
- 创建新分支
- 提交Pull Request
- 遵守代码规范

## ⚠️ 最终声明

### 随机性本质

彩票开奖结果是**完全随机**的，具有以下特性：

1. **独立性**: 每次开奖互不影响
2. **均匀性**: 每个号码概率理论相等
3. **不可预测性**: 历史数据无法预测未来
4. **随机波动**: 任何模式都是随机波动

### 系统局限性

本系统的所有模型和算法：

- ❌ **无法改变随机本质**
- ❌ **无法提供准确预测**
- ❌ **不应用于实际投注**
- ✅ **仅用于演示数据分析技术**
- ✅ **仅用于教育和学术研究**

### 教育价值

本系统展示了以下数据科学技术：

1. **数据处理**: pandas, numpy
2. **特征工程**: 多维度数据分析
3. **机器学习**: 分类预测算法
4. **深度学习**: Transformer架构
5. **集成学习**: 模型融合策略
6. **时间序列**: 序列数据处理
7. **统计分析**: 概率与分布
8. **可视化**: plotly交互图表
9. **Web开发**: Streamlit应用

### 使用责任

使用本系统即表示您：

- 理解彩票的随机性本质
- 不将预测结果用于实际投注
- 仅用于学习数据科学技术
- 承担使用本系统的所有风险
- 同意遵守当地法律法规

## 📚 参考资料

- [pandas文档](https://pandas.pydata.org/)
- [numpy文档](https://numpy.org/)
- [scipy文档](https://scipy.org/)
- [streamlit文档](https://docs.streamlit.io/)
- [plotly文档](https://plotly.com/python/)

## 📧 联系与反馈

- 技术问题: 提交GitHub Issue
- 改进建议: Pull Request
- 教育合作: Email联系

---

**版本**: v2.0 Python Edition  
**作者**: AI Research Team  
**许可**: 仅供教育研究使用  
**日期**: 2026-03-08

**🎓 理性娱乐，远离赌博！**
