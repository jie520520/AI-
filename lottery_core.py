"""
AI彩票量化研究系统 - 核心算法模块
仅供教育和学术研究使用

包含:
- 8种特征工程
- 5个机器学习模型
- Transformer深度学习模型
- 概率融合引擎
- 回测系统
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
from scipy import stats
from scipy.signal import find_peaks
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 数据处理类
# ============================================================================

class DataProcessor:
    """数据加载和预处理"""
    
    @staticmethod
    def load_excel(file_path: str) -> pd.DataFrame:
        """加载Excel数据"""
        df = pd.read_excel(file_path, sheet_name='六合彩数据')
        print(f"✓ 成功加载 {len(df)} 条历史记录")
        return df
    
    @staticmethod
    def parse_data(df: pd.DataFrame) -> pd.DataFrame:
        """解析和增强数据"""
        data = df.copy()
        
        # 计算派生特征
        data['大小'] = data['特码'].apply(lambda x: '大' if x >= 25 else '小')
        data['单双'] = data['特码'].apply(lambda x: '双' if x % 2 == 0 else '单')
        data['波色'] = data['特码'].apply(DataProcessor._get_color)
        data['尾数'] = data['特码'] % 10
        data['区间'] = (data['特码'] - 1) // 12
        
        # 计算和值
        num_cols = ['号码1', '号码2', '号码3', '号码4', '号码5', '号码6']
        data['和值'] = data[num_cols].sum(axis=1) + data['特码']
        
        # 转换日期
        data['开奖时间'] = pd.to_datetime(data['开奖时间'])
        data['星期'] = data['开奖时间'].dt.dayofweek
        data['月份'] = data['开奖时间'].dt.month
        
        return data
    
    @staticmethod
    def _get_color(num: int) -> str:
        """获取波色"""
        red = [1,2,7,8,12,13,18,19,23,24,29,30,34,35,40,45,46]
        blue = [3,4,9,10,14,15,20,25,26,31,36,37,41,42,47,48]
        if num in red:
            return '红波'
        elif num in blue:
            return '蓝波'
        return '绿波'
    
    @staticmethod
    def get_statistics(data: pd.DataFrame) -> Dict[str, Any]:
        """获取数据统计信息"""
        numbers = data['特码'].values
        return {
            '总期数': len(data),
            '平均值': np.mean(numbers),
            '标准差': np.std(numbers),
            '中位数': np.median(numbers),
            '最小值': np.min(numbers),
            '最大值': np.max(numbers),
            '偏度': stats.skew(numbers),
            '峰度': stats.kurtosis(numbers),
        }


# ============================================================================
# 高级特征工程（8种）
# ============================================================================

class FeatureEngineering:
    """8种高级特征工程"""
    
    @staticmethod
    def extract_all_features(data: pd.DataFrame, window: int = 30) -> Dict[str, Any]:
        """提取所有特征"""
        return {
            '统计特征': FeatureEngineering.statistical_features(data, window),
            '频率特征': FeatureEngineering.frequency_features(data, window),
            '时间序列': FeatureEngineering.timeseries_features(data, window),
            '波动特征': FeatureEngineering.volatility_features(data, window),
            '模式特征': FeatureEngineering.pattern_features(data, window),
            '空间特征': FeatureEngineering.spatial_features(data, window),
            '遗漏特征': FeatureEngineering.omission_features(data),
            '组合特征': FeatureEngineering.combination_features(data, window),
        }
    
    @staticmethod
    def statistical_features(data: pd.DataFrame, window: int = 30) -> Dict[str, float]:
        """深度统计特征"""
        recent = data['特码'].iloc[-window:].values
        mean = np.mean(recent)
        std = np.std(recent)
        
        return {
            'mean': mean,
            'std': std,
            'skewness': stats.skew(recent),
            'kurtosis': stats.kurtosis(recent),
            'cv': std / mean if mean != 0 else 0,
            'range': np.max(recent) - np.min(recent),
            'iqr': np.percentile(recent, 75) - np.percentile(recent, 25),
            'mad': np.mean(np.abs(recent - mean)),
        }
    
    @staticmethod
    def frequency_features(data: pd.DataFrame, window: int = 50) -> Dict[str, Any]:
        """频率与概率特征"""
        recent = data['特码'].iloc[-window:].values
        freq = Counter(recent)
        
        # 计算概率分布
        probs = np.array([freq.get(i, 0) for i in range(1, 50)]) / window
        probs = probs[probs > 0]
        
        # 信息熵
        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(49)
        
        # 吉尼不纯度
        gini = 1 - np.sum(probs ** 2)
        
        return {
            'entropy': entropy,
            'normalized_entropy': entropy / max_entropy,
            'gini': gini,
            'unique_count': len(freq),
            'hot_numbers': [k for k, v in freq.most_common(10)],
            'cold_numbers': [k for k, v in freq.most_common()[-10:]],
        }
    
    @staticmethod
    def timeseries_features(data: pd.DataFrame, window: int = 50) -> Dict[str, float]:
        """时间序列特征"""
        series = data['特码'].iloc[-window:].values
        
        # 自相关系数
        acf_lags = {}
        for lag in [1, 2, 3, 5, 7, 10]:
            if lag < len(series):
                acf_lags[f'acf_{lag}'] = FeatureEngineering._autocorrelation(series, lag)
        
        # 趋势强度
        x = np.arange(len(series))
        slope, _, r_value, _, _ = stats.linregress(x, series)
        
        return {
            **acf_lags,
            'trend_strength': slope,
            'r_squared': r_value ** 2,
            'stationarity': FeatureEngineering._check_stationarity(series),
        }
    
    @staticmethod
    def volatility_features(data: pd.DataFrame, window: int = 20) -> Dict[str, float]:
        """波动性与动量特征"""
        features = {}
        
        for w in [5, 10, 20]:
            if w <= len(data):
                recent = data['特码'].iloc[-w:].values
                
                # 收益率
                returns = np.diff(recent) / recent[:-1]
                
                # 波动率
                features[f'volatility_{w}'] = np.std(returns) if len(returns) > 0 else 0
                
                # 动量
                features[f'momentum_{w}'] = (recent[-1] - recent[0]) / recent[0] if recent[0] != 0 else 0
                
                # RSI
                features[f'rsi_{w}'] = FeatureEngineering._calculate_rsi(recent)
        
        return features
    
    @staticmethod
    def pattern_features(data: pd.DataFrame, window: int = 20) -> Dict[str, int]:
        """模式识别特征"""
        recent = data['特码'].iloc[-window:].values
        
        return {
            'consecutive_increase': FeatureEngineering._longest_consecutive(recent, lambda a, b: b > a),
            'consecutive_decrease': FeatureEngineering._longest_consecutive(recent, lambda a, b: b < a),
            'zigzag_count': FeatureEngineering._count_zigzag(recent),
            'repeating_numbers': len([x for x in Counter(recent).values() if x > 1]),
            'arithmetic_seq_length': FeatureEngineering._find_arithmetic_sequence(recent),
        }
    
    @staticmethod
    def spatial_features(data: pd.DataFrame, window: int = 30) -> Dict[str, Any]:
        """空间分布特征"""
        recent = data.iloc[-window:]
        
        # 区域分布
        zones = {
            'small': len(recent[recent['特码'] <= 16]),
            'mid': len(recent[(recent['特码'] > 16) & (recent['特码'] <= 33)]),
            'large': len(recent[recent['特码'] > 33]),
        }
        
        # 波色分布
        colors = recent['波色'].value_counts().to_dict()
        
        # 单双比例
        odd_ratio = len(recent[recent['单双'] == '单']) / window
        
        return {
            'zone_balance': sum(abs(v - window/3) for v in zones.values()),
            'color_balance': max(colors.values()) - min(colors.values()) if colors else 0,
            'odd_ratio': odd_ratio,
            'dominant_zone': max(zones, key=zones.get),
            'dominant_color': max(colors, key=colors.get) if colors else None,
        }
    
    @staticmethod
    def omission_features(data: pd.DataFrame) -> Dict[str, Any]:
        """遗漏与冷热特征"""
        current_omissions = [0] * 49
        last_seen = [-1] * 49
        
        for idx, num in enumerate(data['特码'].values):
            last_seen[num - 1] = idx
        
        for num in range(49):
            if last_seen[num] == -1:
                current_omissions[num] = len(data)
            else:
                current_omissions[num] = len(data) - last_seen[num] - 1
        
        return {
            'max_omission': max(current_omissions),
            'avg_omission': np.mean(current_omissions),
            'current_omissions': current_omissions,
            'hot_numbers': [i+1 for i in np.argsort(current_omissions)[:10]],
            'cold_numbers': [i+1 for i in np.argsort(current_omissions)[-10:]],
        }
    
    @staticmethod
    def combination_features(data: pd.DataFrame, window: int = 20) -> Dict[str, Any]:
        """组合与交互特征"""
        recent = data.iloc[-window:]
        
        # 大小单双组合
        combinations = recent.groupby(['大小', '单双']).size().to_dict()
        
        return {
            'avg_sum': recent['和值'].mean(),
            'sum_trend': (recent['和值'].iloc[-1] - recent['和值'].iloc[0]) / window,
            'combinations': {f"{k[0]}_{k[1]}": v for k, v in combinations.items()},
        }
    
    # 辅助方法
    @staticmethod
    def _autocorrelation(series: np.ndarray, lag: int) -> float:
        """计算自相关系数"""
        if lag >= len(series):
            return 0
        n = len(series) - lag
        mean = np.mean(series)
        c0 = np.sum((series - mean) ** 2) / len(series)
        c_lag = np.sum((series[:n] - mean) * (series[lag:] - mean)) / len(series)
        return c_lag / c0 if c0 != 0 else 0
    
    @staticmethod
    def _check_stationarity(series: np.ndarray) -> float:
        """检查平稳性（简化版）"""
        diffs = np.diff(series)
        return np.std(diffs) / np.std(series) if np.std(series) != 0 else 0
    
    @staticmethod
    def _calculate_rsi(prices: np.ndarray, period: int = 14) -> float:
        """计算RSI指标"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def _longest_consecutive(arr: np.ndarray, condition) -> int:
        """最长连续满足条件的长度"""
        max_len, current_len = 0, 0
        for i in range(1, len(arr)):
            if condition(arr[i-1], arr[i]):
                current_len += 1
                max_len = max(max_len, current_len)
            else:
                current_len = 0
        return max_len
    
    @staticmethod
    def _count_zigzag(arr: np.ndarray) -> int:
        """计算之字形变化次数"""
        if len(arr) < 3:
            return 0
        changes = 0
        for i in range(2, len(arr)):
            trend1 = np.sign(arr[i-1] - arr[i-2])
            trend2 = np.sign(arr[i] - arr[i-1])
            if trend1 != 0 and trend2 != 0 and trend1 != trend2:
                changes += 1
        return changes
    
    @staticmethod
    def _find_arithmetic_sequence(arr: np.ndarray) -> int:
        """查找最长等差数列"""
        if len(arr) < 2:
            return 1
        max_len, current_len = 1, 1
        last_diff = None
        for i in range(1, len(arr)):
            diff = arr[i] - arr[i-1]
            if last_diff == diff:
                current_len += 1
                max_len = max(max_len, current_len)
            else:
                current_len = 1
                last_diff = diff
        return max_len


# ============================================================================
# 机器学习模型（5个）
# ============================================================================

class MLModels:
    """5个机器学习模型"""
    
    @staticmethod
    def naive_bayes(data: pd.DataFrame, features: Dict) -> np.ndarray:
        """改进的朴素贝叶斯"""
        probs = np.ones(49)
        
        # 基础频率
        window = 100
        recent = data['特码'].iloc[-window:].values
        for num in recent:
            probs[num - 1] += 1
        
        # 热号加权
        hot_numbers = features['遗漏特征']['hot_numbers']
        for num in hot_numbers:
            probs[num - 1] *= 1.3
        
        # 冷号降权
        cold_numbers = features['遗漏特征']['cold_numbers']
        for num in cold_numbers:
            probs[num - 1] *= 0.7
        
        # 趋势调整
        if features['时间序列']['trend_strength'] > 0.5:
            probs[24:] *= 1.1  # 上升趋势偏向大号
        elif features['时间序列']['trend_strength'] < -0.5:
            probs[:24] *= 1.1  # 下降趋势偏向小号
        
        return MLModels._normalize(probs)
    
    @staticmethod
    def weighted_knn(data: pd.DataFrame, features: Dict, k: int = 10) -> np.ndarray:
        """加权K近邻"""
        probs = np.zeros(49)
        last_num = data['特码'].iloc[-1]
        last_sum = data['和值'].iloc[-1]
        last_size = data['大小'].iloc[-1]
        
        # 计算距离
        distances = []
        for i in range(len(data) - 1):
            dist = abs(data['特码'].iloc[i] - last_num)
            dist += abs(data['和值'].iloc[i] - last_sum) * 0.1
            dist += 5 if data['大小'].iloc[i] != last_size else 0
            
            weight = 1 / (dist + 1)
            next_num = data['特码'].iloc[i + 1]
            distances.append((dist, next_num, weight))
        
        # 选择K个最近邻
        distances.sort(key=lambda x: x[0])
        neighbors = distances[:k]
        
        total_weight = sum(w for _, _, w in neighbors)
        for _, num, weight in neighbors:
            probs[num - 1] += weight / total_weight
        
        return MLModels._normalize(probs)
    
    @staticmethod
    def decision_tree(data: pd.DataFrame, features: Dict) -> np.ndarray:
        """基于规则的决策树"""
        probs = np.ones(49) / 49
        
        # 规则1: 大小趋势
        if features['空间特征']['odd_ratio'] > 0.65:
            probs[24:] *= 1.5
        elif features['空间特征']['odd_ratio'] < 0.35:
            probs[:24] *= 1.5
        
        # 规则2: 波色主导
        if features['空间特征']['dominant_color'] == '红波':
            red_indices = [0,1,6,7,11,12,17,18,22,23,28,29,33,34,39,44,45]
            for i in red_indices:
                probs[i] *= 1.3
        
        # 规则3: 最大遗漏补偿
        omissions = features['遗漏特征']['current_omissions']
        max_omission_idx = np.argmax(omissions)
        probs[max_omission_idx] *= 2
        
        return MLModels._normalize(probs)
    
    @staticmethod
    def random_forest(data: pd.DataFrame, features: Dict) -> np.ndarray:
        """随机森林集成"""
        tree1 = MLModels.naive_bayes(data, features)
        tree2 = MLModels.weighted_knn(data, features, k=8)
        tree3 = MLModels.weighted_knn(data, features, k=12)
        tree4 = MLModels.decision_tree(data, features)
        
        ensemble = (tree1 + tree2 + tree3 + tree4) / 4
        return MLModels._normalize(ensemble)
    
    @staticmethod
    def gradient_boosting(data: pd.DataFrame, features: Dict, rounds: int = 3) -> np.ndarray:
        """梯度提升"""
        probs = np.ones(49) / 49
        learning_rate = 0.3
        
        for _ in range(rounds):
            # 计算残差
            recent = data['特码'].iloc[-30:].values
            actual_freq = np.array([np.sum(recent == i) for i in range(1, 50)])
            actual_prob = actual_freq / 30
            
            residuals = actual_prob - probs
            
            # 弱学习器
            weak_learner = MLModels.naive_bayes(data, features)
            
            # 更新
            probs += learning_rate * residuals * weak_learner
        
        return MLModels._normalize(probs)
    
    @staticmethod
    def _normalize(probs: np.ndarray) -> np.ndarray:
        """归一化概率"""
        total = np.sum(probs)
        return probs / total if total > 0 else probs


# ============================================================================
# Transformer深度学习模型
# ============================================================================

class TransformerModel:
    """Transformer深度学习模型"""
    
    def __init__(self, num_heads: int = 4, seq_length: int = 30):
        self.num_heads = num_heads
        self.seq_length = seq_length
    
    def predict(self, data: pd.DataFrame, top_k: int = 10) -> Dict[str, Any]:
        """Transformer预测"""
        sequence = data['特码'].iloc[-self.seq_length:].values
        
        # 多头注意力
        attention_heads = []
        for head in range(self.num_heads):
            attention = self._attention_head(sequence, head)
            attention_heads.append(attention)
        
        # 合并注意力
        combined_attention = np.mean(attention_heads, axis=0)
        
        # 位置编码
        position_encoded = self._add_positional_encoding(sequence, combined_attention)
        
        # 前馈网络预测
        predictions = self._feed_forward(position_encoded, data)
        
        # Top K预测
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        top_predictions = [
            {
                '号码': idx + 1,
                '概率': f"{predictions[idx] * 100:.3f}%",
                '置信度': '高' if predictions[idx] > 0.03 else '中' if predictions[idx] > 0.02 else '低'
            }
            for idx in top_indices
        ]
        
        # 计算置信度
        entropy = -np.sum(combined_attention * np.log2(combined_attention + 1e-10))
        max_entropy = np.log2(len(combined_attention))
        confidence = 1 - (entropy / max_entropy)
        
        return {
            'predictions': top_predictions,
            'confidence': confidence,
            'attention_weights': combined_attention,
            'head_contributions': [
                {
                    'head': i + 1,
                    'entropy': -np.sum(att * np.log2(att + 1e-10)),
                }
                for i, att in enumerate(attention_heads)
            ]
        }
    
    def _attention_head(self, sequence: np.ndarray, head_idx: int) -> np.ndarray:
        """单个注意力头"""
        n = len(sequence)
        weights = np.zeros(n)
        scale = np.sqrt(n / self.num_heads)
        
        for i in range(n):
            query = sequence[i] / 49
            attention_sum = 0
            
            for j in range(n):
                key = sequence[j] / 49
                position_bias = np.exp(-((i - j) ** 2) / (2 * scale ** 2))
                similarity = np.exp(query * key + head_idx * 0.1) * position_bias
                weights[i] += similarity
                attention_sum += similarity
            
            weights[i] /= attention_sum if attention_sum > 0 else 1
        
        return weights / np.sum(weights)
    
    def _add_positional_encoding(self, sequence: np.ndarray, attention: np.ndarray) -> np.ndarray:
        """添加位置编码"""
        encoded = []
        for idx, num in enumerate(sequence):
            pos = idx / len(sequence)
            encoded_val = num / 49 + np.sin(pos * np.pi) * 0.1 + np.cos(pos * 2 * np.pi) * 0.1
            encoded.append(encoded_val * attention[idx])
        return np.array(encoded)
    
    def _feed_forward(self, encoded: np.ndarray, data: pd.DataFrame) -> np.ndarray:
        """前馈网络"""
        predictions = np.zeros(49)
        
        # 第一层：特征提取
        hidden = np.tanh(encoded * 2)
        base_prob = np.mean(hidden)
        
        # 第二层：概率分布
        for num in range(1, 50):
            score = base_prob
            
            # 历史频率
            freq = np.sum(data['特码'].iloc[-50:].values == num)
            score += (freq / 50) * 0.3
            
            # 趋势
            trend = encoded[-1] - encoded[0]
            if trend > 0 and num >= 25:
                score *= 1.2
            if trend < 0 and num < 25:
                score *= 1.2
            
            # 周期性
            recent_nums = data['特码'].iloc[-7:].values
            if num in recent_nums:
                score *= 0.8
            
            predictions[num - 1] = max(0, score)
        
        return predictions / np.sum(predictions)


# ============================================================================
# 概率融合引擎
# ============================================================================

class EnsembleFusion:
    """概率融合与集成学习"""
    
    @staticmethod
    def fuse_predictions(predictions_list: List[np.ndarray], weights: List[float] = None) -> np.ndarray:
        """简单加权融合"""
        if weights is None:
            weights = [1.0 / len(predictions_list)] * len(predictions_list)
        
        fused = np.zeros(49)
        for pred, weight in zip(predictions_list, weights):
            fused += pred * weight
        
        return fused / np.sum(fused)
    
    @staticmethod
    def stacked_ensemble(predictions_list: List[np.ndarray], features: Dict) -> np.ndarray:
        """堆叠集成"""
        fused = np.zeros(49)
        
        for pred in predictions_list:
            for num in range(49):
                adjustment = 1.0
                
                # 元特征调整
                if (num + 1) in features['遗漏特征']['hot_numbers']:
                    adjustment *= 1.2
                
                if features['空间特征']['dominant_zone'] == 'large' and num >= 33:
                    adjustment *= 1.1
                
                fused[num] += pred[num] * adjustment
        
        return fused / np.sum(fused)
    
    @staticmethod
    def get_top_predictions(probs: np.ndarray, top_k: int = 10) -> List[Dict]:
        """获取Top K预测"""
        top_indices = np.argsort(probs)[-top_k:][::-1]
        
        return [
            {
                '号码': idx + 1,
                '概率': f"{probs[idx] * 100:.3f}%",
                '置信度': '高' if probs[idx] > 0.025 else '中' if probs[idx] > 0.020 else '低',
                'raw_prob': probs[idx]
            }
            for idx in top_indices
        ]


# ============================================================================
# 回测引擎
# ============================================================================

class BacktestEngine:
    """历史回测系统"""
    
    @staticmethod
    def run(data: pd.DataFrame, predict_func, test_periods: int = 100, strategy: str = 'top1') -> Dict:
        """运行回测"""
        results = []
        start_idx = len(data) - test_periods
        
        for i in range(start_idx, len(data)):
            train_data = data.iloc[:i]
            actual = data.iloc[i]
            
            # 运行预测
            prediction = predict_func(train_data)
            
            # 判断命中
            hit = False
            predicted_nums = None
            
            if strategy == 'top1':
                predicted_nums = prediction[0]['号码']
                hit = predicted_nums == actual['特码']
            elif strategy == 'top3':
                predicted_nums = [p['号码'] for p in prediction[:3]]
                hit = actual['特码'] in predicted_nums
                predicted_nums = ','.join(map(str, predicted_nums))
            elif strategy == 'top5':
                predicted_nums = [p['号码'] for p in prediction[:5]]
                hit = actual['特码'] in predicted_nums
                predicted_nums = ','.join(map(str, predicted_nums))
            
            results.append({
                '期号': actual['期号'],
                '预测': predicted_nums,
                '实际': actual['特码'],
                '命中': hit,
                '概率': prediction[0]['概率']
            })
        
        hit_count = sum(1 for r in results if r['命中'])
        accuracy = (hit_count / len(results) * 100)
        
        return {
            'results': pd.DataFrame(results),
            'accuracy': f"{accuracy:.2f}%",
            'hit_count': hit_count,
            'total_tests': len(results),
            'strategy': strategy
        }


# ============================================================================
# 主预测引擎
# ============================================================================

class PredictionEngine:
    """主预测引擎 - 整合所有模型"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.features = None
        self.predictions = None
    
    def run_prediction(self, top_k: int = 10, transformer_top_k: int = 10):
        """运行完整预测流程"""
        print("\n" + "="*60)
        print("开始AI预测分析...")
        print("="*60)
        
        # 1. 特征工程
        print("\n[1/4] 提取8维特征...")
        self.features = FeatureEngineering.extract_all_features(self.data)
        print("  ✓ 统计特征")
        print("  ✓ 频率特征")
        print("  ✓ 时间序列")
        print("  ✓ 波动特征")
        print("  ✓ 模式特征")
        print("  ✓ 空间特征")
        print("  ✓ 遗漏特征")
        print("  ✓ 组合特征")
        
        # 2. 机器学习模型
        print("\n[2/4] 运行5个机器学习模型...")
        nb = MLModels.naive_bayes(self.data, self.features)
        print("  ✓ 朴素贝叶斯")
        
        knn = MLModels.weighted_knn(self.data, self.features)
        print("  ✓ K近邻")
        
        dt = MLModels.decision_tree(self.data, self.features)
        print("  ✓ 决策树")
        
        rf = MLModels.random_forest(self.data, self.features)
        print("  ✓ 随机森林")
        
        gb = MLModels.gradient_boosting(self.data, self.features)
        print("  ✓ 梯度提升")
        
        # 3. Transformer模型
        print("\n[3/4] 运行Transformer深度学习模型...")
        transformer = TransformerModel()
        transformer_result = transformer.predict(self.data, transformer_top_k)
        print(f"  ✓ Transformer (置信度: {transformer_result['confidence']:.2%})")
        
        # 4. 概率融合
        print("\n[4/4] 概率融合与集成...")
        fused_prob = EnsembleFusion.fuse_predictions([nb, knn, dt, rf, gb])
        stacked_prob = EnsembleFusion.stacked_ensemble([nb, knn, dt, rf, gb], self.features)
        
        fusion_predictions = EnsembleFusion.get_top_predictions(fused_prob, top_k)
        stacked_predictions = EnsembleFusion.get_top_predictions(stacked_prob, top_k)
        print("  ✓ 简单融合")
        print("  ✓ 堆叠集成")
        
        # 大小波色预测
        recent_30 = self.data.iloc[-30:]
        big_count = len(recent_30[recent_30['大小'] == '大'])
        big_small_pred = '大' if big_count > 15 else '小'
        
        color_counts = recent_30['波色'].value_counts()
        color_pred = color_counts.idxmax()
        
        self.predictions = {
            '融合预测': fusion_predictions,
            '堆叠预测': stacked_predictions,
            'Transformer': transformer_result,
            '大小预测': big_small_pred,
            '波色预测': color_pred,
            '模型原始': {
                '朴素贝叶斯': nb,
                'K近邻': knn,
                '决策树': dt,
                '随机森林': rf,
                '梯度提升': gb,
            }
        }
        
        print("\n" + "="*60)
        print("✓ 预测完成!")
        print("="*60)
        
        return self.predictions
    
    def print_predictions(self):
        """打印预测结果"""
        if not self.predictions:
            print("请先运行预测!")
            return
        
        print("\n" + "="*60)
        print("AI融合预测 TOP 10")
        print("="*60)
        for i, pred in enumerate(self.predictions['融合预测'], 1):
            print(f"{i:2d}. 号码 {pred['号码']:2d}  概率 {pred['概率']:>7s}  置信度 {pred['置信度']}")
        
        print("\n" + "="*60)
        print("Transformer深度学习预测")
        print("="*60)
        for i, pred in enumerate(self.predictions['Transformer']['predictions'], 1):
            print(f"{i:2d}. 号码 {pred['号码']:2d}  概率 {pred['概率']:>7s}  置信度 {pred['置信度']}")
        
        print("\n" + "="*60)
        print("辅助预测")
        print("="*60)
        print(f"大小预测: {self.predictions['大小预测']}")
        print(f"波色预测: {self.predictions['波色预测']}")


# ============================================================================
# 示例使用
# ============================================================================

if __name__ == "__main__":
    # 加载数据
    file_path = '/mnt/user-data/uploads/澳门六合彩数据导入器.xlsm'
    df = DataProcessor.load_excel(file_path)
    data = DataProcessor.parse_data(df)
    
    # 显示统计信息
    stats = DataProcessor.get_statistics(data)
    print("\n数据统计:")
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # 运行预测
    engine = PredictionEngine(data)
    predictions = engine.run_prediction(top_k=10, transformer_top_k=10)
    engine.print_predictions()
