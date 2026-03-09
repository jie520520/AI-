"""
AI彩票量化研究系统 - 增强版核心算法模块
仅供教育和学术研究使用

新增功能:
- 大小、单双、波色的AI预测模型
- 大小、单双的历史回测
- 历史记录查看功能
- 可自定义预测数量(1-49)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
from scipy import stats
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 导入原有的基础类
from lottery_core import (
    DataProcessor, FeatureEngineering, MLModels, 
    TransformerModel, EnsembleFusion, BacktestEngine
)


# ============================================================================
# 辅助预测模型（大小、单双、波色）- 使用真实AI算法
# ============================================================================

class AuxiliaryPredictor:
    """大小、单双、波色的AI预测模型"""
    
    @staticmethod
    def predict_size(data: pd.DataFrame, features: Dict) -> List[Dict]:
        """预测大小（基于多个模型融合）"""
        recent_50 = data.iloc[-50:]
        big_count = len(recent_50[recent_50['大小'] == '大'])
        
        # 模型1: 频率统计
        freq_big_prob = big_count / 50
        
        # 模型2: 趋势分析
        trend = features['时间序列']['trend_strength']
        trend_big_prob = 0.5 + (trend * 0.3)  # 趋势影响
        
        # 模型3: 周期性分析
        recent_10 = data.iloc[-10:]
        big_10 = len(recent_10[recent_10['大小'] == '大'])
        cycle_big_prob = 0.5
        if big_10 > 7:  # 最近大数过多，预测小
            cycle_big_prob = 0.3
        elif big_10 < 3:  # 最近小数过多，预测大
            cycle_big_prob = 0.7
        
        # 模型4: RSI指标
        rsi = features['波动特征'].get('rsi_20', 50)
        rsi_big_prob = 0.3 if rsi > 70 else 0.7 if rsi < 30 else 0.5
        
        # 融合四个模型（加权平均）
        big_prob = (freq_big_prob * 0.3 + 
                   trend_big_prob * 0.3 + 
                   cycle_big_prob * 0.25 + 
                   rsi_big_prob * 0.15)
        
        big_prob = max(0.2, min(0.8, big_prob))  # 限制范围
        small_prob = 1 - big_prob
        
        results = [
            {
                '类型': '大',
                '概率': f'{big_prob*100:.2f}%',
                '置信度': '高' if big_prob > 0.65 else '中' if big_prob > 0.55 else '低',
                'raw_prob': big_prob
            },
            {
                '类型': '小',
                '概率': f'{small_prob*100:.2f}%',
                '置信度': '高' if small_prob > 0.65 else '中' if small_prob > 0.55 else '低',
                'raw_prob': small_prob
            }
        ]
        
        return sorted(results, key=lambda x: x['raw_prob'], reverse=True)
    
    @staticmethod
    def predict_odd_even(data: pd.DataFrame, features: Dict) -> List[Dict]:
        """预测单双（基于多个模型融合）"""
        recent_50 = data.iloc[-50:]
        odd_count = len(recent_50[recent_50['单双'] == '单'])
        
        # 模型1: 频率统计
        freq_odd_prob = odd_count / 50
        
        # 模型2: 周期性分析
        recent_10 = data.iloc[-10:]
        odd_10 = len(recent_10[recent_10['单双'] == '单'])
        cycle_odd_prob = 0.5
        if odd_10 > 7:
            cycle_odd_prob = 0.35
        elif odd_10 < 3:
            cycle_odd_prob = 0.65
        
        # 模型3: 模式识别
        recent_20 = data.iloc[-20:]
        odd_pattern = [1 if row['单双'] == '单' else 0 for _, row in recent_20.iterrows()]
        pattern_score = sum(odd_pattern) / 20
        
        # 融合模型
        odd_prob = (freq_odd_prob * 0.4 + 
                   cycle_odd_prob * 0.35 + 
                   pattern_score * 0.25)
        
        odd_prob = max(0.3, min(0.7, odd_prob))
        even_prob = 1 - odd_prob
        
        results = [
            {
                '类型': '单',
                '概率': f'{odd_prob*100:.2f}%',
                '置信度': '高' if odd_prob > 0.6 else '中' if odd_prob > 0.5 else '低',
                'raw_prob': odd_prob
            },
            {
                '类型': '双',
                '概率': f'{even_prob*100:.2f}%',
                '置信度': '高' if even_prob > 0.6 else '中' if even_prob > 0.5 else '低',
                'raw_prob': even_prob
            }
        ]
        
        return sorted(results, key=lambda x: x['raw_prob'], reverse=True)
    
    @staticmethod
    def predict_color(data: pd.DataFrame, features: Dict) -> List[Dict]:
        """预测波色（基于多个模型融合）"""
        recent_50 = data.iloc[-50:]
        color_counts = recent_50['波色'].value_counts()
        
        # 模型1: 频率统计
        total = len(recent_50)
        freq_probs = {}
        for color in ['红波', '蓝波', '绿波']:
            count = color_counts.get(color, 0)
            freq_probs[color] = count / total
        
        # 模型2: 趋势调整
        recent_10 = data.iloc[-10:]
        recent_color_counts = recent_10['波色'].value_counts()
        
        trend_probs = {}
        for color in freq_probs.keys():
            recent_count = recent_color_counts.get(color, 0)
            if recent_count > 5:  # 最近出现过多
                trend_probs[color] = freq_probs[color] * 0.7
            elif recent_count < 2:  # 最近出现过少
                trend_probs[color] = freq_probs[color] * 1.3
            else:
                trend_probs[color] = freq_probs[color]
        
        # 模型3: 遗漏分析
        omission_probs = {}
        for color in ['红波', '蓝波', '绿波']:
            # 计算该波色的平均遗漏
            last_idx = -1
            for i in range(len(data)-1, -1, -1):
                if data.iloc[i]['波色'] == color:
                    last_idx = i
                    break
            
            if last_idx >= 0:
                omission = len(data) - last_idx - 1
                # 遗漏越大，概率调整越大
                omission_probs[color] = 1.0 + (omission / 20)
            else:
                omission_probs[color] = 1.5
        
        # 融合三个模型
        final_probs = {}
        for color in ['红波', '蓝波', '绿波']:
            final_probs[color] = (freq_probs[color] * 0.4 + 
                                 trend_probs[color] * 0.4 + 
                                 (omission_probs[color] / sum(omission_probs.values())) * 0.2)
        
        # 归一化
        total_prob = sum(final_probs.values())
        final_probs = {k: v/total_prob for k, v in final_probs.items()}
        
        results = []
        for color, prob in final_probs.items():
            results.append({
                '类型': color,
                '概率': f'{prob*100:.2f}%',
                '置信度': '高' if prob > 0.4 else '中' if prob > 0.32 else '低',
                'raw_prob': prob
            })
        
        return sorted(results, key=lambda x: x['raw_prob'], reverse=True)


# ============================================================================
# 辅助属性回测引擎
# ============================================================================

class AuxiliaryBacktest:
    """大小、单双、波色的历史回测"""
    
    @staticmethod
    def backtest_size(data: pd.DataFrame, test_periods: int = 50) -> Dict:
        """大小回测"""
        results = []
        start_idx = len(data) - test_periods
        
        for i in range(start_idx, len(data)):
            train_data = data.iloc[:i]
            actual = data.iloc[i]
            
            # 提取特征并预测
            features = FeatureEngineering.extract_all_features(train_data, window=30)
            predictions = AuxiliaryPredictor.predict_size(train_data, features)
            
            predicted = predictions[0]['类型']
            actual_size = actual['大小']
            hit = (predicted == actual_size)
            
            results.append({
                '期号': actual['期号'],
                '预测': predicted,
                '实际': actual_size,
                '命中': hit,
                '概率': predictions[0]['概率']
            })
        
        hit_count = sum(1 for r in results if r['命中'])
        accuracy = (hit_count / len(results) * 100)
        
        return {
            'results': pd.DataFrame(results),
            'accuracy': f"{accuracy:.2f}%",
            'hit_count': hit_count,
            'total_tests': len(results),
            'type': '大小'
        }
    
    @staticmethod
    def backtest_odd_even(data: pd.DataFrame, test_periods: int = 50) -> Dict:
        """单双回测"""
        results = []
        start_idx = len(data) - test_periods
        
        for i in range(start_idx, len(data)):
            train_data = data.iloc[:i]
            actual = data.iloc[i]
            
            features = FeatureEngineering.extract_all_features(train_data, window=30)
            predictions = AuxiliaryPredictor.predict_odd_even(train_data, features)
            
            predicted = predictions[0]['类型']
            actual_odd_even = actual['单双']
            hit = (predicted == actual_odd_even)
            
            results.append({
                '期号': actual['期号'],
                '预测': predicted,
                '实际': actual_odd_even,
                '命中': hit,
                '概率': predictions[0]['概率']
            })
        
        hit_count = sum(1 for r in results if r['命中'])
        accuracy = (hit_count / len(results) * 100)
        
        return {
            'results': pd.DataFrame(results),
            'accuracy': f"{accuracy:.2f}%",
            'hit_count': hit_count,
            'total_tests': len(results),
            'type': '单双'
        }
    
    @staticmethod
    def backtest_color(data: pd.DataFrame, test_periods: int = 50) -> Dict:
        """波色回测"""
        results = []
        start_idx = len(data) - test_periods
        
        for i in range(start_idx, len(data)):
            train_data = data.iloc[:i]
            actual = data.iloc[i]
            
            features = FeatureEngineering.extract_all_features(train_data, window=30)
            predictions = AuxiliaryPredictor.predict_color(train_data, features)
            
            predicted = predictions[0]['类型']
            actual_color = actual['波色']
            hit = (predicted == actual_color)
            
            results.append({
                '期号': actual['期号'],
                '预测': predicted,
                '实际': actual_color,
                '命中': hit,
                '概率': predictions[0]['概率']
            })
        
        hit_count = sum(1 for r in results if r['命中'])
        accuracy = (hit_count / len(results) * 100)
        
        return {
            'results': pd.DataFrame(results),
            'accuracy': f"{accuracy:.2f}%",
            'hit_count': hit_count,
            'total_tests': len(results),
            'type': '波色'
        }


# ============================================================================
# 增强版预测引擎
# ============================================================================

class EnhancedPredictionEngine:
    """增强版主预测引擎 - 整合所有新功能"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.features = None
        self.predictions = None
    
    def run_prediction(self, top_k: int = 10, transformer_top_k: int = 10):
        """运行完整预测流程（支持1-49的自定义数量）"""
        print("\n" + "="*60)
        print("开始增强版AI预测分析...")
        print("="*60)
        
        # 验证参数
        top_k = max(1, min(49, top_k))
        transformer_top_k = max(1, min(49, transformer_top_k))
        
        # 1. 特征工程
        print("\n[1/5] 提取8维特征...")
        self.features = FeatureEngineering.extract_all_features(self.data)
        print("  ✓ 所有特征提取完成")
        
        # 2. 机器学习模型
        print("\n[2/5] 运行5个机器学习模型...")
        nb = MLModels.naive_bayes(self.data, self.features)
        knn = MLModels.weighted_knn(self.data, self.features)
        dt = MLModels.decision_tree(self.data, self.features)
        rf = MLModels.random_forest(self.data, self.features)
        gb = MLModels.gradient_boosting(self.data, self.features)
        print("  ✓ 所有ML模型完成")
        
        # 3. Transformer模型
        print("\n[3/5] 运行Transformer深度学习模型...")
        transformer = TransformerModel()
        transformer_result = transformer.predict(self.data, transformer_top_k)
        print(f"  ✓ Transformer完成 (置信度: {transformer_result['confidence']:.2%})")
        
        # 4. 概率融合
        print("\n[4/5] 概率融合与集成...")
        fused_prob = EnsembleFusion.fuse_predictions([nb, knn, dt, rf, gb])
        stacked_prob = EnsembleFusion.stacked_ensemble([nb, knn, dt, rf, gb], self.features)
        
        fusion_predictions = EnsembleFusion.get_top_predictions(fused_prob, top_k)
        stacked_predictions = EnsembleFusion.get_top_predictions(stacked_prob, top_k)
        print("  ✓ 融合完成")
        
        # 5. 辅助预测（使用AI模型）
        print("\n[5/5] 运行辅助AI预测（大小、单双、波色）...")
        size_predictions = AuxiliaryPredictor.predict_size(self.data, self.features)
        odd_even_predictions = AuxiliaryPredictor.predict_odd_even(self.data, self.features)
        color_predictions = AuxiliaryPredictor.predict_color(self.data, self.features)
        print("  ✓ 辅助预测完成")
        
        self.predictions = {
            '融合预测': fusion_predictions,
            '堆叠预测': stacked_predictions,
            'Transformer': transformer_result,
            '大小预测': size_predictions,
            '单双预测': odd_even_predictions,
            '波色预测': color_predictions,
            '模型原始': {
                '朴素贝叶斯': nb,
                'K近邻': knn,
                '决策树': dt,
                '随机森林': rf,
                '梯度提升': gb,
            }
        }
        
        print("\n" + "="*60)
        print("✓ 增强版预测完成!")
        print("="*60)
        
        return self.predictions
    
    def print_predictions(self):
        """打印预测结果"""
        if not self.predictions:
            print("请先运行预测!")
            return
        
        print("\n" + "="*60)
        print("AI融合预测 TOP", len(self.predictions['融合预测']))
        print("="*60)
        for i, pred in enumerate(self.predictions['融合预测'], 1):
            print(f"{i:2d}. 号码 {pred['号码']:2d}  概率 {pred['概率']:>7s}  置信度 {pred['置信度']}")
        
        print("\n" + "="*60)
        print("辅助AI预测")
        print("="*60)
        print("\n【大小预测】")
        for pred in self.predictions['大小预测']:
            print(f"  {pred['类型']:2s}  概率 {pred['概率']:>7s}  置信度 {pred['置信度']}")
        
        print("\n【单双预测】")
        for pred in self.predictions['单双预测']:
            print(f"  {pred['类型']:2s}  概率 {pred['概率']:>7s}  置信度 {pred['置信度']}")
        
        print("\n【波色预测】")
        for pred in self.predictions['波色预测']:
            print(f"  {pred['类型']:4s}  概率 {pred['概率']:>7s}  置信度 {pred['置信度']}")
    
    def get_copy_format(self) -> str:
        """生成可复制的预测格式"""
        lines = []
        lines.append("="*50)
        lines.append("AI预测结果")
        lines.append("="*50)
        
        lines.append("\n【特码预测】")
        nums = [str(pred['号码']) for pred in self.predictions['融合预测']]
        lines.append(f"号码: {' '.join(nums)}")
        
        lines.append("\n【辅助预测】")
        lines.append(f"大小: {self.predictions['大小预测'][0]['类型']} ({self.predictions['大小预测'][0]['概率']})")
        lines.append(f"单双: {self.predictions['单双预测'][0]['类型']} ({self.predictions['单双预测'][0]['概率']})")
        lines.append(f"波色: {self.predictions['波色预测'][0]['类型']} ({self.predictions['波色预测'][0]['概率']})")
        
        lines.append("\n" + "="*50)
        lines.append("⚠️ 仅供教育研究，切勿实际投注")
        lines.append("="*50)
        
        return "\n".join(lines)


# ============================================================================
# 历史记录查看器
# ============================================================================

class HistoryViewer:
    """历史记录查看和分析"""
    
    @staticmethod
    def get_recent_history(data: pd.DataFrame, periods: int = 20) -> pd.DataFrame:
        """获取最近N期历史记录"""
        recent = data.iloc[-periods:].copy()
        
        # 选择关键列
        display_cols = ['期号', '开奖时间', '特码', '大小', '单双', '波色', '和值']
        available_cols = [col for col in display_cols if col in recent.columns]
        
        return recent[available_cols]
    
    @staticmethod
    def analyze_history(data: pd.DataFrame, periods: int = 50) -> Dict:
        """分析历史数据统计"""
        recent = data.iloc[-periods:]
        
        return {
            '期数': periods,
            '大数次数': len(recent[recent['大小'] == '大']),
            '小数次数': len(recent[recent['大小'] == '小']),
            '单数次数': len(recent[recent['单双'] == '单']),
            '双数次数': len(recent[recent['单双'] == '双']),
            '红波次数': len(recent[recent['波色'] == '红波']),
            '蓝波次数': len(recent[recent['波色'] == '蓝波']),
            '绿波次数': len(recent[recent['波色'] == '绿波']),
            '平均特码': recent['特码'].mean(),
            '最大特码': recent['特码'].max(),
            '最小特码': recent['特码'].min(),
        }


# ============================================================================
# 导出所有增强功能
# ============================================================================

__all__ = [
    'AuxiliaryPredictor',
    'AuxiliaryBacktest', 
    'EnhancedPredictionEngine',
    'HistoryViewer',
    # 原有功能
    'DataProcessor',
    'FeatureEngineering',
    'MLModels',
    'TransformerModel',
    'EnsembleFusion',
    'BacktestEngine'
]
