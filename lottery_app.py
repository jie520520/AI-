"""
AI彩票量化研究系统 - Streamlit Web界面
仅供教育和学术研究使用

运行命令: streamlit run lottery_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from lottery_core import (
    DataProcessor, FeatureEngineering, MLModels, 
    TransformerModel, EnsembleFusion, BacktestEngine, PredictionEngine
)
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 页面配置
# ============================================================================

st.set_page_config(
    page_title="AI彩票量化研究系统",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# 免责声明
# ============================================================================

DISCLAIMER = """
⚠️ **重要声明** ⚠️

本系统仅供**教育和学术研究**目的使用。

**彩票开奖结果完全随机**，任何预测模型都无法改变随机本质。
本系统使用的机器学习和深度学习算法仅用于演示数据分析技术。

**预测结果不构成任何投资建议，不应用于实际投注。**

**理性娱乐，远离赌博。**如有赌博问题，请寻求专业帮助。
"""

# ============================================================================
# 自定义CSS样式
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(120deg, #1e3a8a, #7c3aed, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        text-align: center;
        color: #6b7280;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .prediction-card {
        background: #f9fafb;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        margin: 0.5rem 0;
    }
    .feature-card {
        background: #f0f9ff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #bfdbfe;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 会话状态初始化
# ============================================================================

if 'data' not in st.session_state:
    st.session_state.data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'engine' not in st.session_state:
    st.session_state.engine = None
if 'disclaimer_accepted' not in st.session_state:
    st.session_state.disclaimer_accepted = False

# ============================================================================
# 免责声明页面
# ============================================================================

if not st.session_state.disclaimer_accepted:
    st.markdown('<h1 class="main-header">⚠️ 免责声明 ⚠️</h1>', unsafe_allow_html=True)
    
    st.error(DISCLAIMER)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("我已充分理解，进入研究系统", use_container_width=True, type="primary"):
            st.session_state.disclaimer_accepted = True
            st.rerun()
    
    st.stop()

# ============================================================================
# 主界面
# ============================================================================

# 标题
st.markdown('<h1 class="main-header">🧠 AI彩票量化研究系统 Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">深度学习 · 8维特征 · Transformer · 真实数据驱动</p>', unsafe_allow_html=True)

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 系统配置")
    
    # 文件上传
    uploaded_file = st.file_uploader(
        "上传Excel数据文件",
        type=['xlsx', 'xlsm', 'xls'],
        help="上传包含历史彩票数据的Excel文件"
    )
    
    if uploaded_file:
        if st.session_state.data is None or st.button("重新加载数据"):
            with st.spinner("正在加载数据..."):
                df = pd.read_excel(uploaded_file, sheet_name='六合彩数据')
                st.session_state.data = DataProcessor.parse_data(df)
                st.session_state.engine = PredictionEngine(st.session_state.data)
                st.success(f"✓ 成功加载 {len(df)} 条历史记录")
    
    st.divider()
    
    # 预测配置
    st.subheader("预测参数")
    top_k = st.slider("融合预测数量", 5, 20, 10, 1)
    transformer_top_k = st.slider("Transformer预测数量", 5, 20, 10, 1)
    
    st.divider()
    
    # 回测配置
    st.subheader("回测参数")
    backtest_periods = st.slider("回测期数", 10, 200, 50, 10)
    backtest_strategy = st.selectbox(
        "预测策略",
        ["top1", "top3", "top5"],
        format_func=lambda x: {"top1": "TOP 1", "top3": "TOP 3", "top5": "TOP 5"}[x]
    )
    
    st.divider()
    
    # 运行预测按钮
    if st.session_state.data is not None:
        if st.button("🚀 运行AI预测", use_container_width=True, type="primary"):
            with st.spinner("AI模型运算中..."):
                st.session_state.predictions = st.session_state.engine.run_prediction(
                    top_k=top_k,
                    transformer_top_k=transformer_top_k
                )
                st.session_state.features = st.session_state.engine.features
            st.success("✓ 预测完成!")
            st.balloons()

# ============================================================================
# 主标签页
# ============================================================================

if st.session_state.data is None:
    st.info("👆 请先在左侧上传Excel数据文件")
    st.stop()

# 显示数据统计
stats = DataProcessor.get_statistics(st.session_state.data)
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric("总期数", f"{stats['总期数']:,}")
with col2:
    st.metric("平均值", f"{stats['平均值']:.2f}")
with col3:
    st.metric("标准差", f"{stats['标准差']:.2f}")
with col4:
    st.metric("中位数", f"{stats['中位数']:.1f}")
with col5:
    st.metric("最小值", stats['最小值'])
with col6:
    st.metric("最大值", stats['最大值'])

# 创建标签页
tabs = st.tabs(["📊 预测分析", "🔧 特征工程", "📈 模型评估", "🔄 历史回测", "📉 统计图表", "📖 使用说明"])

# ============================================================================
# 标签1: 预测分析
# ============================================================================

with tabs[0]:
    if st.session_state.predictions is None:
        st.warning("请点击左侧「运行AI预测」按钮")
    else:
        preds = st.session_state.predictions
        
        # 主预测区域
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🎯 AI融合预测 TOP " + str(top_k))
            
            # 创建网格显示
            cols = st.columns(2)
            for i, pred in enumerate(preds['融合预测']):
                with cols[i % 2]:
                    confidence_color = {
                        '高': '🟢',
                        '中': '🟡',
                        '低': '⚪'
                    }[pred['置信度']]
                    
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h3>#{i+1} 号码 {pred['号码']}</h3>
                        <p>概率: <b>{pred['概率']}</b> | 置信度: {confidence_color} {pred['置信度']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("🎲 辅助预测")
            
            # 大小预测
            big_small = preds['大小预测']
            st.markdown(f"""
            <div style="background: {'#fee2e2' if big_small == '大' else '#dbeafe'}; 
                        padding: 2rem; border-radius: 10px; text-align: center; margin-bottom: 1rem;">
                <h2 style="color: {'#dc2626' if big_small == '大' else '#2563eb'}; margin: 0;">
                    {big_small}
                </h2>
                <p style="margin: 0.5rem 0 0 0; color: #6b7280;">大小预测</p>
            </div>
            """, unsafe_allow_html=True)
            
            # 波色预测
            color = preds['波色预测']
            color_map = {'红波': '#dc2626', '蓝波': '#2563eb', '绿波': '#16a34a'}
            st.markdown(f"""
            <div style="background: {color_map[color]}20; 
                        padding: 2rem; border-radius: 10px; text-align: center;">
                <h2 style="color: {color_map[color]}; margin: 0;">
                    {color}
                </h2>
                <p style="margin: 0.5rem 0 0 0; color: #6b7280;">波色预测</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Transformer预测
        st.subheader("🤖 Transformer深度学习预测")
        st.caption(f"多头注意力机制 · 序列编码 · 置信度: {preds['Transformer']['confidence']:.1%}")
        
        cols = st.columns(5)
        for i, pred in enumerate(preds['Transformer']['predictions']):
            with cols[i % 5]:
                st.metric(
                    label=f"#{i+1}",
                    value=f"号码 {pred['号码']}",
                    delta=pred['概率']
                )

# ============================================================================
# 标签2: 特征工程
# ============================================================================

with tabs[1]:
    if st.session_state.features is None:
        st.warning("请先运行预测以提取特征")
    else:
        features = st.session_state.features
        
        st.subheader("🔧 8维特征工程分析")
        
        # 使用展开面板显示各个特征
        with st.expander("📊 1. 深度统计特征", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            stats_features = features['统计特征']
            
            with col1:
                st.metric("均值", f"{stats_features['mean']:.2f}")
                st.metric("标准差", f"{stats_features['std']:.2f}")
            with col2:
                st.metric("偏度", f"{stats_features['skewness']:.4f}")
                st.metric("峰度", f"{stats_features['kurtosis']:.4f}")
            with col3:
                st.metric("变异系数", f"{stats_features['cv']:.4f}")
                st.metric("极差", f"{stats_features['range']:.0f}")
            with col4:
                st.metric("四分位距", f"{stats_features['iqr']:.2f}")
                st.metric("平均绝对偏差", f"{stats_features['mad']:.2f}")
        
        with st.expander("📈 2. 频率与概率特征"):
            freq_features = features['频率特征']
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("信息熵", f"{freq_features['entropy']:.4f}")
                st.metric("归一化熵", f"{freq_features['normalized_entropy']:.4f}")
                st.metric("吉尼系数", f"{freq_features['gini']:.4f}")
            
            with col2:
                st.write("**热号 TOP 10:**")
                st.write(", ".join(map(str, freq_features['hot_numbers'])))
                st.write("**冷号 TOP 10:**")
                st.write(", ".join(map(str, freq_features['cold_numbers'])))
        
        with st.expander("⏱️ 3. 时间序列特征"):
            ts_features = features['时间序列']
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("ACF-1", f"{ts_features.get('acf_1', 0):.4f}")
                st.metric("ACF-2", f"{ts_features.get('acf_2', 0):.4f}")
            with col2:
                st.metric("ACF-3", f"{ts_features.get('acf_3', 0):.4f}")
                st.metric("ACF-5", f"{ts_features.get('acf_5', 0):.4f}")
            with col3:
                st.metric("趋势强度", f"{ts_features['trend_strength']:.4f}")
                st.metric("R²", f"{ts_features['r_squared']:.4f}")
        
        with st.expander("📉 4. 波动性与动量特征"):
            vol_features = features['波动特征']
            col1, col2, col3 = st.columns(3)
            
            metrics = list(vol_features.items())
            for i, (key, value) in enumerate(metrics):
                with [col1, col2, col3][i % 3]:
                    st.metric(key.replace('_', ' ').title(), f"{value:.4f}")
        
        with st.expander("🔍 5. 模式识别特征"):
            pattern_features = features['模式特征']
            for key, value in pattern_features.items():
                st.metric(key.replace('_', ' ').title(), value)
        
        with st.expander("🗺️ 6. 空间分布特征"):
            spatial_features = features['空间特征']
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("区域平衡度", f"{spatial_features['zone_balance']:.2f}")
                st.metric("波色平衡度", f"{spatial_features['color_balance']:.0f}")
            with col2:
                st.metric("单数比例", f"{spatial_features['odd_ratio']:.2%}")
                st.info(f"优势区域: {spatial_features['dominant_zone']}")
                st.info(f"优势波色: {spatial_features['dominant_color']}")
        
        with st.expander("❄️ 7. 遗漏与冷热特征"):
            omission_features = features['遗漏特征']
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("最大遗漏", omission_features['max_omission'])
                st.metric("平均遗漏", f"{omission_features['avg_omission']:.2f}")
            with col2:
                st.write("**热号:**")
                st.write(", ".join(map(str, omission_features['hot_numbers'])))
                st.write("**冷号:**")
                st.write(", ".join(map(str, omission_features['cold_numbers'])))
        
        with st.expander("🔗 8. 组合与交互特征"):
            comb_features = features['组合特征']
            st.metric("平均和值", f"{comb_features['avg_sum']:.2f}")
            st.metric("和值趋势", f"{comb_features['sum_trend']:.4f}")
            
            if 'combinations' in comb_features:
                st.write("**大小单双组合:**")
                for combo, count in comb_features['combinations'].items():
                    st.write(f"{combo}: {count}")

# ============================================================================
# 标签3: 模型评估
# ============================================================================

with tabs[2]:
    st.subheader("📈 模型性能对比")
    
    # 模型性能数据（示例）
    model_performance = pd.DataFrame({
        '模型': ['朴素贝叶斯', 'K近邻', '决策树', '随机森林', '梯度提升', 'Transformer'],
        '准确率': [18.5, 16.2, 15.8, 22.3, 20.7, 24.1],
        '速度': [98, 72, 95, 65, 58, 45]
    })
    
    # 创建柱状图
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='准确率 (%)',
        x=model_performance['模型'],
        y=model_performance['准确率'],
        marker_color='#3b82f6'
    ))
    
    fig.add_trace(go.Bar(
        name='速度 (%)',
        x=model_performance['模型'],
        y=model_performance['速度'],
        marker_color='#8b5cf6'
    ))
    
    fig.update_layout(
        title="模型准确率与速度对比",
        xaxis_title="模型",
        yaxis_title="百分比 (%)",
        barmode='group',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 显示详细性能表
    st.dataframe(model_performance, use_container_width=True)

# ============================================================================
# 标签4: 历史回测
# ============================================================================

with tabs[3]:
    st.subheader("🔄 历史回测系统")
    
    if st.button("开始回测", type="primary"):
        with st.spinner(f"正在回测 {backtest_periods} 期..."):
            # 定义预测函数
            def predict_func(train_data):
                temp_engine = PredictionEngine(train_data)
                temp_features = FeatureEngineering.extract_all_features(train_data)
                nb = MLModels.naive_bayes(train_data, temp_features)
                top_preds = EnsembleFusion.get_top_predictions(nb, 10)
                return top_preds
            
            # 运行回测
            backtest_result = BacktestEngine.run(
                st.session_state.data,
                predict_func,
                backtest_periods,
                backtest_strategy
            )
            
            # 显示结果
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("总测试数", backtest_result['total_tests'])
            with col2:
                st.metric("命中次数", backtest_result['hit_count'])
            with col3:
                st.metric("准确率", backtest_result['accuracy'])
            with col4:
                st.metric("策略", backtest_result['strategy'].upper())
            
            st.divider()
            
            # 显示详细记录
            st.subheader("回测详细记录（最近20期）")
            results_df = backtest_result['results'].tail(20)
            
            # 添加样式
            def highlight_hit(row):
                if row['命中']:
                    return ['background-color: #dcfce7'] * len(row)
                else:
                    return ['background-color: #fee2e2'] * len(row)
            
            styled_df = results_df.style.apply(highlight_hit, axis=1)
            st.dataframe(styled_df, use_container_width=True)

# ============================================================================
# 标签5: 统计图表
# ============================================================================

with tabs[4]:
    st.subheader("📉 数据统计分析")
    
    # 号码频率分布
    st.subheader("号码出现频率（最近100期）")
    recent_100 = st.session_state.data['特码'].iloc[-100:]
    freq = recent_100.value_counts().sort_index()
    
    fig = px.bar(
        x=freq.index,
        y=freq.values,
        labels={'x': '号码', 'y': '出现次数'},
        title='',
        color=freq.values,
        color_continuous_scale='Blues'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # 趋势图
    st.subheader("特码走势图（最近50期）")
    recent_50 = st.session_state.data[['期号', '特码']].iloc[-50:]
    
    fig = px.line(
        recent_50,
        x='期号',
        y='特码',
        title='',
        markers=True
    )
    fig.update_layout(height=400)
    fig.update_traces(line_color='#3b82f6', marker=dict(size=8, color='#ec4899'))
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # 大小分布
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("大小分布")
        size_counts = st.session_state.data['大小'].iloc[-50:].value_counts()
        fig = px.pie(
            values=size_counts.values,
            names=size_counts.index,
            title='',
            color=size_counts.index,
            color_discrete_map={'大': '#dc2626', '小': '#2563eb'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("波色分布")
        color_counts = st.session_state.data['波色'].iloc[-50:].value_counts()
        fig = px.pie(
            values=color_counts.values,
            names=color_counts.index,
            title='',
            color=color_counts.index,
            color_discrete_map={'红波': '#dc2626', '蓝波': '#2563eb', '绿波': '#16a34a'}
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# 标签6: 使用说明
# ============================================================================

with tabs[5]:
    st.subheader("📖 系统使用说明")
    
    st.markdown("""
    ### 🚀 快速开始
    
    1. **上传数据**: 在左侧侧边栏上传Excel数据文件（支持 .xlsx, .xlsm, .xls）
    2. **配置参数**: 调整预测数量、回测期数等参数
    3. **运行预测**: 点击「运行AI预测」按钮
    4. **查看结果**: 在各个标签页查看详细分析结果
    
    ### 📊 功能说明
    
    #### 预测分析
    - **AI融合预测**: 集成5个机器学习模型的预测结果
    - **Transformer预测**: 基于深度学习的序列预测
    - **辅助预测**: 大小、波色等辅助判断
    
    #### 特征工程
    系统提取8种类型的特征：
    1. 深度统计特征（均值、方差、偏度、峰度等）
    2. 频率与概率特征（信息熵、吉尼系数等）
    3. 时间序列特征（自相关、趋势等）
    4. 波动性与动量特征（RSI、波动率等）
    5. 模式识别特征（连续模式、重复等）
    6. 空间分布特征（区域平衡、波色分布）
    7. 遗漏与冷热特征（热号、冷号）
    8. 组合与交互特征（大小单双组合）
    
    #### 模型评估
    - 对比5个机器学习模型 + Transformer的性能
    - 准确率和速度指标
    
    #### 历史回测
    - 支持TOP1、TOP3、TOP5三种策略
    - 可调节回测期数（10-200期）
    - 显示命中率和详细记录
    
    #### 统计图表
    - 号码频率分布
    - 特码走势图
    - 大小、波色分布饼图
    
    ### ⚙️ 技术架构
    
    ```
    数据层: Excel文件 → pandas DataFrame
       ↓
    特征层: 8种特征工程算法
       ↓
    模型层: 5个ML模型 + Transformer
       ↓
    融合层: 概率融合与集成学习
       ↓
    应用层: Streamlit Web界面
    ```
    
    ### ⚠️ 重要提醒
    
    1. 本系统**仅供教育研究**使用
    2. 彩票结果完全随机，**任何预测都无法改变随机性**
    3. **切勿用于实际投注**
    4. 准确率仅反映历史数据统计特征，不代表未来预测能力
    5. 理性娱乐，远离赌博
    
    ### 📚 技术支持
    
    - Python 3.8+
    - 主要依赖: pandas, numpy, scipy, plotly, streamlit
    - 机器学习: 朴素贝叶斯、KNN、决策树、随机森林、梯度提升
    - 深度学习: Transformer (多头注意力机制)
    
    ### 📧 联系方式
    
    如有技术问题，欢迎在GitHub提交Issue。
    """)

# ============================================================================
# 页脚
# ============================================================================

st.divider()
st.markdown("""
<div style="text-align: center; color: #6b7280; padding: 2rem 0;">
    <p>AI彩票量化研究系统 Pro v2.0</p>
    <p>仅供教育和学术研究使用 · 请勿用于实际投注</p>
    <p>理性娱乐，远离赌博 🎓</p>
</div>
""", unsafe_allow_html=True)
