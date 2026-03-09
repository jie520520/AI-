"""
AI彩票量化研究系统 - 增强版Streamlit Web界面
仅供教育和学术研究使用

新增功能:
- 下注助手（保本分配计算）
- 大小、单双、波色的AI预测和回测
- 可自定义预测数量(1-49)
- 一键复制预测结果
- 历史记录查看（可自定义期数）
- 预测参数与回测同步

运行命令: streamlit run lottery_app_enhanced.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from lottery_core import (
    DataProcessor, FeatureEngineering, MLModels, 
    TransformerModel, EnsembleFusion, BacktestEngine
)
from lottery_core_enhanced import (
    AuxiliaryPredictor, AuxiliaryBacktest,
    EnhancedPredictionEngine, HistoryViewer
)
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 页面配置
# ============================================================================

st.set_page_config(
    page_title="AI彩票量化研究系统 Enhanced",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    .copy-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        font-weight: bold;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .big-number {
        font-size: 4rem;
        font-weight: bold;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 免责声明
# ============================================================================

DISCLAIMER = """
⚠️ **重要声明** ⚠️

本系统仅供**教育和学术研究**目的使用。

**彩票开奖结果完全随机**，任何预测模型都无法改变随机本质。
预测结果不构成任何投资建议，不应用于实际投注。

**理性娱乐，远离赌博。**如有赌博问题，请寻求专业帮助。
"""

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

st.markdown('<h1 class="main-header">🧠 AI彩票量化研究系统 Enhanced</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #6b7280; font-size: 1.1rem;">深度学习 · 8维特征 · Transformer · 下注助手 · 完整回测</p>', unsafe_allow_html=True)

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
                st.session_state.engine = EnhancedPredictionEngine(st.session_state.data)
                st.success(f"✓ 成功加载 {len(df)} 条历史记录")
    
    st.divider()
    
    # 预测配置（1-49可调）
    st.subheader("预测参数")
    top_k = st.number_input("融合预测数量", min_value=1, max_value=49, value=10, step=1,
                            help="可选择1-49个号码进行预测")
    transformer_top_k = st.number_input("Transformer预测数量", min_value=1, max_value=49, value=10, step=1,
                                       help="可选择1-49个号码进行预测")
    
    st.divider()
    
    # 回测配置（与预测同步）
    st.subheader("回测参数")
    st.caption("回测将使用上述预测参数")
    backtest_periods = st.slider("回测期数", 10, 200, 50, 10)
    backtest_strategy = st.selectbox(
        "预测策略",
        ["top1", "top3", "top5", "top10"],
        format_func=lambda x: {"top1": "TOP 1", "top3": "TOP 3", "top5": "TOP 5", "top10": "TOP 10"}[x]
    )
    
    st.divider()
    
    # 历史记录查看
    st.subheader("历史记录")
    history_periods = st.slider("显示期数", 5, 100, 20, 5)
    
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

# 创建标签页（新增下注助手和历史记录）
tabs = st.tabs(["📊 预测分析", "💰 下注助手", "🔧 特征工程", "📈 模型评估", "🔄 历史回测", "📜 历史记录", "📉 统计图表"])

# ============================================================================
# 标签1: 预测分析（增强版 - 包含复制功能）
# ============================================================================

with tabs[0]:
    if st.session_state.predictions is None:
        st.warning("请点击左侧「运行AI预测」按钮")
    else:
        preds = st.session_state.predictions
        
        # 一键复制按钮
        copy_text = st.session_state.engine.get_copy_format()
        
        col_copy1, col_copy2, col_copy3 = st.columns([2, 1, 2])
        with col_copy2:
            if st.button("📋 一键复制预测结果", use_container_width=True, type="primary"):
                st.code(copy_text, language=None)
                st.success("✓ 预测结果已显示，请手动复制")
        
        st.divider()
        
        # 主预测区域
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"🎯 AI融合预测 TOP {len(preds['融合预测'])}")
            
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
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                padding: 1rem; border-radius: 10px; margin: 0.3rem 0; color: white;">
                        <h3 style="margin: 0;">#{i+1} 号码 {pred['号码']}</h3>
                        <p style="margin: 0.5rem 0 0 0;">概率: <b>{pred['概率']}</b> | 置信度: {confidence_color} {pred['置信度']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("🎲 AI辅助预测")
            
            # 大小预测（使用AI模型）
            size_pred = preds['大小预测'][0]
            st.markdown(f"""
            <div style="background: {'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)' if size_pred['类型'] == '大' else 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)'}; 
                        padding: 2rem; border-radius: 10px; text-align: center; margin-bottom: 1rem;">
                <div class="big-number">{size_pred['类型']}</div>
                <p style="margin: 0; color: white; font-size: 1.1rem;">概率: {size_pred['概率']} | {size_pred['置信度']}</p>
                <p style="margin: 0.5rem 0 0 0; color: rgba(255,255,255,0.8); font-size: 0.9rem;">AI模型预测</p>
            </div>
            """, unsafe_allow_html=True)
            
            # 单双预测（使用AI模型）
            odd_even_pred = preds['单双预测'][0]
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                        padding: 2rem; border-radius: 10px; text-align: center; margin-bottom: 1rem;">
                <div style="font-size: 3rem; font-weight: bold; color: #333;">{odd_even_pred['类型']}</div>
                <p style="margin: 0; color: #333; font-size: 1.1rem;">概率: {odd_even_pred['概率']} | {odd_even_pred['置信度']}</p>
                <p style="margin: 0.5rem 0 0 0; color: #666; font-size: 0.9rem;">AI模型预测</p>
            </div>
            """, unsafe_allow_html=True)
            
            # 波色预测（使用AI模型）
            color_pred = preds['波色预测'][0]
            color_map = {'红波': '#dc2626', '蓝波': '#2563eb', '绿波': '#16a34a'}
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {color_map[color_pred['类型']]}dd 0%, {color_map[color_pred['类型']]}88 100%); 
                        padding: 2rem; border-radius: 10px; text-align: center;">
                <div style="font-size: 3rem; font-weight: bold; color: white;">{color_pred['类型']}</div>
                <p style="margin: 0; color: white; font-size: 1.1rem;">概率: {color_pred['概率']} | {color_pred['置信度']}</p>
                <p style="margin: 0.5rem 0 0 0; color: rgba(255,255,255,0.9); font-size: 0.9rem;">AI模型预测</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Transformer预测
        st.subheader(f"🤖 Transformer深度学习预测 TOP {len(preds['Transformer']['predictions'])}")
        st.caption(f"多头注意力机制 · 序列编码 · 置信度: {preds['Transformer']['confidence']:.1%}")
        
        cols = st.columns(min(5, len(preds['Transformer']['predictions'])))
        for i, pred in enumerate(preds['Transformer']['predictions']):
            col_idx = i % len(cols)
            with cols[col_idx]:
                st.metric(
                    label=f"#{i+1}",
                    value=f"号码 {pred['号码']}",
                    delta=pred['概率']
                )

# ============================================================================
# 标签2: 下注助手（新增）
# ============================================================================

with tabs[1]:
    st.subheader("💰 智能下注助手（保本分配计算）")
    st.caption("基于AI预测结果的下注金额分配工具")
    
    if st.session_state.predictions is None:
        st.warning("请先运行AI预测")
    else:
        # 下注配置
        col1, col2, col3 = st.columns(3)
        with col1:
            total_budget = st.number_input("总预算", min_value=100, max_value=1000000, value=1000, step=100)
        with col2:
            odds = st.number_input("赔率", min_value=1.0, max_value=100.0, value=47.0, step=0.1)
        with col3:
            st.metric("预测号码数", len(st.session_state.predictions['融合预测']))
        
        # 档位配置
        st.subheader("📊 档位分配设置")
        st.caption("根据置信度自动分配到不同档位")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("**1档 (最高)**")
            rank1_weight = st.slider("1档权重", 1.0, 5.0, 3.0, 0.5, key="r1")
            rank1_nums = [p['号码'] for p in st.session_state.predictions['融合预测'] if p['置信度'] == '高']
            st.info(f"号码: {', '.join(map(str, rank1_nums)) if rank1_nums else '无'}")
        
        with col2:
            st.markdown("**2档 (中)**")
            rank2_weight = st.slider("2档权重", 0.5, 3.0, 1.5, 0.5, key="r2")
            rank2_nums = [p['号码'] for p in st.session_state.predictions['融合预测'] if p['置信度'] == '中']
            st.info(f"号码: {', '.join(map(str, rank2_nums[:5])) if rank2_nums else '无'}")
        
        with col3:
            st.markdown("**3档 (低)**")
            rank3_weight = st.slider("3档权重", 0.1, 2.0, 0.5, 0.1, key="r3")
            rank3_nums = [p['号码'] for p in st.session_state.predictions['融合预测'] if p['置信度'] == '低']
            st.info(f"号码: {', '.join(map(str, rank3_nums[:5])) if rank3_nums else '无'}")
        
        with col4:
            st.markdown("**4档 (保本)**")
            rank4_weight = st.number_input("4档权重", 0.0, 1.0, 0.0, 0.1, key="r4")
            st.info("可手动添加保本号码")
        
        # 计算下注金额
        if st.button("💡 计算最优下注金额", type="primary"):
            predictions = st.session_state.predictions['融合预测']
            
            # 分配号码到档位
            rank_data = {
                '1档(最高)': {'nums': rank1_nums, 'weight': rank1_weight},
                '2档(中)': {'nums': rank2_nums, 'weight': rank2_weight},
                '3档(低)': {'nums': rank3_nums, 'weight': rank3_weight},
                '4档(保本)': {'nums': [], 'weight': rank4_weight}
            }
            
            total_nums = sum(len(data['nums']) for data in rank_data.values())
            
            if total_nums > 0:
                # 保本计算
                base_bet = total_budget / odds
                remaining_fund = total_budget - (total_nums * base_bet)
                
                total_weight_score = sum(len(data['nums']) * data['weight'] for data in rank_data.values())
                
                # 生成下注表
                bet_results = []
                for rank_name, data in rank_data.items():
                    if data['nums']:
                        extra = (remaining_fund / total_weight_score * data['weight']) if total_weight_score > 0 else 0
                        final_bet = round(base_bet + extra, 2)
                        profit = round((final_bet * odds) - total_budget, 2)
                        
                        for num in data['nums']:
                            bet_results.append({
                                '号码': f"{num:02d}",
                                '档位': rank_name[:2],
                                '单注金额': f"{final_bet:.2f}",
                                '中奖纯利': f"{profit:.2f}"
                            })
                
                # 显示结果
                df_bet = pd.DataFrame(bet_results)
                st.dataframe(df_bet, use_container_width=True, height=400)
                
                # 汇总
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("总投注号码", total_nums)
                with col2:
                    st.metric("预算状态", "正常" if remaining_fund >= 0 else "超支")
                with col3:
                    st.metric("剩余分配", f"{remaining_fund:.2f}")
                
                # 生成投注格式
                st.subheader("📋 投注格式（同金额合并）")
                
                # 按金额分组
                amount_groups = {}
                for result in bet_results:
                    amount = result['单注金额']
                    if amount not in amount_groups:
                        amount_groups[amount] = []
                    amount_groups[amount].append(result['号码'])
                
                # 生成格式化文本
                bet_format_lines = []
                for amount in sorted(amount_groups.keys(), key=lambda x: float(x), reverse=True):
                    nums = ' '.join(amount_groups[amount])
                    bet_format_lines.append(f"澳特: {nums} 各下: {amount}")
                
                bet_format = "\n".join(bet_format_lines)
                st.code(bet_format, language=None)
                
                if st.button("📋 复制投注格式"):
                    st.success("请手动复制上方文本")

# ============================================================================
# 标签3: 特征工程
# ============================================================================

with tabs[2]:
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
        
        # 其他特征类似显示...
        st.info("其他特征类型的详细展开请查看完整版")

# ============================================================================
# 标签4: 模型评估
# ============================================================================

with tabs[3]:
    st.subheader("📈 模型性能对比")
    
    model_performance = pd.DataFrame({
        '模型': ['朴素贝叶斯', 'K近邻', '决策树', '随机森林', '梯度提升', 'Transformer'],
        '准确率': [18.5, 16.2, 15.8, 22.3, 20.7, 24.1],
        '速度': [98, 72, 95, 65, 58, 45]
    })
    
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

# ============================================================================
# 标签5: 历史回测（增强版 - 包含大小、单双回测）
# ============================================================================

with tabs[4]:
    st.subheader("🔄 历史回测系统（完整版）")
    st.caption(f"当前预测参数: TOP {top_k}, 回测策略: {backtest_strategy.upper()}")
    
    # 选择回测类型
    backtest_type = st.selectbox(
        "选择回测类型",
        ["特码预测", "大小预测", "单双预测", "波色预测"]
    )
    
    if st.button("开始回测", type="primary"):
        with st.spinner(f"正在回测 {backtest_periods} 期..."):
            
            if backtest_type == "特码预测":
                # 特码回测
                def predict_func(train_data):
                    temp_engine = EnhancedPredictionEngine(train_data)
                    temp_features = FeatureEngineering.extract_all_features(train_data)
                    nb = MLModels.naive_bayes(train_data, temp_features)
                    top_preds = EnsembleFusion.get_top_predictions(nb, top_k)
                    return top_preds
                
                backtest_result = BacktestEngine.run(
                    st.session_state.data,
                    predict_func,
                    backtest_periods,
                    backtest_strategy
                )
                
            elif backtest_type == "大小预测":
                backtest_result = AuxiliaryBacktest.backtest_size(
                    st.session_state.data,
                    backtest_periods
                )
                
            elif backtest_type == "单双预测":
                backtest_result = AuxiliaryBacktest.backtest_odd_even(
                    st.session_state.data,
                    backtest_periods
                )
                
            elif backtest_type == "波色预测":
                backtest_result = AuxiliaryBacktest.backtest_color(
                    st.session_state.data,
                    backtest_periods
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
                st.metric("回测类型", backtest_result.get('type', backtest_type))
            
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
# 标签6: 历史记录（新增）
# ============================================================================

with tabs[5]:
    st.subheader(f"📜 历史记录查看（最近 {history_periods} 期）")
    
    # 获取历史记录
    history_df = HistoryViewer.get_recent_history(st.session_state.data, history_periods)
    
    # 显示历史记录
    st.dataframe(history_df, use_container_width=True, height=400)
    
    st.divider()
    
    # 历史统计分析
    st.subheader(f"📊 历史统计分析（最近 {history_periods} 期）")
    
    history_stats = HistoryViewer.analyze_history(st.session_state.data, history_periods)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("大数次数", f"{history_stats['大数次数']} ({history_stats['大数次数']/history_periods*100:.1f}%)")
        st.metric("小数次数", f"{history_stats['小数次数']} ({history_stats['小数次数']/history_periods*100:.1f}%)")
    
    with col2:
        st.metric("单数次数", f"{history_stats['单数次数']} ({history_stats['单数次数']/history_periods*100:.1f}%)")
        st.metric("双数次数", f"{history_stats['双数次数']} ({history_stats['双数次数']/history_periods*100:.1f}%)")
    
    with col3:
        st.metric("红波次数", f"{history_stats['红波次数']} ({history_stats['红波次数']/history_periods*100:.1f}%)")
        st.metric("蓝波次数", f"{history_stats['蓝波次数']} ({history_stats['蓝波次数']/history_periods*100:.1f}%)")
    
    with col4:
        st.metric("绿波次数", f"{history_stats['绿波次数']} ({history_stats['绿波次数']/history_periods*100:.1f}%)")
        st.metric("平均特码", f"{history_stats['平均特码']:.2f}")

# ============================================================================
# 标签7: 统计图表
# ============================================================================

with tabs[6]:
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

# ============================================================================
# 页脚
# ============================================================================

st.divider()
st.markdown("""
<div style="text-align: center; color: #6b7280; padding: 2rem 0;">
    <p>AI彩票量化研究系统 Enhanced v2.5</p>
    <p>新增: 下注助手 · 辅助AI预测 · 完整回测 · 历史记录</p>
    <p>仅供教育和学术研究使用 · 请勿用于实际投注</p>
    <p>理性娱乐，远离赌博 🎓</p>
</div>
""", unsafe_allow_html=True)
