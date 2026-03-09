#!/usr/bin/env python3
"""
AI彩票量化研究系统 - 命令行演示脚本
快速测试核心功能

使用方法:
    python demo.py <excel_file_path>

示例:
    python demo.py 澳门六合彩数据导入器.xlsm
"""

import sys
import pandas as pd
from lottery_core import (
    DataProcessor, FeatureEngineering, MLModels,
    TransformerModel, EnsembleFusion, BacktestEngine, PredictionEngine
)

def print_separator(title=""):
    """打印分隔线"""
    if title:
        print("\n" + "="*70)
        print(f"  {title}")
        print("="*70)
    else:
        print("-"*70)

def main():
    """主函数"""
    
    # 显示免责声明
    print("\n" + "="*70)
    print("  ⚠️  AI彩票量化研究系统 - 命令行演示  ⚠️")
    print("="*70)
    print("\n⚠️ 重要声明:")
    print("本系统仅供教育和学术研究使用。")
    print("彩票结果完全随机，任何预测都无法改变随机性。")
    print("切勿用于实际投注！理性娱乐，远离赌博。")
    print("="*70)
    
    input("\n按Enter键继续...")
    
    # 检查参数
    if len(sys.argv) < 2:
        print("\n❌ 错误: 请提供Excel文件路径")
        print(f"使用方法: python {sys.argv[0]} <excel_file_path>")
        print(f"示例: python {sys.argv[0]} 澳门六合彩数据导入器.xlsm")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    try:
        # 1. 加载数据
        print_separator("步骤1: 加载数据")
        print(f"正在读取: {file_path}")
        
        df = DataProcessor.load_excel(file_path)
        data = DataProcessor.parse_data(df)
        
        print(f"✓ 成功加载 {len(data)} 条历史记录")
        print(f"✓ 数据范围: {data['期号'].iloc[0]} - {data['期号'].iloc[-1]}")
        
        # 2. 显示统计信息
        print_separator("步骤2: 数据统计")
        stats = DataProcessor.get_statistics(data)
        
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key:12s}: {value:8.2f}")
            else:
                print(f"  {key:12s}: {value:8}")
        
        # 3. 提取特征
        print_separator("步骤3: 特征工程（8维）")
        print("正在提取特征...")
        
        features = FeatureEngineering.extract_all_features(data)
        
        print("\n特征统计:")
        for feature_type, feature_data in features.items():
            if isinstance(feature_data, dict):
                count = len(feature_data)
                print(f"  ✓ {feature_type:12s}: {count} 个特征")
        
        # 显示部分特征示例
        print("\n统计特征示例:")
        for key, value in list(features['统计特征'].items())[:5]:
            print(f"  {key:20s}: {value:.4f}")
        
        # 4. 运行机器学习模型
        print_separator("步骤4: 运行5个机器学习模型")
        
        print("  [1/5] 朴素贝叶斯...")
        nb = MLModels.naive_bayes(data, features)
        
        print("  [2/5] K近邻...")
        knn = MLModels.weighted_knn(data, features)
        
        print("  [3/5] 决策树...")
        dt = MLModels.decision_tree(data, features)
        
        print("  [4/5] 随机森林...")
        rf = MLModels.random_forest(data, features)
        
        print("  [5/5] 梯度提升...")
        gb = MLModels.gradient_boosting(data, features)
        
        print("\n✓ 所有模型运行完成")
        
        # 5. Transformer模型
        print_separator("步骤5: Transformer深度学习")
        print("正在运行Transformer模型...")
        
        transformer = TransformerModel()
        transformer_result = transformer.predict(data, top_k=10)
        
        print(f"✓ Transformer完成 (置信度: {transformer_result['confidence']:.2%})")
        
        # 6. 概率融合
        print_separator("步骤6: 概率融合与集成")
        print("正在融合5个模型的预测结果...")
        
        fused_prob = EnsembleFusion.fuse_predictions([nb, knn, dt, rf, gb])
        fusion_predictions = EnsembleFusion.get_top_predictions(fused_prob, 10)
        
        print("✓ 融合完成")
        
        # 7. 显示预测结果
        print_separator("预测结果")
        
        print("\n🎯 AI融合预测 TOP 10:")
        print(f"{'排名':^6} {'号码':^6} {'概率':^10} {'置信度':^8}")
        print("-"*36)
        
        for i, pred in enumerate(fusion_predictions, 1):
            confidence_symbol = {
                '高': '🟢',
                '中': '🟡',
                '低': '⚪'
            }[pred['置信度']]
            
            print(f"{i:^6} {pred['号码']:^6} {pred['概率']:^10} {confidence_symbol} {pred['置信度']:^6}")
        
        print("\n🤖 Transformer深度学习预测 TOP 10:")
        print(f"{'排名':^6} {'号码':^6} {'概率':^10} {'置信度':^8}")
        print("-"*36)
        
        for i, pred in enumerate(transformer_result['predictions'], 1):
            confidence_symbol = {
                '高': '🟢',
                '中': '🟡',
                '低': '⚪'
            }[pred['置信度']]
            
            print(f"{i:^6} {pred['号码']:^6} {pred['概率']:^10} {confidence_symbol} {pred['置信度']:^6}")
        
        # 8. 辅助预测
        print("\n🎲 辅助预测:")
        recent_30 = data.iloc[-30:]
        big_count = len(recent_30[recent_30['大小'] == '大'])
        big_small_pred = '大' if big_count > 15 else '小'
        
        color_counts = recent_30['波色'].value_counts()
        color_pred = color_counts.idxmax()
        
        print(f"  大小预测: {big_small_pred}")
        print(f"  波色预测: {color_pred}")
        
        # 9. 历史回测（可选）
        print_separator("步骤7: 历史回测（可选）")
        
        run_backtest = input("\n是否运行历史回测？(y/n): ").lower().strip()
        
        if run_backtest == 'y':
            print("\n正在运行回测...")
            
            # 定义预测函数
            def predict_func(train_data):
                temp_features = FeatureEngineering.extract_all_features(train_data)
                temp_nb = MLModels.naive_bayes(train_data, temp_features)
                return EnsembleFusion.get_top_predictions(temp_nb, 10)
            
            # 运行回测
            backtest_result = BacktestEngine.run(
                data,
                predict_func,
                test_periods=50,
                strategy='top1'
            )
            
            print("\n回测结果:")
            print(f"  总测试数: {backtest_result['total_tests']}")
            print(f"  命中次数: {backtest_result['hit_count']}")
            print(f"  准确率  : {backtest_result['accuracy']}")
            print(f"  策略    : {backtest_result['strategy'].upper()}")
            
            # 显示最近10条记录
            print("\n最近10条回测记录:")
            print(f"{'期号':^10} {'预测':^6} {'实际':^6} {'结果':^6}")
            print("-"*32)
            
            results_df = backtest_result['results']
            for _, row in results_df.tail(10).iterrows():
                result_symbol = '✓' if row['命中'] else '✗'
                print(f"{row['期号']:^10} {row['预测']:^6} {row['实际']:^6} {result_symbol:^6}")
        
        # 10. 完成
        print_separator("演示完成")
        print("\n✓ 所有功能演示完成！")
        print("\n提醒:")
        print("  • 这些预测仅供学习算法使用")
        print("  • 彩票结果完全随机，不可预测")
        print("  • 切勿用于实际投注")
        print("  • 理性娱乐，远离赌博")
        
        print("\n💡 要使用完整的Web界面，请运行:")
        print("    streamlit run lottery_app.py")
        
        print("\n" + "="*70)
        
    except FileNotFoundError:
        print(f"\n❌ 错误: 找不到文件 '{file_path}'")
        print("请检查文件路径是否正确")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ 发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
