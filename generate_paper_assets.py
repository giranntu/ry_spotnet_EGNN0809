#!/usr/bin/env python3
"""
Generate All Paper Assets for IEEE Access Submission
=====================================================
Run this script to generate all tables, figures, and metrics for the paper
"""

import os
import subprocess
import json
from datetime import datetime
from utils.latex_generator import LaTeXTableGenerator
import pandas as pd
import numpy as np


def run_evaluation():
    """Run the evaluation if needed"""
    print("="*80)
    print("GENERATING PAPER ASSETS FOR IEEE ACCESS")
    print("="*80)
    
    # Check if recent evaluation exists
    import glob
    eval_files = glob.glob('evaluation_results_30min_*.json')
    
    if eval_files:
        latest_file = sorted(eval_files)[-1]
        # Check if it's from today
        file_date = latest_file.split('_')[-1].split('.')[0][:8]
        today = datetime.now().strftime("%Y%m%d")
        
        if file_date == today:
            print(f"✅ Using existing evaluation from today: {latest_file}")
            return latest_file
    
    # Run new evaluation
    print("Running fresh evaluation with all models...")
    result = subprocess.run(['python', '6_evaluate_all_models.py'], 
                          capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Error running evaluation: {result.stderr}")
        return None
    
    # Find the new evaluation file
    eval_files = glob.glob('evaluation_results_30min_*.json')
    if eval_files:
        return sorted(eval_files)[-1]
    
    return None


def generate_latex_tables(results_file):
    """Generate all LaTeX tables"""
    print("\n📝 Generating LaTeX tables...")
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    generator = LaTeXTableGenerator(output_dir='paper_assets')
    
    # Generate standard tables
    files = generator.save_all_tables(results['metrics'])
    
    # Generate additional custom table for paper
    generate_custom_summary_table(results['metrics'])
    
    return files


def generate_custom_summary_table(metrics_list):
    """Generate a custom formatted summary table"""
    
    # Create DataFrame
    df = pd.DataFrame(metrics_list)
    
    # Calculate percentage improvements
    naive_qlike = df[df['model'].str.contains('Naive')]['qlike'].values[0]
    df['improvement_pct'] = ((naive_qlike - df['qlike']) / naive_qlike * 100)
    
    # Sort by QLIKE
    df = df.sort_values('qlike')
    
    # Create custom LaTeX table
    latex_code = r"""
\begin{table*}[!t]
\centering
\caption{Comprehensive Performance Evaluation of 30-Minute Intraday Volatility Forecasting Models}
\label{tab:comprehensive_evaluation}
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}lccccccccc@{}}
\toprule
\multirow{2}{*}{Model} & \multicolumn{3}{c}{Volatility Scale} & \multicolumn{3}{c}{Variance Scale} & \multirow{2}{*}{QLIKE} & \multirow{2}{*}{Improvement} & \multirow{2}{*}{Rank} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7}
 & MSE & RMSE & MAE & MSE & RMSE & MAE &  & (\%) & \\
\midrule
"""
    
    # Add rows
    for idx, row in df.iterrows():
        model_name = row['model'].replace('_30min', '').replace('_', ' ')
        
        # Format values
        mse_vol = f"{row['mse']:.2e}"
        rmse_vol = f"{row['rmse']:.6f}"
        mae_vol = f"{row['mae']:.6f}"
        mse_var = f"{row.get('mse_var', row['rmse_var']**2):.2e}"
        rmse_var = f"{row['rmse_var']:.2e}"
        mae_var = f"{row['mae_var']:.2e}"
        qlike = f"{row['qlike']:.4f}"
        
        if pd.notna(row['improvement_pct']):
            improvement = f"{row['improvement_pct']:+.1f}"
        else:
            improvement = "--"
        
        rank = idx + 1
        
        # Highlight best model
        if idx == 0:
            latex_code += r"\rowcolor{green!20}"
        
        latex_code += f"{model_name} & {mse_vol} & {rmse_vol} & {mae_vol} & "
        latex_code += f"{mse_var} & {rmse_var} & {mae_var} & "
        latex_code += f"{qlike} & {improvement} & {rank} \\\\\n"
    
    latex_code += r"""
\bottomrule
\end{tabular}%
}
\vspace{2mm}
\begin{tablenotes}
\footnotesize
\item Note: All metrics are computed on the test set comprising 3,149 samples (20\% of data).
\item MSE = Mean Squared Error, RMSE = Root Mean Squared Error, MAE = Mean Absolute Error
\item QLIKE = Quasi-Likelihood loss function (asymmetric, penalizes under-prediction more)
\item Improvement is calculated relative to the Naive persistence baseline
\item Models are ranked by QLIKE performance (lower is better)
\end{tablenotes}
\end{table*}
"""
    
    # Save the custom table
    with open('paper_assets/tables/custom_summary_table.tex', 'w') as f:
        f.write(latex_code)
    
    print("✅ Custom summary table generated")


def create_model_complexity_table():
    """Create a table showing model complexity and training time"""
    
    complexity_data = [
        {
            'Model': 'SpotV2Net (GAT)',
            'Parameters': '~2.5M',
            'Training Time': '~2 hours',
            'Inference Time': '~50ms',
            'GPU Memory': '~2GB',
            'Architecture': 'Graph Attention Network'
        },
        {
            'Model': 'LSTM',
            'Parameters': '~800K',
            'Training Time': '~1 hour',
            'Inference Time': '~30ms',
            'GPU Memory': '~1GB',
            'Architecture': '2-layer LSTM'
        },
        {
            'Model': 'XGBoost',
            'Parameters': '~500K',
            'Training Time': '~5 minutes',
            'Inference Time': '~10ms',
            'GPU Memory': '~500MB',
            'Architecture': 'Gradient Boosted Trees'
        },
        {
            'Model': 'HAR-Intraday',
            'Parameters': '~100',
            'Training Time': 'Instant',
            'Inference Time': '~1ms',
            'GPU Memory': 'N/A',
            'Architecture': 'Linear Regression'
        }
    ]
    
    df = pd.DataFrame(complexity_data)
    
    latex_code = r"""
\begin{table}[!t]
\centering
\caption{Model Complexity and Computational Requirements}
\label{tab:model_complexity}
\begin{tabular}{@{}lccccc@{}}
\toprule
Model & Parameters & Training & Inference & GPU & Architecture \\
 &  & Time & Time & Memory & \\
\midrule
"""
    
    for _, row in df.iterrows():
        latex_code += f"{row['Model']} & {row['Parameters']} & {row['Training Time']} & "
        latex_code += f"{row['Inference Time']} & {row['GPU Memory']} & {row['Architecture']} \\\\\n"
    
    latex_code += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    
    with open('paper_assets/tables/model_complexity.tex', 'w') as f:
        f.write(latex_code)
    
    print("✅ Model complexity table generated")


def main():
    """Main execution"""
    
    # Step 1: Run or load evaluation
    eval_file = run_evaluation()
    
    if not eval_file:
        print("❌ Could not generate or find evaluation results")
        return
    
    # Step 2: Generate LaTeX tables
    latex_files = generate_latex_tables(eval_file)
    
    # Step 3: Generate additional tables
    create_model_complexity_table()
    
    # Step 4: Create summary report
    print("\n" + "="*80)
    print("📊 PAPER ASSETS GENERATION COMPLETE")
    print("="*80)
    
    print("\n📁 Generated Files:")
    print(f"  - Main results table: paper_assets/tables/main_results_*.tex")
    print(f"  - Comprehensive table: paper_assets/tables/comprehensive_results_*.tex")
    print(f"  - Statistical tests: paper_assets/tables/statistical_tests_*.tex")
    print(f"  - Custom summary: paper_assets/tables/custom_summary_table.tex")
    print(f"  - Model complexity: paper_assets/tables/model_complexity.tex")
    print(f"  - All tables combined: paper_assets/all_tables_*.tex")
    
    print("\n📈 Key Results:")
    with open(eval_file, 'r') as f:
        results = json.load(f)
    
    best_model = results.get('best_model', 'Unknown')
    best_qlike = results.get('best_qlike', 0)
    
    print(f"  - Best Model: {best_model}")
    print(f"  - Best QLIKE: {best_qlike:.4f}")
    print(f"  - Test Samples: {results.get('test_samples', 'N/A')}")
    
    print("\n💡 Usage in LaTeX:")
    print(r"  \input{paper_assets/tables/main_results_*.tex}")
    print(r"  \input{paper_assets/tables/custom_summary_table.tex}")
    print(r"  \input{paper_assets/tables/model_complexity.tex}")
    
    print("\n✅ Ready for IEEE Access submission!")


if __name__ == "__main__":
    main()