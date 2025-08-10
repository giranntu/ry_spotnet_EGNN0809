#!/usr/bin/env python3
"""
LaTeX Table Generator for IEEE Access Paper
============================================
Generates publication-ready tables with all metrics including MSE
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Dict, List, Optional
import json


class LaTeXTableGenerator:
    """Generate LaTeX tables for academic papers with IEEE Access formatting"""
    
    def __init__(self, output_dir: str = 'paper_assets'):
        """
        Initialize LaTeX generator
        
        Args:
            output_dir: Directory to save LaTeX files and assets
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'tables'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
        
        # IEEE Access table formatting
        self.ieee_preamble = r"""% IEEE Access Table Format
\documentclass[journal]{IEEEtran}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{amsmath}
\usepackage{array}
\usepackage{siunitx}
\usepackage{xcolor}
\usepackage{colortbl}

% Define colors for highlighting
\definecolor{bestcolor}{RGB}{200,255,200}
\definecolor{secondcolor}{RGB}{220,220,255}
"""
        
    def generate_main_results_table(self, metrics_list: List[Dict], 
                                   caption: str = None,
                                   label: str = "tab:main_results") -> str:
        """
        Generate main results table with all metrics
        
        Args:
            metrics_list: List of dictionaries containing model metrics
            caption: Table caption
            label: LaTeX label for referencing
            
        Returns:
            LaTeX table code
        """
        if caption is None:
            caption = "Performance comparison of volatility forecasting models on 30-minute intraday data"
        
        # Create DataFrame
        df = pd.DataFrame(metrics_list)
        
        # Ensure all metrics are present
        required_cols = ['model', 'mse', 'rmse', 'mae', 'qlike']
        for col in required_cols:
            if col not in df.columns:
                if col == 'mse' and 'rmse' in df.columns:
                    df['mse'] = df['rmse'] ** 2
        
        # Sort by QLIKE (primary metric)
        df = df.sort_values('qlike')
        
        # Calculate improvement over naive
        naive_qlike = df[df['model'].str.contains('Naive', case=False)]['qlike'].values
        if len(naive_qlike) > 0:
            df['improvement'] = ((naive_qlike[0] - df['qlike']) / naive_qlike[0] * 100)
        
        # Format the table
        latex_code = r"\begin{table}[!t]" + "\n"
        latex_code += r"\centering" + "\n"
        latex_code += r"\caption{" + caption + "}\n"
        latex_code += r"\label{" + label + "}\n"
        latex_code += r"\begin{tabular}{@{}lcccccc@{}}" + "\n"
        latex_code += r"\toprule" + "\n"
        latex_code += r"Model & MSE & RMSE & MAE & QLIKE & Improvement (\%) \\" + "\n"
        latex_code += r"\midrule" + "\n"
        
        # Add each model's results
        for idx, row in df.iterrows():
            model_name = row['model'].replace('_30min', '').replace('_', ' ')
            
            # Highlight best model
            if idx == 0:  # Best model (lowest QLIKE)
                latex_code += r"\rowcolor{bestcolor}"
            
            # Format numbers
            mse_str = f"{row.get('mse', row.get('rmse', 0)**2):.2e}"
            rmse_str = f"{row.get('rmse', 0):.6f}"
            mae_str = f"{row.get('mae', 0):.6f}"
            qlike_str = f"{row.get('qlike', 0):.4f}"
            
            if 'improvement' in row and not pd.isna(row['improvement']):
                imp_str = f"{row['improvement']:+.2f}"
            else:
                imp_str = "--"
            
            latex_code += f"{model_name} & {mse_str} & {rmse_str} & {mae_str} & {qlike_str} & {imp_str} \\\\\n"
        
        latex_code += r"\bottomrule" + "\n"
        latex_code += r"\end{tabular}" + "\n"
        latex_code += r"\end{table}" + "\n"
        
        return latex_code
    
    def generate_statistical_significance_table(self, metrics_list: List[Dict],
                                               caption: str = None,
                                               label: str = "tab:statistical_tests") -> str:
        """
        Generate table with statistical significance tests (Diebold-Mariano)
        """
        if caption is None:
            caption = "Statistical significance of forecast improvements (Diebold-Mariano test)"
        
        df = pd.DataFrame(metrics_list)
        models = df['model'].values
        n_models = len(models)
        
        # Create pairwise comparison matrix
        latex_code = r"\begin{table}[!t]" + "\n"
        latex_code += r"\centering" + "\n"
        latex_code += r"\caption{" + caption + "}\n"
        latex_code += r"\label{" + label + "}\n"
        latex_code += r"\begin{tabular}{l" + "c" * n_models + "}\n"
        latex_code += r"\toprule" + "\n"
        
        # Header
        header = "Model"
        for model in models:
            model_short = model.replace('_30min', '').replace('_', ' ')[:10]
            header += f" & {model_short}"
        latex_code += header + r" \\" + "\n"
        latex_code += r"\midrule" + "\n"
        
        # Add comparison results (actual DM test not yet implemented)
        for i, model1 in enumerate(models):
            model1_short = model1.replace('_30min', '').replace('_', ' ')
            row = model1_short
            for j, model2 in enumerate(models):
                if i == j:
                    row += " & --"
                else:
                    # DM test not yet implemented - showing as pending
                    row += " & TBD"  # To be calculated with actual DM test
            latex_code += row + r" \\" + "\n"
        
        latex_code += r"\bottomrule" + "\n"
        latex_code += r"\multicolumn{" + str(n_models + 1) + r"}{l}{\footnotesize Note: *** p<0.001, ** p<0.01, * p<0.05}" + "\n"
        latex_code += r"\end{tabular}" + "\n"
        latex_code += r"\end{table}" + "\n"
        
        return latex_code
    
    def generate_temporal_analysis_table(self, metrics_by_interval: pd.DataFrame,
                                        caption: str = None,
                                        label: str = "tab:temporal_analysis") -> str:
        """
        Generate table showing performance by time of day
        """
        if caption is None:
            caption = "Model performance by intraday interval"
        
        latex_code = r"\begin{table}[!t]" + "\n"
        latex_code += r"\centering" + "\n"
        latex_code += r"\caption{" + caption + "}\n"
        latex_code += r"\label{" + label + "}\n"
        latex_code += r"\begin{tabular}{@{}lccccc@{}}" + "\n"
        latex_code += r"\toprule" + "\n"
        latex_code += r"Time Interval & Time of Day & MSE & RMSE & MAE & QLIKE \\" + "\n"
        latex_code += r"\midrule" + "\n"
        
        # Market periods
        latex_code += r"\multicolumn{6}{l}{\textbf{Market Open}} \\" + "\n"
        # Add morning intervals
        
        latex_code += r"\midrule" + "\n"
        latex_code += r"\multicolumn{6}{l}{\textbf{Midday}} \\" + "\n"
        # Add midday intervals
        
        latex_code += r"\midrule" + "\n"
        latex_code += r"\multicolumn{6}{l}{\textbf{Market Close}} \\" + "\n"
        # Add closing intervals
        
        latex_code += r"\bottomrule" + "\n"
        latex_code += r"\end{tabular}" + "\n"
        latex_code += r"\end{table}" + "\n"
        
        return latex_code
    
    def generate_model_comparison_summary(self, metrics_list: List[Dict]) -> str:
        """
        Generate a comprehensive summary table for the paper
        """
        df = pd.DataFrame(metrics_list)
        
        # Calculate additional statistics
        summary_stats = []
        for _, row in df.iterrows():
            model_stats = {
                'Model': row['model'].replace('_30min', '').replace('_', ' '),
                'MSE': row.get('mse', row.get('rmse', 0)**2),
                'RMSE (Vol)': row.get('rmse', 0),
                'MAE (Vol)': row.get('mae', 0),
                'QLIKE': row.get('qlike', 0),
                'RMSE (Var)': row.get('rmse_var', 0),
                'MAE (Var)': row.get('mae_var', 0)
            }
            summary_stats.append(model_stats)
        
        summary_df = pd.DataFrame(summary_stats)
        
        # Create LaTeX code
        latex_code = r"\begin{table*}[!t]" + "\n"
        latex_code += r"\centering" + "\n"
        latex_code += r"\caption{Comprehensive performance metrics for 30-minute intraday volatility forecasting}" + "\n"
        latex_code += r"\label{tab:comprehensive_results}" + "\n"
        latex_code += r"\begin{tabular}{@{}lS[table-format=1.2e-1]S[table-format=1.6]S[table-format=1.6]S[table-format=1.4]S[table-format=1.2e-1]S[table-format=1.2e-1]@{}}" + "\n"
        latex_code += r"\toprule" + "\n"
        latex_code += r"{Model} & {MSE} & {RMSE (Vol)} & {MAE (Vol)} & {QLIKE} & {RMSE (Var)} & {MAE (Var)} \\" + "\n"
        latex_code += r"\midrule" + "\n"
        
        # Add data rows
        for idx, row in summary_df.iterrows():
            if idx == 0:  # Best model
                latex_code += r"\rowcolor{bestcolor}"
            
            latex_code += f"{row['Model']} & "
            latex_code += f"{row['MSE']:.2e} & "
            latex_code += f"{row['RMSE (Vol)']:.6f} & "
            latex_code += f"{row['MAE (Vol)']:.6f} & "
            latex_code += f"{row['QLIKE']:.4f} & "
            latex_code += f"{row['RMSE (Var)']:.2e} & "
            latex_code += f"{row['MAE (Var)']:.2e} \\\\\n"
        
        latex_code += r"\bottomrule" + "\n"
        latex_code += r"\end{tabular}" + "\n"
        latex_code += r"\end{table*}" + "\n"
        
        return latex_code
    
    def save_all_tables(self, metrics_list: List[Dict], 
                       timestamp: str = None) -> Dict[str, str]:
        """
        Save all tables to files
        
        Returns:
            Dictionary with paths to saved files
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        saved_files = {}
        
        # Main results table
        main_table = self.generate_main_results_table(metrics_list)
        main_path = os.path.join(self.output_dir, 'tables', f'main_results_{timestamp}.tex')
        with open(main_path, 'w') as f:
            f.write(self.ieee_preamble + "\n\\begin{document}\n" + main_table + "\n\\end{document}")
        saved_files['main_results'] = main_path
        
        # Comprehensive summary table
        summary_table = self.generate_model_comparison_summary(metrics_list)
        summary_path = os.path.join(self.output_dir, 'tables', f'comprehensive_results_{timestamp}.tex')
        with open(summary_path, 'w') as f:
            f.write(self.ieee_preamble + "\n\\begin{document}\n" + summary_table + "\n\\end{document}")
        saved_files['comprehensive_results'] = summary_path
        
        # Statistical significance table
        stat_table = self.generate_statistical_significance_table(metrics_list)
        stat_path = os.path.join(self.output_dir, 'tables', f'statistical_tests_{timestamp}.tex')
        with open(stat_path, 'w') as f:
            f.write(self.ieee_preamble + "\n\\begin{document}\n" + stat_table + "\n\\end{document}")
        saved_files['statistical_tests'] = stat_path
        
        # Create master file with all tables
        master_content = self.ieee_preamble + "\n\\begin{document}\n\n"
        master_content += "% Main Results Table\n" + main_table + "\n\n"
        master_content += "% Comprehensive Results Table\n" + summary_table + "\n\n"
        master_content += "% Statistical Significance Table\n" + stat_table + "\n\n"
        master_content += "\\end{document}"
        
        master_path = os.path.join(self.output_dir, f'all_tables_{timestamp}.tex')
        with open(master_path, 'w') as f:
            f.write(master_content)
        saved_files['all_tables'] = master_path
        
        # Save paths to JSON for reference
        json_path = os.path.join(self.output_dir, f'table_paths_{timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(saved_files, f, indent=2)
        
        print(f"✅ LaTeX tables saved to: {self.output_dir}")
        print(f"   - Main results: {os.path.basename(main_path)}")
        print(f"   - Comprehensive: {os.path.basename(summary_path)}")
        print(f"   - Statistical: {os.path.basename(stat_path)}")
        print(f"   - All tables: {os.path.basename(master_path)}")
        
        return saved_files


def generate_paper_assets(evaluation_results_file: str = None):
    """
    Generate all paper assets from evaluation results
    """
    # Load evaluation results
    if evaluation_results_file is None:
        # Find the latest evaluation results
        import glob
        result_files = glob.glob('evaluation_results_30min_*.json')
        if result_files:
            evaluation_results_file = sorted(result_files)[-1]
    
    if not os.path.exists(evaluation_results_file):
        print(f"Error: Evaluation results file not found: {evaluation_results_file}")
        return
    
    with open(evaluation_results_file, 'r') as f:
        results = json.load(f)
    
    # Generate LaTeX tables
    generator = LaTeXTableGenerator()
    saved_files = generator.save_all_tables(
        results['metrics'], 
        timestamp=results.get('timestamp', datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
    
    return saved_files


if __name__ == "__main__":
    # TEST ONLY - This dummy data is ONLY for testing the LaTeX table generation
    # NEVER use this for actual results
    test_metrics = [
        {'model': 'SpotV2Net_30min', 'mse': 9.37e-08, 'rmse': 0.000306, 'mae': 0.000138, 'qlike': 0.1543},
        {'model': 'XGBoost_30min', 'mse': 1.38e-07, 'rmse': 0.000371, 'mae': 0.000184, 'qlike': 0.2671},
        {'model': 'LSTM_30min', 'mse': 1.77e-07, 'rmse': 0.000421, 'mae': 0.000206, 'qlike': 0.3096},
        {'model': 'HAR_30min', 'mse': 1.26e-07, 'rmse': 0.000355, 'mae': 0.000169, 'qlike': 0.2456},
        {'model': 'Naive_30min', 'mse': 1.75e-07, 'rmse': 0.000418, 'mae': 0.000209, 'qlike': 0.3378},
    ]
    
    generator = LaTeXTableGenerator()
    generator.save_all_tables(test_metrics)