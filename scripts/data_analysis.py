"""
Data Analysis for Advanced Image Sensor Interface

This script provides tools for analyzing simulation results and real-world test data
from the Advanced Image Sensor Interface project.

Usage:
    python data_analysis.py [options] <input_files>

Options:
    --plot                    Generate plots for the analyzed data
    --output OUTPUT           Output file for analysis results (default: analysis_results.json)
    --compare                 Compare multiple input files

Example:
-------
    python data_analysis.py --plot --output analysis.json simulation_results.json test_data.json

"""

import argparse
import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def load_data(file_path: str) -> dict[str, Any]:
    """Load data from a JSON file."""
    with open(file_path) as f:
        return json.load(f)

def analyze_data(data: dict[str, Any]) -> dict[str, Any]:
    """Perform statistical analysis on the input data."""
    analysis = {}
    for key, value in data.items():
        if isinstance(value, (int, float)):
            analysis[key] = {
                'value': value,
                'unit': get_unit(key)
            }
        elif isinstance(value, list):
            analysis[key] = {
                'mean': np.mean(value),
                'std': np.std(value),
                'min': np.min(value),
                'max': np.max(value),
                'unit': get_unit(key)
            }
    return analysis

def get_unit(metric: str) -> str:
    """Return the appropriate unit for a given metric."""
    units = {
        'snr': 'dB',
        'dynamic_range': 'dB',
        'color_accuracy': 'Delta E',
        'power_consumption': 'W',
        'processing_time': 's'
    }
    return units.get(metric, '')

def plot_data(data: dict[str, Any], output_prefix: str):
    """Generate plots for the analyzed data."""
    for key, value in data.items():
        if isinstance(value, dict) and 'mean' in value:
            plt.figure(figsize=(10, 6))
            plt.bar(['Mean', 'Min', 'Max'], [value['mean'], value['min'], value['max']])
            plt.title(f"{key.replace('_', ' ').title()} Analysis")
            plt.ylabel(f"{key.replace('_', ' ').title()} ({value['unit']})")
            plt.savefig(f"{output_prefix}_{key}_analysis.png")
            plt.close()

def compare_data(data_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare multiple datasets and compute relative improvements."""
    comparison = {}
    baseline = data_list[0]
    for i, data in enumerate(data_list[1:], 1):
        comparison[f'comparison_{i}'] = {}
        for key in baseline.keys():
            if key in data:
                baseline_value = baseline[key]['mean'] if isinstance(baseline[key], dict) else baseline[key]
                current_value = data[key]['mean'] if isinstance(data[key], dict) else data[key]
                improvement = (current_value - baseline_value) / baseline_value * 100
                comparison[f'comparison_{i}'][key] = {
                    'baseline': baseline_value,
                    'current': current_value,
                    'improvement': improvement,
                    'unit': get_unit(key)
                }
    return comparison

def main():
    parser = argparse.ArgumentParser(description="Data Analysis for Advanced Image Sensor Interface")
    parser.add_argument('input_files', nargs='+', help='Input JSON files to analyze')
    parser.add_argument('--plot', action='store_true', help='Generate plots for the analyzed data')
    parser.add_argument('--output', default='analysis_results.json', help='Output file for analysis results')
    parser.add_argument('--compare', action='store_true', help='Compare multiple input files')

    args = parser.parse_args()

    results = []
    for file_path in args.input_files:
        data = load_data(file_path)
        analysis = analyze_data(data)
        results.append(analysis)

        print(f"Analysis for {file_path}:")
        for key, value in analysis.items():
            if isinstance(value, dict) and 'mean' in value:
                print(f"{key}: Mean = {value['mean']:.2f} ± {value['std']:.2f} {value['unit']}")
            else:
                print(f"{key}: {value['value']} {value['unit']}")
        print()

    if args.plot:
        for i, analysis in enumerate(results):
            plot_data(analysis, f"plot_{i}")
        print("Plots generated.")

    if args.compare and len(results) > 1:
        comparison = compare_data(results)
        print("Comparison Results:")
        for comp_key, comp_data in comparison.items():
            print(f"\n{comp_key}:")
            for key, value in comp_data.items():
                print(f"{key}: Improvement = {value['improvement']:.2f}% ({value['baseline']:.2f} -> {value['current']:.2f} {value['unit']})")

        results.append({"comparison": comparison})

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Analysis results saved to {args.output}")

if __name__ == "__main__":
    main()
