#!/usr/bin/env python3
"""
Simple parser and analyzer for RUL comparison logs.

Parses lines like:
 - GroundTruth RUL=2892.5 hours (173550.0 min) after cycle 0/66667.
 - Published RUL prediction for hydraulic_press at cycle 114: RUL=3676.89 min, Active Model=Base Model, Stage_0_Prob=0.926...

Outputs:
 - CSV of matched GT vs Prediction
 - JSON metrics summary
 - PNG scatter and time-series plots (if matplotlib available)
"""

import re
import os
import sys
import json
from collections import defaultdict

try:
    import pandas as pd
    import numpy as np
except Exception:
    print("This script requires pandas and numpy. Please install them in your environment.")
    raise

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False


def parse_file(path):
    gt = {}  # key: (machine, cycle) -> gt_min
    preds = {}  # key: (machine, cycle) -> dict

    gt_re = re.compile(r"\[(?P<node>[^\]]+)\].*GroundTruth RUL=(?P<gt_hours>[\d\.]+) hours \((?P<gt_min>[\d\.]+) min\) after cycle (?P<cycle>\d+)/\d+\.")
    # General prediction pattern; additional fields optional
    pred_re = re.compile(r"Published RUL prediction for (?P<machine>[^\s]+) at cycle (?P<cycle>\d+):\s*RUL=(?P<pred>[\d\.]+) min(?:,\s*Active Model=(?P<active>[^,]+))?(?:,\s*Stage_0_Prob=(?P<stage0>[\d\.eE+-]+))?(?:,\s*Stage_1_Prob=(?P<stage1>[\d\.eE+-]+))?")

    debug_lines = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            raw_line = line.rstrip('\n')
            line = line.strip()
            if not line:
                continue

            # Capture TEMP:DEBUG lines for later inspection
            if '[TEMP:DEBUG]' in raw_line:
                debug_lines.append(raw_line)

            m = gt_re.search(line)
            if m:
                node = m.group('node')
                # map node name to machine
                if 'process_pump' in node:
                    machine = 'process_pump'
                elif 'hydraulic_press' in node:
                    machine = 'hydraulic_press'
                else:
                    # fallback: try to find machine name in line
                    machine = 'unknown'
                cycle = int(m.group('cycle'))
                gt_min = float(m.group('gt_min'))
                gt[(machine, cycle)] = gt_min
                continue

            p = pred_re.search(line)
            if p:
                machine = p.group('machine')
                cycle = int(p.group('cycle'))
                pred = float(p.group('pred'))
                active = p.group('active') if p.group('active') else None
                stage0 = float(p.group('stage0')) if p.group('stage0') else None
                stage1 = float(p.group('stage1')) if p.group('stage1') else None
                preds[(machine, cycle)] = {
                    'pred_min': pred,
                    'active_model': active,
                    'stage0_prob': stage0,
                    'stage1_prob': stage1,
                }
                continue

    # return debug lines as well
    return gt, preds, debug_lines

    return gt, preds


def build_df(gt, preds):
    rows = []
    for (machine, cycle), predinfo in preds.items():
        gt_key = (machine, cycle)
        if gt_key in gt:
            rows.append({
                'machine': machine,
                'cycle': cycle,
                'gt_min': gt[gt_key],
                'pred_min': predinfo['pred_min'],
                'active_model': predinfo['active_model'],
                'stage0_prob': predinfo['stage0_prob'],
                'stage1_prob': predinfo['stage1_prob'],
            })
        else:
            # No exact matching GT for this cycle; optionally we could try nearest-neighbor later
            rows.append({
                'machine': machine,
                'cycle': cycle,
                'gt_min': None,
                'pred_min': predinfo['pred_min'],
                'active_model': predinfo['active_model'],
                'stage0_prob': predinfo['stage0_prob'],
                'stage1_prob': predinfo['stage1_prob'],
            })

    df = pd.DataFrame(rows)
    # bring GTs from gt dict that have no preds as separate rows? For now focus on matched and pred rows
    return df


def compute_metrics(df):
    metrics = {}
    # consider only rows that have gt
    df_valid = df[df['gt_min'].notna()].copy()
    if df_valid.shape[0] == 0:
        metrics['note'] = 'No matches between predictions and ground truth cycles found.'
        return metrics

    df_valid['abs_err_min'] = (df_valid['gt_min'] - df_valid['pred_min']).abs()
    metrics['count'] = int(df_valid.shape[0])
    metrics['mae_min'] = float(df_valid['abs_err_min'].mean())
    metrics['mae_percent_of_mean_gt'] = float(metrics['mae_min'] / df_valid['gt_min'].mean())
    metrics['within_10min_pct'] = float((df_valid['abs_err_min'] <= 10).mean())
    metrics['pred_over_gt_mean_ratio'] = float((df_valid['pred_min'] / df_valid['gt_min']).mean())

    # breakdowns
    def subset_and_metrics(cond, key):
        d = df_valid[cond]
        if d.shape[0] == 0:
            return None
        return {
            'count': int(d.shape[0]),
            'mae_min': float(d['abs_err_min'].mean()),
            'within_10min_pct': float((d['abs_err_min'] <= 10).mean())
        }

    metrics['mae_le_20min'] = subset_and_metrics(df_valid['gt_min'] <= 20, 'le_20')
    metrics['mae_le_30min'] = subset_and_metrics(df_valid['gt_min'] <= 30, 'le_30')
    metrics['mae_le_100min'] = subset_and_metrics(df_valid['gt_min'] <= 100, 'le_100')

    return metrics


def save_outputs(df, metrics, outdir):
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(os.path.join(outdir, 'gt_pred_matches.csv'), index=False)
    with open(os.path.join(outdir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    if MATPLOTLIB_AVAILABLE and df.shape[0] > 0:
        try:
            dfv = df[df['gt_min'].notna()]
            # scatter plot GT vs Pred
            plt.figure(figsize=(6,6))
            plt.scatter(dfv['gt_min'], dfv['pred_min'], alpha=0.7)
            mn = min(dfv['gt_min'].min(), dfv['pred_min'].min())
            mx = max(dfv['gt_min'].max(), dfv['pred_min'].max())
            plt.plot([mn, mx], [mn, mx], color='k', linestyle='--')
            plt.xlabel('GT RUL (min)')
            plt.ylabel('Pred RUL (min)')
            plt.title('Predicted vs GroundTruth RUL')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, 'gt_vs_pred_scatter.png'))
            plt.close()

            # Time series for hydraulic_press (if present)
            for machine in df['machine'].unique():
                dmf = df[(df['machine'] == machine) & df['gt_min'].notna()].sort_values('cycle')
                if dmf.shape[0] > 0:
                    plt.figure(figsize=(10,4))
                    plt.plot(dmf['cycle'], dmf['gt_min'], label='GT')
                    plt.plot(dmf['cycle'], dmf['pred_min'], label='Pred')
                    plt.xlabel('Cycle')
                    plt.ylabel('RUL (min)')
                    plt.title(f'RUL Timeseries - {machine}')
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(os.path.join(outdir, f'{machine}_timeseries.png'))
                    plt.close()
        except Exception as e:
            print('Plot generation failed:', e)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Parse and analyze RUL logs')
    parser.add_argument('--input', '-i', default='rul_comparison.txt', help='Input log file (default: rul_comparison.txt)')
    parser.add_argument('--outdir', '-o', default='logs_analysis_output', help='Output directory')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f'Input file not found: {args.input}')
        sys.exit(1)

    gt, preds, debug_lines = parse_file(args.input)

    print(f'Parsed {len(gt)} ground-truth entries and {len(preds)} prediction entries from {args.input}')

    df = build_df(gt, preds)

    metrics = compute_metrics(df)

    save_outputs(df, metrics, args.outdir)

    # Save debug lines
    dbg_path = os.path.join(args.outdir, 'debug_temp_lines.txt')
    with open(dbg_path, 'w', encoding='utf-8') as f:
        for l in debug_lines:
            f.write(l + '\n')

    print('\n=== Metrics Summary ===')
    print(json.dumps(metrics, indent=2))
    print(f'Outputs saved to: {os.path.abspath(args.outdir)}')
    print(f'TEMP debug lines saved to: {dbg_path} (count={len(debug_lines)})')
