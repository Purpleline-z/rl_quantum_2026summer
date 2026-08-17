#!/usr/bin/env python3
"""
Summarize experiment logs and utility caches into two tables:
- accuracy_table: strategy vs experiment -> final test accuracy
- utility_table: strategy vs experiment -> average delta_accuracy over ALL selected pairs (all rounds)
"""

import os
import glob
import json
import ast
import pandas as pd
from collections import defaultdict

def read_log(csv_path):
    """Read experiment log, return dict strategy -> list of rows"""
    df = pd.read_csv(csv_path)
    # Convert string lists to actual lists
    for col in ['selected_pair_ids', 'selected_clusters']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else [])
    groups = {}
    for strategy, group in df.groupby('strategy'):
        rows = group.to_dict('records')
        groups[strategy] = rows
    return groups

def load_utility_cache(json_path):
    """Load utility cache, return dict candidate_pair_id -> delta_accuracy"""
    if not os.path.exists(json_path):
        return {}
    with open(json_path, 'r') as f:
        data = json.load(f)
    mapping = {}
    for entry in data.values():
        cid = entry.get('candidate_pair_id')
        delta = entry.get('delta_accuracy')
        if cid and delta is not None:
            mapping[cid] = delta
    return mapping

def main():
    # Find all CSV log files
    csv_files = glob.glob('*experiment_log.csv')
    if not csv_files:
        print("No experiment_log.csv files found.")
        return

    accuracy_data = defaultdict(dict)
    utility_data = defaultdict(dict)

    for csv_path in csv_files:
        # Locate corresponding utility_cache.json by common prefix
        prefix = csv_path.replace('experiment_log.csv', '')
        json_path = prefix + 'utility_cache.json'
        if not os.path.exists(json_path):
            # fallback: try to find any utility_cache with same prefix (case-insensitive)
            possible = glob.glob(prefix + '*utility_cache.json')
            if possible:
                json_path = possible[0]
            else:
                print(f"Warning: No utility_cache.json found for {csv_path}, utility will be empty.")

        util_mapping = load_utility_cache(json_path)
        groups = read_log(csv_path)

        # Use a simplified experiment ID from the file name
        exp_id = os.path.basename(csv_path).replace('experiment_log.csv', '').rstrip('_')

        for strategy, rows in groups.items():
            # ----- Accuracy: use the final row (final==True) -----
            final_rows = [r for r in rows if r.get('final') == True]
            final_row = final_rows[0] if final_rows else rows[-1]
            test_acc = final_row.get('test_accuracy')
            if test_acc is not None:
                accuracy_data[strategy][exp_id] = test_acc

            # ----- Utility: average over ALL selected pairs across all rounds -----
            all_deltas = []
            for row in rows:
                selected_ids = row.get('selected_pair_ids', [])
                for cid in selected_ids:
                    if cid in util_mapping:
                        all_deltas.append(util_mapping[cid])
            if all_deltas:
                avg_util = sum(all_deltas) / len(all_deltas)
                utility_data[strategy][exp_id] = avg_util

    acc_df = pd.DataFrame(accuracy_data).T.fillna('-')
    util_df = pd.DataFrame(utility_data).T.fillna('-')

    print("===== Accuracy Table (Final test accuracy per strategy and experiment) =====")
    print(acc_df)
    print("\n===== Utility Table (Average delta_accuracy over ALL selected pairs) =====")
    print(util_df)

    acc_df.to_csv('accuracy_summary_all_rounds.csv')
    util_df.to_csv('utility_summary_all_rounds.csv')
    print("\nTables saved to accuracy_summary_all_rounds.csv and utility_summary_all_rounds.csv")

if __name__ == '__main__':
    main()