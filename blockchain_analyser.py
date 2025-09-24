import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
import networkx as nx

class BlockchainAnalyzer:
    def __init__(self, base_path="../tree_exports"):
        self.base_path = base_path
        self.simulation_params = [
            # (sim_id, n, z0, z1, ttx, I, T)
            (1, 10, 20, 30, 1000.0, 10000.0, 120000.0),
            (2, 10, 20, 30, 1000.0, 3000.0, 120000.0),
            (3, 10, 20, 30, 1000.0, 1000.0, 120000.0),
            (4, 10, 20, 30, 1000.0, 500.0, 120000.0),
            (5, 10, 50, 50, 1000.0, 10000.0, 120000.0),
            (6, 10, 50, 50, 1000.0, 3000.0, 120000.0),
            (7, 10, 50, 50, 1000.0, 1000.0, 120000.0),
            (8, 10, 50, 50, 1000.0, 500.0, 120000.0),
            (9, 20, 30, 30, 1000.0, 10000.0, 120000.0),
            (10, 20, 30, 30, 1000.0, 3000.0, 120000.0),
            (11, 20, 30, 30, 1000.0, 1000.0, 120000.0),
            (12, 20, 30, 30, 1000.0, 500.0, 120000.0),
            (13, 20, 50, 20, 1000.0, 10000.0, 120000.0),
            (14, 20, 50, 20, 1000.0, 3000.0, 120000.0),
            (15, 20, 50, 20, 1000.0, 1000.0, 120000.0),
            (16, 20, 50, 20, 1000.0, 500.0, 120000.0),
        ]
        self.param_map = {sim_id: params for sim_id, *params in self.simulation_params}
        
    def load_simulation_data(self):
        """Load data from all simulation folders"""
        all_data = []
        
        for sim_folder in sorted(os.listdir(self.base_path)):
            if sim_folder.startswith("simulation_"):
                sim_id = int(sim_folder.split("_")[1])
                sim_path = os.path.join(self.base_path, sim_folder)
                
                # Get simulation parameters
                if sim_id in self.param_map:
                    n, z0, z1, ttx, I, T = self.param_map[sim_id]
                else:
                    print(f"Warning: No parameters found for simulation {sim_id}")
                    continue
                
                # Load all node files for this simulation
                sim_data = self.load_simulation_nodes(sim_path, sim_id, n, z0, z1, ttx, I, T)
                all_data.extend(sim_data)
        
        return pd.DataFrame(all_data)
    
    def load_simulation_nodes(self, sim_path, sim_id, n, z0, z1, ttx, I, T):
        """Load data from all nodes in a simulation"""
        node_data = []
        
        for file_name in os.listdir(sim_path):
            if file_name.startswith("node_") and file_name.endswith("_tree.json"):
                file_path = os.path.join(sim_path, file_name)
                
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Add simulation parameters
                    data['sim_id'] = sim_id
                    data['n'] = n
                    data['z0'] = z0
                    data['z1'] = z1
                    data['ttx'] = ttx
                    data['I'] = I
                    data['T'] = T
                    data['blocks_per_second'] = 1000.0 / I  # Convert ms to blocks/second
                    
                    # Calculate derived metrics
                    data['success_ratio'] = (data['self_blocks_in_longest_chain'] / data['blocks_created'] 
                                           if data['blocks_created'] > 0 else 0)
                    
                    node_data.append(data)
                    
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        return node_data
    
    def calculate_throughput_and_stale_rate(self, df):
        """Calculate throughput and stale rate for each simulation"""
        results = []
        
        for sim_id in df['sim_id'].unique():
            sim_data = df[df['sim_id'] == sim_id]
            
            # Calculate throughput: total non-coinbase transactions in longest chain / time
            total_non_coinbase_txns = 0
            total_longest_chain_blocks = 0
            
            # Get one representative node for longest chain analysis
            representative_node = sim_data.iloc[0]
            
            # Count transactions in longest chain (assuming each block has at least 1 coinbase)
            if 'tree_nodes' in representative_node and representative_node['tree_nodes']:
                longest_chain_blocks = [
                    block for block in representative_node['tree_nodes'].values()
                    if block.get('is_in_longest_chain', False)
                ]
                
                total_non_coinbase_txns = sum(
                    max(0, block['num_transactions'] - 1)  # Subtract coinbase transaction
                    for block in longest_chain_blocks
                )
                total_longest_chain_blocks = len(longest_chain_blocks)
            
            throughput = total_non_coinbase_txns / (representative_node['T'] / 1000.0)  # txns per second
            
            # Calculate stale rate: orphaned blocks / total blocks
            total_orphaned = sim_data['total_blocks'].sum() - sim_data['longest_chain_length'].sum()
            total_blocks = sim_data['total_blocks'].sum()
            stale_rate = total_orphaned / total_blocks if total_blocks > 0 else 0
            
            results.append({
                'sim_id': sim_id,
                'n': representative_node['n'],
                'z0': representative_node['z0'],
                'z1': representative_node['z1'],
                'I': representative_node['I'],
                'blocks_per_second': representative_node['blocks_per_second'],
                'throughput': throughput,
                'stale_rate': stale_rate,
                'total_blocks': total_blocks,
                'total_orphaned': total_orphaned,
                'longest_chain_length': representative_node['longest_chain_length'],
                'total_non_coinbase_txns': total_non_coinbase_txns
            })
        
        return pd.DataFrame(results)
    
    def create_main_plots(self, results_df):
        """Create the required throughput and stale rate plots"""
        # Group configurations
        configs = [
            (10, 20, 30, "n=10, 20% slow, 30% low CPU"),
            (10, 50, 50, "n=10, 50% slow, 50% low CPU"), 
            (20, 30, 30, "n=20, 30% slow, 30% low CPU"),
            (20, 50, 20, "n=20, 50% slow, 20% low CPU")
        ]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        axes = [ax1, ax2, ax3, ax4]
        
        for i, (n, z0, z1, label) in enumerate(configs):
            config_data = results_df[
                (results_df['n'] == n) & 
                (results_df['z0'] == z0) & 
                (results_df['z1'] == z1)
            ].sort_values('blocks_per_second')
            
            if not config_data.empty:
                # Throughput plot
                axes[i].plot(config_data['blocks_per_second'], config_data['throughput'], 
                           'o-', color='blue', linewidth=2, markersize=8, label='Throughput')
                axes[i].set_xlabel('1/I (blocks/s)', fontsize=12)
                axes[i].set_ylabel('Throughput (txns/s)', fontsize=12)
                axes[i].set_title(f'Throughput vs Block Rate\n{label}', fontsize=12, fontweight='bold')
                axes[i].grid(True, alpha=0.3)
                axes[i].legend()
        
        plt.tight_layout()
        plt.savefig('throughput_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Stale Rate Plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        axes = [ax1, ax2, ax3, ax4]
        
        for i, (n, z0, z1, label) in enumerate(configs):
            config_data = results_df[
                (results_df['n'] == n) & 
                (results_df['z0'] == z0) & 
                (results_df['z1'] == z1)
            ].sort_values('blocks_per_second')
            
            if not config_data.empty:
                # Stale rate plot
                axes[i].plot(config_data['blocks_per_second'], config_data['stale_rate'], 
                           'o-', color='red', linewidth=2, markersize=8, label='Stale Rate')
                axes[i].set_xlabel('1/I (blocks/s)', fontsize=12)
                axes[i].set_ylabel('Stale Rate', fontsize=12)
                axes[i].set_title(f'Stale Rate vs Block Rate\n{label}', fontsize=12, fontweight='bold')
                axes[i].grid(True, alpha=0.3)
                axes[i].legend()
        
        plt.tight_layout()
        plt.savefig('stale_rate_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_node_performance(self, df):
        """Analyze performance by node characteristics"""
        print("=== Node Performance Analysis ===\n")
        
        # Group by node characteristics
        performance = df.groupby(['is_fast', 'is_high_cpu']).agg({
            'success_ratio': ['mean', 'std', 'count'],
            'blocks_created': ['mean', 'sum'],
            'self_blocks_in_longest_chain': ['mean', 'sum']
        }).round(4)
        
        print("Performance by Node Type:")
        print(performance)
        print()
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Success ratio by node type
        node_types = []
        success_ratios = []
        for (is_fast, is_high_cpu), group in df.groupby(['is_fast', 'is_high_cpu']):
            node_type = f"{'Fast' if is_fast else 'Slow'} + {'High CPU' if is_high_cpu else 'Low CPU'}"
            node_types.append(node_type)
            success_ratios.append(group['success_ratio'].mean())
        
        ax1.bar(node_types, success_ratios, color=['green', 'orange', 'red', 'darkred'])
        ax1.set_ylabel('Average Success Ratio')
        ax1.set_title('Block Success Ratio by Node Type')
        ax1.tick_params(axis='x', rotation=45)
        
        # Blocks created vs blocks in longest chain
        for (is_fast, is_high_cpu), group in df.groupby(['is_fast', 'is_high_cpu']):
            node_type = f"{'Fast' if is_fast else 'Slow'} + {'High CPU' if is_high_cpu else 'Low CPU'}"
            ax2.scatter(group['blocks_created'], group['self_blocks_in_longest_chain'], 
                       label=node_type, alpha=0.7, s=50)
        
        ax2.plot([0, df['blocks_created'].max()], [0, df['blocks_created'].max()], 
                'k--', alpha=0.5, label='Perfect Success Line')
        ax2.set_xlabel('Blocks Created')
        ax2.set_ylabel('Blocks in Longest Chain')
        ax2.set_title('Block Success by Node Type')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('node_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return performance
    
    def analyze_branch_lengths_500ms_sims(self, df):
        """Analyze branch lengths specifically for 500ms block time simulations (sim 4, 8, 12, 16)"""
        print("=== Branch Length Analysis for 500ms Simulations ===\n")
        
        # Filter for only 500ms simulations
        target_sims = [4, 8, 12, 16]
        filtered_df = df[df['sim_id'].isin(target_sims)].copy()
        
        if filtered_df.empty:
            print("No data found for simulations 4, 8, 12, 16")
            return pd.DataFrame()
        
        branch_data = []
        
        for _, node_data in filtered_df.iterrows():
            if 'tree_nodes' not in node_data or not node_data['tree_nodes']:
                continue
                
            tree_nodes = node_data['tree_nodes']
            
            # Find all blocks not in longest chain (potential branch starting points)
            non_longest_chain_blocks = [
                block for block in tree_nodes.values()
                if not block.get('is_in_longest_chain', False) and not block.get('is_genesis', False)
            ]
            
            # Find the longest branch for this node's tree
            max_branch_length = 0
            longest_branch_block_id = None
            
            # for block in non_longest_chain_blocks:
            #     branch_length = self.calculate_longest_branch_from_block(block['block_id'], tree_nodes)
            #     if branch_length > max_branch_length:
            #         max_branch_length = branch_length
            #         longest_branch_block_id = block['block_id']

            max_branch_length, longest_branch_block_id = self.calculate_longest_branch_in_tree(tree_nodes)
            
            # Get simulation parameters for context
            sim_params = self.param_map[node_data['sim_id']]
            n, z0, z1, ttx, I, T = sim_params
            
            branch_data.append({
                'sim_id': node_data['sim_id'],
                'node_id': node_data['node_id'],
                'is_fast': node_data['is_fast'],
                'is_high_cpu': node_data['is_high_cpu'],
                'longest_branch_length': max_branch_length,
                'longest_branch_block_id': longest_branch_block_id,
                'total_blocks': node_data['total_blocks'],
                'blocks_created': node_data['blocks_created'],
                'self_blocks_in_longest_chain': node_data['self_blocks_in_longest_chain'],
                'success_ratio': node_data['success_ratio'],
                'n': n,
                'z0': z0,
                'z1': z1,
                'num_alternative_branches': len(non_longest_chain_blocks)
            })
        
        branch_df = pd.DataFrame(branch_data)
        
        if branch_df.empty:
            print("No branch data found")
            return branch_df
        
        # Print summary statistics
        print(f"Analyzed {len(branch_df)} nodes across simulations {target_sims}")
        print(f"Average longest branch length: {branch_df['longest_branch_length'].mean():.2f}")
        print(f"Max longest branch length: {branch_df['longest_branch_length'].max()}")
        print(f"Min longest branch length: {branch_df['longest_branch_length'].min()}")
        print()
        
        print("Longest branch length by node type:")
        node_type_stats = branch_df.groupby(['is_fast', 'is_high_cpu']).agg({
            'longest_branch_length': ['mean', 'std', 'max', 'count']
        }).round(2)
        print(node_type_stats)
        print()
        
        # Create visualization similar to node performance analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Define node types and colors
        node_types = {
            (True, True): ('Fast + High CPU', 'green'),
            (True, False): ('Fast + Low CPU', 'orange'), 
            (False, True): ('Slow + High CPU', 'blue'),
            (False, False): ('Slow + Low CPU', 'red')
        }
        
        # Plot 1: Longest Branch Length by Node Type (Bar Chart)
        node_type_means = []
        node_type_labels = []
        node_type_colors = []
        
        for (is_fast, is_high_cpu), (label, color) in node_types.items():
            group_data = branch_df[(branch_df['is_fast'] == is_fast) & (branch_df['is_high_cpu'] == is_high_cpu)]
            if not group_data.empty:
                node_type_means.append(group_data['longest_branch_length'].mean())
                node_type_labels.append(label)
                node_type_colors.append(color)
        
        bars = ax1.bar(node_type_labels, node_type_means, color=node_type_colors, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Average Longest Branch Length', fontsize=12)
        ax1.set_title('Average Longest Branch Length by Node Type\n(500ms Block Time Simulations)', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, node_type_means):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Scatter - Longest Branch Length vs Blocks Created (colored by node type)
        for (is_fast, is_high_cpu), (label, color) in node_types.items():
            group_data = branch_df[(branch_df['is_fast'] == is_fast) & (branch_df['is_high_cpu'] == is_high_cpu)]
            if not group_data.empty:
                ax2.scatter(group_data['blocks_created'], group_data['longest_branch_length'], 
                          label=label, color=color, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
        
        ax2.set_xlabel('Blocks Created by Node', fontsize=12)
        ax2.set_ylabel('Longest Branch Length', fontsize=12)
        ax2.set_title('Longest Branch Length vs Blocks Created\n(500ms Block Time Simulations)', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Longest Branch Length vs Success Ratio
        for (is_fast, is_high_cpu), (label, color) in node_types.items():
            group_data = branch_df[(branch_df['is_fast'] == is_fast) & (branch_df['is_high_cpu'] == is_high_cpu)]
            if not group_data.empty:
                ax3.scatter(group_data['success_ratio'], group_data['longest_branch_length'], 
                          label=label, color=color, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
        
        ax3.set_xlabel('Success Ratio (Blocks in Chain / Blocks Created)', fontsize=12)
        ax3.set_ylabel('Longest Branch Length', fontsize=12)
        ax3.set_title('Longest Branch Length vs Success Ratio\n(500ms Block Time Simulations)', fontsize=12, fontweight='bold')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Longest Branch Length by Simulation Configuration
        sim_configs = {
            4: "n=10, 20% slow, 30% low CPU",
            8: "n=10, 50% slow, 50% low CPU", 
            12: "n=20, 30% slow, 30% low CPU",
            16: "n=20, 50% slow, 20% low CPU"
        }
        
        sim_means = []
        sim_labels = []
        sim_colors = ['lightcoral', 'lightsalmon', 'lightblue', 'lightsteelblue']
        
        for sim_id in target_sims:
            sim_data = branch_df[branch_df['sim_id'] == sim_id]
            if not sim_data.empty:
                sim_means.append(sim_data['longest_branch_length'].mean())
                sim_labels.append(f"Sim {sim_id}\n{sim_configs[sim_id]}")
        
        bars = ax4.bar(range(len(sim_means)), sim_means, color=sim_colors[:len(sim_means)], 
                      alpha=0.7, edgecolor='black')
        ax4.set_xticks(range(len(sim_labels)))
        ax4.set_xticklabels(sim_labels, fontsize=10)
        ax4.set_ylabel('Average Longest Branch Length', fontsize=12)
        ax4.set_title('Average Longest Branch Length by Simulation\n(500ms Block Time)', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, sim_means):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('longest_branch_analysis_500ms.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return branch_df
    
    def calculate_longest_branch_from_block(self, start_block_id, tree_nodes):
        """Calculate the longest path from a given block to any leaf (tip)"""
        
        def dfs_longest_path(block_id, visited):
            """DFS to find longest path from current block to any leaf"""
            if block_id in visited:
                return 0  # Avoid cycles
            
            visited.add(block_id)
            
            block = tree_nodes.get(block_id)
            if not block:
                visited.remove(block_id)
                return 0
            
            children = block.get('children', [])
            
            if not children:
                # This is a leaf node
                visited.remove(block_id)
                return 1
            
            # Find the longest path among all children
            max_length = 0
            for child_id in children:
                child_length = dfs_longest_path(child_id, visited)
                max_length = max(max_length, child_length)
            
            visited.remove(block_id)
            return 1 + max_length
        
        return dfs_longest_path(start_block_id, set())
    

    def calculate_longest_branch_in_tree(self, tree_nodes):
        """
        Return (max_branch_length, starting_block_id) for the given tree_nodes.
        - max_branch_length: number of blocks in the longest branch (counts the starting non-main block).
        - starting_block_id: block id where the branch forks off the main chain (None if no branches).
        Assumes tree_nodes is a dict: block_id -> block_dict, and each block_dict has:
        - 'children': list of child block_ids
        - 'is_in_longest_chain': bool
        - optional: 'is_genesis'
        """
        # build parent map (child_id -> parent_id)
        parent = {}
        for bid, block in tree_nodes.items():
            for c in block.get('children', []):
                parent[c] = bid

        # collect all non-main blocks
        non_main_blocks = [bid for bid, b in tree_nodes.items()
                        if not b.get('is_in_longest_chain', False) and not b.get('is_genesis', False)]

        if not non_main_blocks:
            return 0, None

        # find fork starters: non-main blocks whose parent is None or is in the main chain
        starters = []
        for bid in non_main_blocks:
            p = parent.get(bid)
            if p is None or tree_nodes.get(p, {}).get('is_in_longest_chain', False):
                starters.append(bid)

        # if no starters found (weird tree shape), fallback to using all non-main blocks as possible starts
        if not starters:
            starters = non_main_blocks

        # DFS that only follows non-main children
        def dfs_len(start_id, visited):
            if start_id in visited:
                return 0
            visited.add(start_id)
            block = tree_nodes.get(start_id)
            if not block:
                visited.remove(start_id)
                return 0

            max_len = 0
            for c in block.get('children', []):
                # don't traverse into main-chain children
                if tree_nodes.get(c, {}).get('is_in_longest_chain', False):
                    continue
                child_len = dfs_len(c, visited)
                if child_len > max_len:
                    max_len = child_len

            visited.remove(start_id)
            return 1 + max_len  # count this block

        # compute longest branch among starters
        max_branch_len = 0
        max_start = None
        for s in starters:
            length = dfs_len(s, set())
            if length > max_branch_len:
                max_branch_len = length
                max_start = s

        return max_branch_len, max_start

    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("Loading simulation data...")
        df = self.load_simulation_data()
        print(f"Loaded data for {len(df)} nodes across {df['sim_id'].nunique()} simulations\n")
        
        print("Calculating throughput and stale rates...")
        results_df = self.calculate_throughput_and_stale_rate(df)
        
        print("Creating main plots...")
        self.create_main_plots(results_df)
        
        print("Analyzing node performance...")
        performance_stats = self.analyze_node_performance(df)
        
        print("Analyzing longest branches for 500ms simulations...")
        branch_stats_500ms = self.analyze_branch_lengths_500ms_sims(df)
        
        # Print summary statistics
        print("=== Summary Statistics ===")
        print("\nThroughput and Stale Rate by Configuration:")
        summary = results_df.groupby(['n', 'z0', 'z1']).agg({
            'throughput': ['mean', 'std'],
            'stale_rate': ['mean', 'std']
        }).round(4)
        print(summary)
        
        return df, results_df, performance_stats, branch_stats_500ms

# Usage
if __name__ == "__main__":
    analyzer = BlockchainAnalyzer("../tree_exports")  # Update path as needed
    df, results_df, performance_stats, branch_stats = analyzer.run_complete_analysis()