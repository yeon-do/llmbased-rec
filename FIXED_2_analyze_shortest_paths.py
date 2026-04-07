#!/usr/bin/env python3
"""
Analyze Shortest Paths in Knowledge Graph for Training Samples

For each training sample:
- Input sequence: user_sequence[:-2] (augmented)
- Find shortest paths from last item in input to target item
- CONSTRAINT: Exclude co-interaction edges (force at least 2-hop paths)

Statistics to compute:
1. Number of shortest paths per sample (min, max, avg)
2. Edge types used in paths
3. Path lengths (min, max, avg)

Usage:
    python 2_analyze_shortest_paths.py --dataset Beauty --save_paths
"""

import os
import sys

# IMPORTANT: Set CPU threads BEFORE other imports (서버 리소스 보호)
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['NUMEXPR_NUM_THREADS'] = '8'

import json
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict, Counter
import networkx as nx

print(f"CPU threads limited to: 8 (OMP_NUM_THREADS)")



def load_kg(dataset_name):
    """Load pre-built knowledge graph"""
    kg_path = f"rec_datasets/{dataset_name}/kg/knowledge_graph.pkl"
    
    if not os.path.exists(kg_path):
        raise FileNotFoundError(
            f"Knowledge graph not found: {kg_path}\n"
            f"Please run: python build_kg.py --dataset {dataset_name}"
        )
    
    with open(kg_path, "rb") as f:
        kg_data = pickle.load(f)
    
    return kg_data


def load_metadata(dataset_name):
    """Load metadata"""
    kg_dir = f"rec_datasets/{dataset_name}/kg"
    
    with open(f"{kg_dir}/original_id_to_iid.json", "r") as f:
        original_id_to_iid = json.load(f)
    
    return original_id_to_iid


def build_networkx_graph(kg_data):
    """
    Build NetworkX graph from KG data (Undirected)
    
    Args:
        kg_data: KG dictionary
    
    Returns:
        G: NetworkX MultiGraph with edge types
    """
    G = nx.MultiGraph()  # Undirected graph
    
    # Add nodes
    num_items = kg_data['num_nodes']['item']
    G.add_nodes_from(range(num_items), node_type='item')
    
    # Add attribute nodes (excluding title)
    offset = num_items
    for attr_type in ['brand', 'category', 'description', 'price']:
        if attr_type in kg_data['num_nodes']:
            num_attr_nodes = kg_data['num_nodes'][attr_type]
            G.add_nodes_from(
                range(offset, offset + num_attr_nodes),
                node_type=attr_type
            )
            offset += num_attr_nodes
    
    # Add edges
    edge_offset = {
        'item': 0,
        # 🔥 REMOVED: 'title': num_items,
        'brand': num_items,
        'category': num_items + kg_data['num_nodes'].get('brand', 0),
        'description': num_items + kg_data['num_nodes'].get('brand', 0) + kg_data['num_nodes'].get('category', 0),
        'price': num_items + kg_data['num_nodes'].get('brand', 0) + kg_data['num_nodes'].get('category', 0) + kg_data['num_nodes'].get('description', 0),
    }
    
    for edge_type, edges in kg_data['edges'].items():
        if edge_type == 'co_interacted':
            # Add co-interaction edges (undirected)
            for i, j, cnt in edges:
                G.add_edge(i, j, edge_type=edge_type, weight=cnt)
        else:
            # Add attribute edges (undirected, so only add once)
            attr_type = edge_type
            for item_id, attr_id in edges:
                global_attr_id = edge_offset[attr_type] + attr_id
                G.add_edge(item_id, global_attr_id, edge_type=edge_type)
    
    return G


def has_direct_cointeraction(G, source, target):
    """
    Check if source and target are directly connected via co_interacted edge
    
    Args:
        G: NetworkX graph
        source: Source node
        target: Target node
    
    Returns:
        True if direct co_interacted edge exists
    """
    if not G.has_edge(source, target):
        return False
    
    # Check edge types
    edge_data = G.get_edge_data(source, target)
    if edge_data:
        for key, data in edge_data.items():
            if data.get('edge_type') == 'co_interacted':
                return True
    
    return False


def find_all_shortest_paths(G, source, target, exclude_direct_cointeraction=True, max_paths=None):
    """
    Find all shortest paths between source and target
    Excludes paths that use direct co-interaction edge between source and target
    
    Args:
        G: NetworkX graph
        source: Source node
        target: Target node
        exclude_direct_cointeraction: If True, exclude direct co-interaction edge
        max_paths: Maximum number of paths to return (None = unlimited)
    
    Returns:
        path_infos: List of dicts:
        {
         "nodes": [n1, n2, n3],
         "edge_weights": [w1, w2],
        }
        path_length: Length of shortest paths (None if no path exists)
    """
    try:
        # Check if there's a direct co-interaction edge
        has_direct = has_direct_cointeraction(G, source, target)
        
        if exclude_direct_cointeraction and has_direct:
            # Temporarily remove co-interaction edges between source and target
            edges_to_remove = []
            edge_data = G.get_edge_data(source, target)
            
            if edge_data:
                for key, data in edge_data.items():
                    if data.get('edge_type') == 'co_interacted':
                        edges_to_remove.append(key)
            
            # Remove edges
            for key in edges_to_remove:
                G.remove_edge(source, target, key=key)
            
            # Find paths
            try:
                if not nx.has_path(G, source, target):
                    # Restore edges
                    for key in edges_to_remove:
                        G.add_edge(source, target, key=key, edge_type='co_interacted')
                    return [], None
                
                path_length = nx.shortest_path_length(G, source, target)
                all_paths = list(nx.all_shortest_paths(G, source, target))
                
                # Restore edges
                for key in edges_to_remove:
                    G.add_edge(source, target, key=key, edge_type='co_interacted')
                
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                # Restore edges
                for key in edges_to_remove:
                    G.add_edge(source, target, key=key, edge_type='co_interacted')
                return [], None
        else:
            # No direct co-interaction or not excluding
            if not nx.has_path(G, source, target):
                return [], None
            
            path_length = nx.shortest_path_length(G, source, target)
            all_paths = list(nx.all_shortest_paths(G, source, target))
        
        # Limit number of paths if specified
        if max_paths and len(all_paths) > max_paths:
            all_paths = all_paths[:max_paths]

        path_infos = []
        for path in all_paths:
            edge_weights = []
            for i in range(len(path)-1):
                u, v = path[i], path[i+1]
                data = G.get_edge_data(u,v)
                if data:
                    first_edge = list(data.values())[0]
                    weight = first_edge.get('weight',1)
                else:
                    weight = 1
                edge_weights.append(weight)
            path_infos.append({
                "nodes":path,
                "edge_weights": edge_weights
                })
        return path_infos, path_length
    
    except:
        return [], None

def get_path_edge_types(G, path):
    """
    Get edge types along a path
    
    Args:
        G: NetworkX graph
        path: List of nodes
    
    Returns:
        edge_types: List of edge types along the path
    """
    edge_types = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        # Get edge data (there might be multiple edges)
        edge_data = G.get_edge_data(u, v)
        if edge_data:
            # Get the first edge's type
            first_edge = list(edge_data.values())[0]
            edge_types.append(first_edge.get('edge_type', 'unknown'))
    
    return edge_types


def load_training_samples_sliding(dataset_name, original_id_to_iid, max_his=20, skip_empty_his=True, max_samples=None):
    
    """
    Sliding-window augmentation identical to multi_task_dataset_gram.py.

    Returns:
        samples: list of dicts
            {
                "history": [...],
                "target": int,
                "sample_id": int
            }
    """

    user_seq_path = f"rec_datasets/{dataset_name}/user_sequence.txt"
    samples = []
    sample_id = 0  # unique sample id
    
    with open(user_seq_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading training samples"):
            toks = line.strip().split()
            if len(toks) < 2:
                continue
            
            item_ids = toks[1:]
            
            # Use only [:-2] for training (validation=-2, test=-1)
            # NOTE: Do NOT truncate to 20 items! max_his=20 is applied to HISTORY only.
            train_items = item_ids[:-2]  # Training split
            
            if len(train_items) == 0:
                continue
            
            # Map to internal IDs
            train_iids = []
            for it in train_items:
                if it in original_id_to_iid:
                    train_iids.append(original_id_to_iid[it])
            
            if len(train_iids) == 0:
                continue
            
            # sliding-window augmentation
            for i in range(len(train_iids)):
                # i = target index
                if i == 0 and skip_empty_his:
                    # empty history skip (matching multi_task_dataset_gram)
                    continue

                target = train_iids[i]
                history = train_iids[:i]

                # apply max history length
                if len(history) > max_his:
                    history = history[-max_his:]

                # create sample
                samples.append({
                    "history": history,
                    "target": target,
                    "sample_id": sample_id,
                })
                sample_id += 1

                # early stopping
                if max_samples and len(samples) >= max_samples:
                    return samples

    return samples


def analyze_shortest_paths(dataset_name, max_paths_per_sample=None, save_paths=False, simple_mode=False):
    """
    Analyze shortest paths for ALL training samples
    Excludes direct co-interaction edges between source and target
    
    Args:
        dataset_name: Dataset name
        max_paths_per_sample: Maximum paths per sample (None = unlimited)
        save_paths: If True, save actual paths to shortest_paths.pkl
        simple_mode: If True, use only last item → target (6x faster!)
    """
    
    print("=" * 70)
    print(f"SHORTEST PATH ANALYSIS FOR {dataset_name.upper()}")
    if simple_mode:
        print("(SIMPLE MODE - User-level only, 6x faster!)")
    print("=" * 70)
    
    # Load KG and metadata
    print("\n[1/5] Loading KG and metadata...")
    kg_data = load_kg(dataset_name)
    original_id_to_iid = load_metadata(dataset_name)
    print(f"  ✓ Loaded KG with {kg_data['num_nodes']['item']:,} items")
    
    # Build NetworkX graph (include all edges)
    print("\n[2/5] Building graph (including all edges)...")
    G = build_networkx_graph(kg_data)
    print(f"  ✓ Graph nodes: {G.number_of_nodes():,}")
    print(f"  ✓ Graph edges: {G.number_of_edges():,}")
    
    # Load training samples
    print(f"\n[3/5] Loading training samples...")
    samples = load_training_samples_sliding(dataset_name, original_id_to_iid, max_his=20)
    print(f"  ✓ Loaded {len(samples):,} training samples")
    
    # Analyze shortest paths
    print(f"\n[4/5] Analyzing shortest paths...")
   
    path_stats = {
            'num_paths': [],           # Number of shortest paths per sample
            'path_lengths': [],        # Length of shortest paths
            'edge_type_counts': Counter(),  # Count of each edge type used
            'path_edge_sequences': Counter(),  # Count of edge type sequences
            'no_path_count': 0,        # Number of samples with no path
            'samples_with_paths': 0,   # Number of samples with at least one path
            }
    # If saving paths, store them
    if save_paths:
        all_sample_paths = {}
    for sample in tqdm(samples, desc="  Finding paths"):
        
        sid = sample["sample_id"]
        history = sample["history"]

        if len(history) == 0:
            continue

        last_item = history[-1]
        target_item = sample["target"]

        # Find all shortest paths (excluding direct co-interaction)
        path_infos, path_length = find_all_shortest_paths(
            G, last_item, target_item, 
            exclude_direct_cointeraction=True,
            max_paths=max_paths_per_sample
        )
        
        if len(path_infos) == 0:
            path_stats['no_path_count'] += 1
            continue
        
        path_stats['samples_with_paths'] += 1
        path_stats['num_paths'].append(len(path_infos))
        path_stats['path_lengths'].append(path_length)
        
        # Save paths if requested
        if save_paths:
            all_sample_paths[sid] = {
                'history_iids': history,
                'source_iid': last_item,
                'target_iid': target_item,
                'path_infos': path_infos,
                'path_length': path_length
            }
        
        # Analyze edge types in paths
        for path in path_infos:
            nodes = path["nodes"]
            edge_types = get_path_edge_types(G, nodes)
            
            # Count individual edge types
            for edge_type in edge_types:
                path_stats['edge_type_counts'][edge_type] += 1
            
            # Count edge type sequences
            edge_seq = ' -> '.join(edge_types)
            path_stats['path_edge_sequences'][edge_seq] += 1
    
    # Print statistics
    print("\n[5/5] Computing statistics...")
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\nTotal samples analyzed: {len(samples):,}")
    print(f"Samples with paths: {path_stats['samples_with_paths']:,} ({path_stats['samples_with_paths']/len(samples)*100:.2f}%)")
    print(f"Samples without paths: {path_stats['no_path_count']:,} ({path_stats['no_path_count']/len(samples)*100:.2f}%)")
    
    if path_stats['num_paths']:
        print("\n" + "-" * 70)
        print("NUMBER OF SHORTEST PATHS PER SAMPLE")
        print("-" * 70)
        print(f"Min:  {min(path_stats['num_paths']):,}")
        print(f"Max:  {max(path_stats['num_paths']):,}")
        print(f"Mean: {np.mean(path_stats['num_paths']):.2f}")
        print(f"Median: {np.median(path_stats['num_paths']):.2f}")
        print(f"Std:  {np.std(path_stats['num_paths']):.2f}")
        
        # Distribution
        num_paths_array = np.array(path_stats['num_paths'])
        print("\nDistribution:")
        for percentile in [25, 50, 75, 90, 95, 99]:
            val = np.percentile(num_paths_array, percentile)
            print(f"  {percentile}th percentile: {val:.0f}")
    
    if path_stats['path_lengths']:
        print("\n" + "-" * 70)
        print("PATH LENGTHS (number of hops)")
        print("-" * 70)
        print(f"Min:  {min(path_stats['path_lengths'])}")
        print(f"Max:  {max(path_stats['path_lengths'])}")
        print(f"Mean: {np.mean(path_stats['path_lengths']):.2f}")
        print(f"Median: {np.median(path_stats['path_lengths']):.2f}")
        
        # Distribution
        path_len_counter = Counter(path_stats['path_lengths'])
        print("\nDistribution:")
        for length in sorted(path_len_counter.keys()):
            count = path_len_counter[length]
            pct = count / len(path_stats['path_lengths']) * 100
            print(f"  {length}-hop: {count:,} ({pct:.2f}%)")
    
    if path_stats['edge_type_counts']:
        print("\n" + "-" * 70)
        print("EDGE TYPE USAGE")
        print("-" * 70)
        total_edges = sum(path_stats['edge_type_counts'].values())
        for edge_type, count in path_stats['edge_type_counts'].most_common():
            pct = count / total_edges * 100
            print(f"  {edge_type:15s}: {count:,} ({pct:.2f}%)")
    
    if path_stats['path_edge_sequences']:
        print("\n" + "-" * 70)
        print("TOP EDGE TYPE SEQUENCES (Top 20)")
        print("-" * 70)
        total_seqs = sum(path_stats['path_edge_sequences'].values())
        for seq, count in path_stats['path_edge_sequences'].most_common(20):
            pct = count / total_seqs * 100
            print(f"  {count:6,} ({pct:5.2f}%)  {seq}")
    
    print("\n" + "=" * 70)
    
    # Save detailed results
    output_dir = f"rec_datasets/{dataset_name}/kg"
    output_file = os.path.join(output_dir, "shortest_path_analysis.json")
    
    results = {
        'total_samples': len(samples),
        'samples_with_paths': path_stats['samples_with_paths'],
        'samples_without_paths': path_stats['no_path_count'],
        'num_paths_stats': {
            'min': int(min(path_stats['num_paths'])) if path_stats['num_paths'] else None,
            'max': int(max(path_stats['num_paths'])) if path_stats['num_paths'] else None,
            'mean': float(np.mean(path_stats['num_paths'])) if path_stats['num_paths'] else None,
            'median': float(np.median(path_stats['num_paths'])) if path_stats['num_paths'] else None,
            'std': float(np.std(path_stats['num_paths'])) if path_stats['num_paths'] else None,
        },
        'path_length_stats': {
            'min': int(min(path_stats['path_lengths'])) if path_stats['path_lengths'] else None,
            'max': int(max(path_stats['path_lengths'])) if path_stats['path_lengths'] else None,
            'mean': float(np.mean(path_stats['path_lengths'])) if path_stats['path_lengths'] else None,
            'median': float(np.median(path_stats['path_lengths'])) if path_stats['path_lengths'] else None,
            'distribution': dict(Counter(path_stats['path_lengths'])),
        },
        'edge_type_counts': dict(path_stats['edge_type_counts']),
        'top_edge_sequences': dict(path_stats['path_edge_sequences'].most_common(50)),
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Detailed results saved to: {output_file}")
    
    # Save actual paths if requested
    if save_paths:
        paths_file = os.path.join(output_dir, "shortest_paths.pkl")
        print(f"\n[SAVE_PATHS] Saving actual paths to: {paths_file}")
        print(f"  ⚠ This file will be large (~several GB)")
        
        with open(paths_file, 'wb') as f:
            pickle.dump(all_sample_paths, f)
        
        file_size_mb = os.path.getsize(paths_file) / (1024 * 1024)
        print(f"  ✓ Saved {len(all_sample_paths):,} samples with paths")
        print(f"  ✓ File size: {file_size_mb:.1f} MB")
    
    print("=" * 70)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze shortest paths in KG")
    parser.add_argument(
        "--dataset",
        type=str,
        default="Beauty",
        choices=["Beauty", "Sports", "Toys", "Yelp"],
        help="Dataset name"
    )
    parser.add_argument(
        "--max_paths",
        type=int,
        default=None,
        help="Maximum number of paths per sample (None = unlimited)"
    )
    parser.add_argument(
        "--save_paths",
        action="store_true",
        help="Save actual paths to shortest_paths.pkl (required for golden path selection)"
    )
    
    args = parser.parse_args()
    
    try:
        # Install networkx if needed
        try:
            import networkx
        except ImportError:
            print("Installing networkx...")
            os.system("pip install networkx --break-system-packages -q")
            import networkx
        
        analyze_shortest_paths(
            args.dataset,
            max_paths_per_sample=args.max_paths,
            save_paths=args.save_paths,
        )
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
