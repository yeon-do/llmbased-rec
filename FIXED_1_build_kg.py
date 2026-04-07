#!/usr/bin/env python3
"""
Build Knowledge Graph for GRAM datasets (Beauty, Sports, Toys, Yelp)
- Item nodes use hierarchical IDs (not titles)
- Title becomes an attribute edge
- Creates UNDIRECTED co-interaction graph from user_sequence.txt (using [:-2] for training)
- Creates item-attribute graph from item_plain_text.txt
- Saves as pickle format for easy loading

Usage:
    python build_kg.py --dataset Beauty
    python build_kg.py --dataset Sports
    python build_kg.py --dataset Toys
    python build_kg.py --dataset Yelp
"""

import os
import sys
import json
import pickle
import argparse
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder


# ======================
# Configuration
# ======================
def get_paths(dataset_name):
    """Get paths for a given dataset"""
    base_dir = os.path.join(os.path.dirname(__file__), "rec_datasets", dataset_name)
    kg_dir = os.path.join(base_dir, "kg")
    
    # Find hierarchical ID file
    hierarchical_id_file = None
    pattern = os.path.join(base_dir, "item_generative_indexing_hierarchy_*.txt")
    matches = glob.glob(pattern)
    if matches:
        hierarchical_id_file = matches[0]
    
    return {
        "user_sequence": os.path.join(base_dir, "user_sequence.txt"),
        "item_plain_text": os.path.join(base_dir, "item_plain_text.txt"),
        "hierarchical_id": hierarchical_id_file,
        "out_graph": os.path.join(kg_dir, "knowledge_graph.pkl"),
        "out_meta_dir": kg_dir,
    }


# ======================
# Utilities
# ======================
def is_invalid(x) -> bool:
    """Check if value is invalid (None, NaN, empty, or 'na'/'unknown')"""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return True
    s = str(x).strip().lower()
    if s == "" or s in {"na", "unknown", "n/a", "none"}:
        return True
    return False


def norm_str(x):
    """Normalize string value"""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip()
    # Remove surrounding quotes
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        s = s[1:-1].strip()
    if s == "" or s.lower() in {"na", "unknown", "n/a", "none"}:
        return np.nan
    return s


def parse_item_plain_text(line):
    """
    Parse item_plain_text.txt line
    Format: ITEM_ID title: ...; brand: ...; categories: ...; description: ...; price: ...; salesrank: ...
    """
    line = line.strip()
    if not line:
        return None
    
    parts = line.split(maxsplit=1)
    if len(parts) < 2:
        return None
    
    item_id = parts[0]
    content = parts[1]
    
    # Parse fields
    fields = {}
    current_key = None
    current_value = []
    
    for segment in content.split(";"):
        segment = segment.strip()
        if not segment:
            continue
        
        if ":" in segment:
            # Save previous field
            if current_key:
                fields[current_key] = " ".join(current_value).strip()
            
            # Start new field
            key, value = segment.split(":", 1)
            current_key = key.strip()
            current_value = [value.strip()]
        else:
            # Continue current field
            current_value.append(segment)
    
    # Save last field
    if current_key:
        fields[current_key] = " ".join(current_value).strip()
    
    return {
        "item_id": item_id,
        "title": fields.get("title", ""),
        "brand": fields.get("brand", ""),
        "categories": fields.get("categories", ""),
        "description": fields.get("description", ""),
        "price": fields.get("price", ""),
    }


def parse_hierarchical_id(line):
    """
    Parse hierarchical ID line
    Format: ITEM_ID |hierarchical|tokens|...
    Returns: (item_id, hierarchical_id_string)
    """
    line = line.strip()
    if not line:
        return None, None
    
    parts = line.split(maxsplit=1)
    if len(parts) < 2:
        return None, None
    
    item_id = parts[0]
    hierarchical_id = parts[1].strip()  # Keep the full hierarchical ID string
    
    return item_id, hierarchical_id


def tokenize_categories(s):
    """Split categories by comma"""
    if is_invalid(s):
        return []
    s = norm_str(s)
    if is_invalid(s):
        return []
    toks = [norm_str(t) for t in str(s).split(",")]
    return [t for t in toks if not is_invalid(t)]


def collect_valid(values):
    """Collect valid (non-empty) values from a list"""
    vals = []
    for v in values:
        if not is_invalid(v):
            vals.append(str(v).strip())
    return vals


# ======================
# Main Build Function
# ======================
def build_kg(dataset_name):
    """Build knowledge graph for a dataset"""
    
    print("=" * 70)
    print(f"BUILDING KNOWLEDGE GRAPH FOR {dataset_name.upper()}")
    print("=" * 70)
    
    paths = get_paths(dataset_name)
    
    # Check if input files exist
    if not os.path.exists(paths["user_sequence"]):
        raise FileNotFoundError(f"user_sequence.txt not found: {paths['user_sequence']}")
    if not os.path.exists(paths["item_plain_text"]):
        raise FileNotFoundError(f"item_plain_text.txt not found: {paths['item_plain_text']}")
    if not paths["hierarchical_id"] or not os.path.exists(paths["hierarchical_id"]):
        raise FileNotFoundError(
            f"hierarchical ID file not found in rec_datasets/{dataset_name}/\n"
            f"Expected pattern: item_generative_indexing_hierarchy_*.txt"
        )
    
    # Create output directory
    os.makedirs(paths["out_meta_dir"], exist_ok=True)
    
    # ======================
    # 1) Load Hierarchical IDs
    # ======================
    print("\n[STEP 1/7] Loading hierarchical IDs...")
    hierarchical_id_map = {}  # item_id -> hierarchical_id_string
    
    with open(paths["hierarchical_id"], "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="  Reading hierarchical IDs"):
            item_id, hier_id = parse_hierarchical_id(line)
            if item_id and hier_id:
                hierarchical_id_map[item_id] = hier_id
    
    print(f"  ✓ Loaded {len(hierarchical_id_map):,} hierarchical IDs")
    
    # Get unique hierarchical IDs (these will be our item nodes)
    unique_hier_ids = sorted(set(hierarchical_id_map.values()))
    print(f"  ✓ Unique hierarchical IDs: {len(unique_hier_ids):,}")
    
    # ======================
    # 2) Load Items from item_plain_text.txt
    # ======================
    print("\n[STEP 2/7] Loading item metadata from item_plain_text.txt...")
    items_data = []
    
    with open(paths["item_plain_text"], "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="  Reading items"):
            item_dict = parse_item_plain_text(line)
            if item_dict:
                items_data.append(item_dict)
    
    items_df = pd.DataFrame(items_data)
    print(f"  ✓ Loaded {len(items_df):,} items")
    
    # Apply normalization (excluding title)
    for col in ["brand", "categories", "description", "price"]:
        items_df[col] = items_df[col].apply(norm_str)
    
    # Add hierarchical ID to items_df
    items_df["hierarchical_id"] = items_df["item_id"].map(hierarchical_id_map)
    
    # Filter items that have hierarchical IDs
    valid_items = items_df[items_df["hierarchical_id"].notna()].copy()
    valid_items = valid_items.reset_index(drop=True)
    print(f"  ✓ Valid items with hierarchical IDs: {len(valid_items):,}")
    
    # ======================
    # 3) Build Encoders
    # ======================
    print("\n[STEP 3/7] Building label encoders...")
    
    # Collect all unique values
    all_cats = []
    for v in valid_items["categories"].tolist():
        all_cats.extend(tokenize_categories(v))
    
    # 🔥 Filter out "Beauty" category (too general)
    all_cats = [c for c in all_cats if c.lower() != "beauty"]
    
    # Use hierarchical IDs as item identifiers
    hier_ids = sorted(set(valid_items["hierarchical_id"].tolist()))
    # 🔥 REMOVED: titles = collect_valid(valid_items["title"])
    brands = collect_valid(valid_items["brand"])
    descs = collect_valid(valid_items["description"])
    prices = collect_valid(valid_items["price"])
    cats_unique = sorted(set(all_cats))
    
    # Build encoders (excluding title)
    enc_item = LabelEncoder().fit(hier_ids)  # Items encoded by hierarchical ID
    # 🔥 REMOVED: enc_title = LabelEncoder().fit(titles if titles else ["<PAD>"])
    enc_brand = LabelEncoder().fit(brands if brands else ["<PAD>"])
    enc_desc = LabelEncoder().fit(descs if descs else ["<PAD>"])
    enc_price = LabelEncoder().fit(prices if prices else ["<PAD>"])
    enc_cat = LabelEncoder().fit(cats_unique if cats_unique else ["<PAD>"])
    
    print(f"  ✓ Items (hierarchical IDs): {len(enc_item.classes_):,}")
    # 🔥 REMOVED: print(f"  ✓ Titles: {len(enc_title.classes_):,}")
    print(f"  ✓ Brands: {len(enc_brand.classes_):,}")
    print(f"  ✓ Categories: {len(enc_cat.classes_):,} (excluding 'Beauty')")
    print(f"  ✓ Descriptions: {len(enc_desc.classes_):,}")
    print(f"  ✓ Prices: {len(enc_price.classes_):,}")
    
    # ======================
    # 4) Save Metadata
    # ======================
    print("\n[STEP 4/7] Saving metadata...")
    pd.Series(enc_item.classes_).to_csv(
        os.path.join(paths["out_meta_dir"], "item_classes.csv"), 
        index=False, header=False
    )
    # 🔥 REMOVED: title_classes.csv
    pd.Series(enc_brand.classes_).to_csv(
        os.path.join(paths["out_meta_dir"], "brand_classes.csv"), 
        index=False, header=False
    )
    pd.Series(enc_desc.classes_).to_csv(
        os.path.join(paths["out_meta_dir"], "description_classes.csv"), 
        index=False, header=False
    )
    pd.Series(enc_price.classes_).to_csv(
        os.path.join(paths["out_meta_dir"], "price_classes.csv"), 
        index=False, header=False
    )
    pd.Series(enc_cat.classes_).to_csv(
        os.path.join(paths["out_meta_dir"], "category_classes.csv"), 
        index=False, header=False
    )
    print(f"  ✓ Saved to {paths['out_meta_dir']}/")
    
    # ======================
    # 5) Item ID Mapping
    # ======================
    print("\n[STEP 5/7] Creating item ID mapping...")
    
    # Create mapping: hierarchical_id -> internal_id
    hier_id_to_iid = {hier_id: i for i, hier_id in enumerate(enc_item.classes_)}
    
    # Create mapping: original_id -> internal_id (via hierarchical_id)
    original_id_to_iid = {}
    for original_id, hier_id in hierarchical_id_map.items():
        if hier_id in hier_id_to_iid:
            original_id_to_iid[original_id] = hier_id_to_iid[hier_id]
    
    # Save mappings
    with open(os.path.join(paths["out_meta_dir"], "original_id_to_iid.json"), "w") as f:
        json.dump(original_id_to_iid, f, indent=2)
    
    with open(os.path.join(paths["out_meta_dir"], "hierarchical_id_to_iid.json"), "w") as f:
        json.dump(hier_id_to_iid, f, indent=2)
    
    print(f"  ✓ Mapped {len(original_id_to_iid):,} original IDs to internal IDs")
    print(f"  ✓ Mapped {len(hier_id_to_iid):,} hierarchical IDs to internal IDs")
    
    # ======================
    # 6) Build Co-interaction Edges (from user_sequence[:-2])
    # ======================
    print("\n[STEP 6/7] Building co-interaction edges (undirected) from user_sequence[:-2]...")
    
    cointeract = defaultdict(int)
    total_users = 0
    skipped_items = 0
    
    with open(paths["user_sequence"], "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="  Reading user interactions"):
            toks = line.strip().split()
            if len(toks) < 2:
                continue
            
            total_users += 1
            item_ids = toks[1:]
            
            # Use only [:-2] for training (validation=-2, test=-1)
            # NOTE: Do NOT truncate to 20 items here! KG should contain ALL items.
            # max_his=20 is applied only during TRAINING, not during KG construction.
            train_items = item_ids[:-2]
            
            # Map to internal IDs
            user_items = []
            for it in train_items:
                if it in original_id_to_iid:
                    user_items.append(original_id_to_iid[it])
                else:
                    skipped_items += 1
            
            # Create co-interaction edges (undirected)
            for i in range(len(user_items)):
                for j in range(i + 1, len(user_items)):
                    edge = (min(user_items[i], user_items[j]), 
                           max(user_items[i], user_items[j]))
                    cointeract[edge] += 1
    
    edges = {
        "co_interacted": [(i, j, cnt) for (i, j), cnt in cointeract.items()]
    }
    
    print(f"  ✓ Processed {total_users:,} users")
    print(f"  ✓ Co-interaction edges: {len(edges['co_interacted']):,}")
    if skipped_items > 0:
        print(f"  ⚠ Skipped {skipped_items:,} items not in hierarchical ID mapping")
    
    # ======================
    # 7) Build Attribute Edges
    # ======================
    print("\n[STEP 7/7] Building attribute edges...")
    
    # 🔥 REMOVED: edges["title"] = []
    edges["brand"] = []
    edges["category"] = []
    edges["description"] = []
    edges["price"] = []
    
    # Create lookup dictionaries (excluding title)
    # 🔥 REMOVED: title_to_id = {t: i for i, t in enumerate(enc_title.classes_)}
    brand_to_id = {b: i for i, b in enumerate(enc_brand.classes_)}
    desc_to_id = {d: i for i, d in enumerate(enc_desc.classes_)}
    price_to_id = {p: i for i, p in enumerate(enc_price.classes_)}
    cat_to_id = {c: i for i, c in enumerate(enc_cat.classes_)}
    
    for _, row in tqdm(valid_items.iterrows(), total=len(valid_items), desc="  Processing items"):
        hier_id = row["hierarchical_id"]
        iid = hier_id_to_iid[hier_id]
        
        # 🔥 REMOVED: Title edge
        
        # Brand edge
        brand = row["brand"]
        if not is_invalid(brand):
            edges["brand"].append((iid, brand_to_id[str(brand)]))
        
        # Description edge
        desc = row["description"]
        if not is_invalid(desc):
            edges["description"].append((iid, desc_to_id[str(desc)]))
        
        # Price edge
        price = row["price"]
        if not is_invalid(price):
            edges["price"].append((iid, price_to_id[str(price)]))
        
        # Category edges (can have multiple, excluding "Beauty")
        for c in tokenize_categories(row["categories"]):
            # 🔥 Skip "Beauty" category
            if c.lower() != "beauty" and c in cat_to_id:
                edges["category"].append((iid, cat_to_id[c]))
    
    # 🔥 REMOVED: print(f"  ✓ Title edges: {len(edges['title']):,}")
    print(f"  ✓ Brand edges: {len(edges['brand']):,}")
    print(f"  ✓ Category edges: {len(edges['category']):,} (excluding 'Beauty')")
    print(f"  ✓ Description edges: {len(edges['description']):,}")
    print(f"  ✓ Price edges: {len(edges['price']):,}")
    
    # ======================
    # 8) Save Knowledge Graph
    # ======================
    print("\n[SAVING] Writing knowledge graph to disk...")
    
    kg_data = {
        "edges": edges,
        "num_nodes": {
            "item": len(enc_item.classes_),
            # 🔥 REMOVED: "title": len(enc_title.classes_),
            "brand": len(enc_brand.classes_),
            "category": len(enc_cat.classes_),
            "description": len(enc_desc.classes_),
            "price": len(enc_price.classes_),
        },
        "metadata": {
            "dataset": dataset_name,
            "num_users": total_users,
            "train_split": "user_sequence[:-2]",
            "item_node_type": "hierarchical_id",
            "hierarchical_id_file": os.path.basename(paths["hierarchical_id"]),
            "excluded_attributes": ["title"],  # 🔥 NEW
            "excluded_categories": ["Beauty"],  # 🔥 NEW
        }
    }
    
    with open(paths["out_graph"], "wb") as f:
        pickle.dump(kg_data, f)
    
    # ======================
    # Summary
    # ======================
    print("\n" + "=" * 70)
    print("✓ SUCCESS!")
    print("=" * 70)
    print(f"\nDataset: {dataset_name}")
    print(f"Users processed: {total_users:,}")
    print(f"Item node type: Hierarchical ID")
    print(f"\nNodes:")
    for k, v in kg_data["num_nodes"].items():
        print(f"  {k:12s}: {v:,}")
    print(f"\nEdges:")
    total_edges = 0
    for k, v in edges.items():
        print(f"  {k:12s}: {len(v):,}")
        total_edges += len(v)
    print(f"\nTotal edges: {total_edges:,}")
    print(f"\n✓ Graph saved to: {paths['out_graph']}")
    print(f"✓ Metadata saved to: {paths['out_meta_dir']}")
    print("=" * 70)
    
    return kg_data


# ======================
# Load Function
# ======================
def load_kg(dataset_name):
    """
    Load pre-built knowledge graph for a dataset
    
    Args:
        dataset_name: Name of dataset (Beauty, Sports, Toys, Yelp)
    
    Returns:
        kg_data: Dictionary containing:
            - edges: Dict of edge lists for each edge type
            - num_nodes: Dict of node counts for each node type
            - metadata: Additional information
    """
    paths = get_paths(dataset_name)
    
    if not os.path.exists(paths["out_graph"]):
        raise FileNotFoundError(
            f"Knowledge graph not found: {paths['out_graph']}\n"
            f"Please run: python build_kg.py --dataset {dataset_name}"
        )
    
    with open(paths["out_graph"], "rb") as f:
        kg_data = pickle.load(f)
    
    return kg_data


# ======================
# Main
# ======================
def main():
    parser = argparse.ArgumentParser(description="Build Knowledge Graph for GRAM datasets")
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True,
        choices=["Beauty", "Sports", "Toys", "Yelp"],
        help="Dataset name"
    )
    
    args = parser.parse_args()
    
    try:
        build_kg(args.dataset)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
