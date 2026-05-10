#!/usr/bin/env python3
"""
generate_full_synthetic_data.py

Generates a rich synthetic multiplex temporal network that mimics the properties
of a real academic department (18 faculty, 120 PGR, 240 UG, 12 semesters,
4 collaboration layers). Outputs CSV and pickle files.

Author: Dr. Sanjay Agal
"""

import argparse
import numpy as np
import networkx as nx
import pandas as pd
import pickle
import os
from tqdm import tqdm
from collections import defaultdict

# ---------- Helper functions ----------
def set_seed(seed):
    np.random.seed(seed)

def assign_roles(num_nodes, num_faculty=18, num_pgr=120, num_ug=240):
    """Assign roles: faculty (0), PGR (1), UG (2)."""
    roles = np.zeros(num_nodes, dtype=int)
    roles[:num_faculty] = 0
    roles[num_faculty:num_faculty+num_pgr] = 1
    roles[num_faculty+num_pgr:] = 2
    np.random.shuffle(roles)
    return roles

def assign_domains(roles, num_domains=5):
    """Assign research domain (0..num_domains-1): faculty spread, students follow faculty."""
    domains = np.zeros(len(roles), dtype=int)
    faculty_ids = np.where(roles == 0)[0]
    for i, fid in enumerate(faculty_ids):
        domains[fid] = i % num_domains
    # Students: 70% same domain as a random faculty, 30% random
    for i, r in enumerate(roles):
        if r != 0:
            if np.random.rand() < 0.7:
                domains[i] = np.random.choice(domains[faculty_ids])
            else:
                domains[i] = np.random.randint(num_domains)
    return domains

def preferential_attachment_existing(G, new_node, m=2, p_attach=0.8):
    """Add edges from new_node to existing nodes with preferential attachment."""
    if len(G) == 0:
        return
    degrees = np.array([G.degree(n) for n in G.nodes()])
    if degrees.sum() == 0:
        probs = np.ones(len(G)) / len(G)
    else:
        probs = degrees / degrees.sum()
    # Choose m neighbors, with some randomness
    candidates = np.random.choice(list(G.nodes()), size=min(m*3, len(G)), replace=False, p=probs)
    for nb in candidates[:m]:
        G.add_edge(new_node, nb)

def evolve_layer_seasonal(base_graph, t, volatility=0.1, rewiring_prob=0.05):
    """Evolve a graph: add nodes, preferential attachment, random rewiring."""
    G = base_graph.copy()
    # Add new nodes (growth)
    n_new = max(1, int(0.05 * len(G) * (1 + 0.1*np.random.randn())))
    for _ in range(n_new):
        new_id = max(G.nodes()) + 1 if G.nodes() else 0
        G.add_node(new_id)
        preferential_attachment_existing(G, new_id, m=np.random.randint(1,4))
    # Rewire some edges (renewal)
    edges = list(G.edges())
    n_rewire = int(rewiring_prob * len(edges))
    for _ in range(n_rewire):
        if len(edges) == 0: break
        u, v = edges.pop(np.random.randint(len(edges)))
        if G.has_edge(u, v):
            G.remove_edge(u, v)
            # Add new edge between random non-adjacent nodes
            nodes = list(G.nodes())
            for _ in range(10):
                a, b = np.random.choice(nodes, 2, replace=False)
                if not G.has_edge(a, b):
                    G.add_edge(a, b)
                    break
    return G

def generate_collaboration_layer(num_nodes, roles, base_density, t, layer_name):
    """Generate a graph for one layer with role-specific edge probabilities."""
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    # Faculty-faculty: highest density
    # Faculty-PGR: medium
    # PGR-PGR: medium-low
    # UG involved only in course and project layers
    if layer_name == 'coauthorship':
        p_ff = 0.12
        p_fp = 0.08
        p_pp = 0.05
        p_rest = 0.01
    elif layer_name == 'supervision':
        # directed, but we store as undirected for simplicity; direction matters in analysis
        p_fp = 0.25   # faculty -> PGR
        p_rest = 0.0
    elif layer_name == 'project':
        p_ff = 0.15
        p_fp = 0.20
        p_pp = 0.12
        p_fu = 0.08
        p_pu = 0.10
        p_uu = 0.03
    elif layer_name == 'course':
        p_ff = 0.05
        p_fp = 0.10
        p_pp = 0.15
        p_fu = 0.12
        p_pu = 0.20
        p_uu = 0.08
    else:
        raise ValueError("Unknown layer")

    # Add edges based on probabilities
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            r_i, r_j = roles[i], roles[j]
            if (r_i, r_j) in [(0,0), (1,1), (2,2)]:
                if r_i == 0:
                    p = p_ff if layer_name != 'supervision' else 0.0
                elif r_i == 1:
                    p = p_pp if layer_name != 'supervision' else 0.0
                else:
                    p = p_uu if 'uu' in locals() else 0.01
            else:
                # cross-role
                if 0 in (r_i, r_j) and 1 in (r_i, r_j):
                    p = p_fp if layer_name != 'supervision' else 0.25
                elif 0 in (r_i, r_j) and 2 in (r_i, r_j):
                    p = p_fu if 'fu' in locals() else 0.05
                elif 1 in (r_i, r_j) and 2 in (r_i, r_j):
                    p = p_pu if 'pu' in locals() else 0.07
                else:
                    p = 0.01
            # Temporal evolution: increase density slowly
            p = min(0.5, p * (1 + 0.02 * t))
            if np.random.rand() < p:
                G.add_edge(i, j)
    return G

def generate_multiplex_temporal(num_nodes=378, num_semesters=12, seed=42):
    set_seed(seed)
    roles = assign_roles(num_nodes)
    domains = assign_domains(roles)
    node_attrs = pd.DataFrame({
        'node_id': np.arange(num_nodes),
        'role': roles,
        'domain': domains,
        'seniority': np.random.randint(0, num_semesters, num_nodes)  # semester joined
    })
    snapshots = []
    for t in range(num_semesters):
        semester_data = {'time': t, 'layers': {}}
        for layer in ['coauthorship', 'supervision', 'project', 'course']:
            G = generate_collaboration_layer(num_nodes, roles, 0.05, t, layer)
            # Convert to edge list
            edges = np.array(list(G.edges()))
            semester_data['layers'][layer] = edges
        snapshots.append(semester_data)
    return snapshots, node_attrs

def save_data(snapshots, node_attrs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # Save node attributes
    node_attrs.to_csv(os.path.join(output_dir, 'node_attributes.csv'), index=False)
    # Save edge lists per semester per layer
    for t, sem in enumerate(snapshots):
        for layer, edges in sem['layers'].items():
            if len(edges) > 0:
                df = pd.DataFrame(edges, columns=['source', 'target'])
                df.to_csv(os.path.join(output_dir, f'edges_t{t:02d}_{layer}.csv'), index=False)
            else:
                # empty file
                pd.DataFrame(columns=['source', 'target']).to_csv(os.path.join(output_dir, f'edges_t{t:02d}_{layer}.csv'), index=False)
    # Save full snapshots as pickle
    with open(os.path.join(output_dir, 'snapshots.pkl'), 'wb') as f:
        pickle.dump(snapshots, f)
    # Also save aggregated statistics
    stats = []
    for t, sem in enumerate(snapshots):
        for layer, edges in sem['layers'].items():
            n_nodes = node_attrs.shape[0]
            n_edges = len(edges)
            density = 2*n_edges / (n_nodes*(n_nodes-1)) if n_nodes>1 else 0
            stats.append({'semester': t, 'layer': layer, 'num_edges': n_edges, 'density': density})
    pd.DataFrame(stats).to_csv(os.path.join(output_dir, 'layer_stats.csv'), index=False)
    print(f"Data saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='data/synthetic')
    parser.add_argument('--num_nodes', type=int, default=378)
    parser.add_argument('--num_semesters', type=int, default=12)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    snapshots, node_attrs = generate_multiplex_temporal(args.num_nodes, args.num_semesters, args.seed)
    save_data(snapshots, node_attrs, args.output)
    print("Synthetic multiplex temporal network generated successfully.")

if __name__ == '__main__':
    main()