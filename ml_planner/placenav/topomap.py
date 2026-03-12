from pathlib import Path

import torch
import yaml


def load_topomap(topomap_path, device):
    topomap_path = Path(topomap_path)
    with topomap_path.open('r', encoding='utf-8') as f:
        topomap = yaml.safe_load(f) or {}

    nodes = topomap.get('nodes', [])
    if not nodes:
        raise ValueError(f'No nodes found in topomap: {topomap_path}')

    node_ids = []
    actions = []
    features = []

    for node in nodes:
        feature = torch.tensor(node['feature'], device=device, dtype=torch.float32)
        if feature.numel() != 512:
            raise ValueError(f'Feature dimension must be 512: node_id={node.get("id")}')

        edges = node.get('edges', [])
        if not edges:
            raise ValueError(f'Node must have at least one edge: node_id={node.get("id")}')

        node_ids.append(int(node['id']))
        actions.append(edges[0]['action'])
        features.append(feature)

    feature_matrix = torch.stack(features, dim=0)
    return {
        'node_ids': node_ids,
        'actions': actions,
        'feature_matrix': feature_matrix,
    }
