import numpy as np
import skimage.morphology
from typing import List, Dict, Tuple
import math

def extract_candidate_skeletons(diagram_no_wire: np.ndarray, detections: List[Dict], crop_shape: Tuple[int, int]) -> List[np.ndarray]:
    skeletons = []
    h, w = crop_shape
    half_h, half_w = h // 2, w // 2
    
    for d in detections:
        cx, cy = d["x"], d["y"]
        y1 = max(0, cy - half_h)
        y2 = min(diagram_no_wire.shape[0], cy + half_h + (h % 2))
        x1 = max(0, cx - half_w)
        x2 = min(diagram_no_wire.shape[1], cx + half_w + (w % 2))
        
        crop = np.zeros(crop_shape, dtype=np.uint8)
        c_y1 = half_h - (cy - y1)
        c_y2 = half_h + (y2 - cy)
        c_x1 = half_w - (cx - x1)
        c_x2 = half_w + (x2 - cx)
        
        crop[c_y1:c_y2, c_x1:c_x2] = diagram_no_wire[y1:y2, x1:x2]
        
        binary_bool = crop > 0
        skeleton_bool = skimage.morphology.skeletonize(binary_bool)
        skeleton = (skeleton_bool * 255).astype(np.uint8)
        skeletons.append(skeleton)
        
    return skeletons

def build_skeleton_graph(skeleton_img: np.ndarray) -> Dict:
    """
    Scans the skeleton mask, computes 8-neighbor degrees, and identifies nodes.
    Returns a dictionary representing the raw pixel adjacency graph.
    """
    y_coords, x_coords = np.where(skeleton_img > 0)
    pixel_set = set(zip(y_coords, x_coords))
    
    nodes = {}
    for y, x in pixel_set:
        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if (ny, nx) in pixel_set:
                    neighbors.append((ny, nx))
        nodes[(y, x)] = neighbors
        
    endpoints = [n for n, neigh in nodes.items() if len(neigh) == 1]
    path_nodes = [n for n, neigh in nodes.items() if len(neigh) == 2]
    branchpoints = [n for n, neigh in nodes.items() if len(neigh) >= 3]
    
    return {
        "nodes": nodes,
        "endpoints": endpoints,
        "path_nodes": path_nodes,
        "branchpoints": branchpoints,
        "total_nodes": len(nodes),
        "total_edges": sum(len(neigh) for neigh in nodes.values()) // 2
    }

def simplify_skeleton_graph(raw_graph: Dict) -> Dict:
    """
    Compresses long degree-2 chains into single graph edges.
    Retains ONLY endpoints, branchpoints, junction nodes, and simplified edge paths.
    Topology descriptors must operate on these simplified symbolic graphs, NOT raw pixel adjacency graphs.
    """
    nodes = raw_graph["nodes"]
    
    # Keypoints are degree != 2 (endpoints, branchpoints, and isolated nodes)
    # A perfect circular loop has only degree-2 nodes. 
    # We must handle pure loops by arbitrarily picking one node as a keypoint if a component has no keypoints.
    keypoints = set(raw_graph["endpoints"] + raw_graph["branchpoints"])
    
    # Find connected components to handle pure loops
    visited_global = set()
    components = []
    
    for n in nodes:
        if n not in visited_global:
            comp = set()
            queue = [n]
            visited_global.add(n)
            while queue:
                curr = queue.pop(0)
                comp.add(curr)
                for neigh in nodes[curr]:
                    if neigh not in visited_global:
                        visited_global.add(neigh)
                        queue.append(neigh)
            components.append(comp)
            
    # Add one keypoint per pure loop component
    for comp in components:
        if not any(k in keypoints for k in comp):
            keypoints.add(next(iter(comp)))

    simplified_edges = []
    visited_edges = set() # Store as undirected (min(u,v), max(u,v))
    
    for start_node in keypoints:
        for next_node in nodes[start_node]:
            edge_id = tuple(sorted([start_node, next_node]))
            if edge_id in visited_edges:
                continue
                
            # Traverse path
            curr = next_node
            prev = start_node
            path_length = 1
            path_pixels = [start_node, curr]
            
            visited_edges.add(edge_id)
            
            while curr not in keypoints and curr in nodes:
                # Find the next neighbor that is not prev
                neighbors = nodes[curr]
                if len(neighbors) != 2:
                    break # Should not happen unless it's a keypoint
                next_step = neighbors[0] if neighbors[0] != prev else neighbors[1]
                
                edge_id = tuple(sorted([curr, next_step]))
                visited_edges.add(edge_id)
                
                prev = curr
                curr = next_step
                path_length += 1
                path_pixels.append(curr)
                
            simplified_edges.append({
                "u": start_node,
                "v": curr,
                "length": path_length,
                "path": path_pixels
            })
            
    return {
        "keypoints": list(keypoints),
        "endpoints": raw_graph["endpoints"],
        "branchpoints": raw_graph["branchpoints"],
        "edges": simplified_edges,
        "components": components
    }

def extract_template_topology(template_skeleton: np.ndarray) -> Dict:
    raw_graph = build_skeleton_graph(template_skeleton)
    simplified_graph = simplify_skeleton_graph(raw_graph)
    descriptor = compute_topology_descriptor(simplified_graph, raw_graph)
    return descriptor

def compute_topology_descriptor(simplified_graph: Dict, raw_graph: Dict) -> Dict:
    """
    Condenses the simplified graph into compact descriptors.
    """
    V = raw_graph["total_nodes"]
    E = raw_graph["total_edges"]
    C = len(simplified_graph["components"])
    
    # Cycle estimation is approximate and used ONLY as a soft structural descriptor, NOT as a hard symbolic invariant.
    estimated_cycle_count = E - V + C if V > 0 else 0
    
    endpoints = len(simplified_graph["endpoints"])
    branchpoints = len(simplified_graph["branchpoints"])
    
    edge_lengths = [e["length"] for e in simplified_graph["edges"]]
    mean_len = float(np.mean(edge_lengths)) if edge_lengths else 0.0
    std_len = float(np.std(edge_lengths)) if edge_lengths else 0.0
    max_len = float(np.max(edge_lengths)) if edge_lengths else 0.0
    
    largest_comp_ratio = 0.0
    if V > 0:
        max_comp_size = max([len(comp) for comp in simplified_graph["components"]], default=0)
        largest_comp_ratio = max_comp_size / V
        
    graph_density = E / V if V > 0 else 0.0
    
    # Node degree histogram
    deg_hist = {0:0, 1:0, 2:0, 3:0, 4:0}
    for n, neigh in raw_graph["nodes"].items():
        d = len(neigh)
        if d >= 4:
            deg_hist[4] += 1
        else:
            deg_hist[d] += 1
            
    return {
        "endpoint_count": endpoints,
        "branchpoint_count": branchpoints,
        "connected_component_count": C,
        "estimated_cycle_count": estimated_cycle_count,
        "edge_length_mean": mean_len,
        "edge_length_std": std_len,
        "edge_length_max": max_len,
        "graph_density": graph_density,
        "largest_component_ratio": largest_comp_ratio,
        "node_degree_histogram": deg_hist,
        "total_nodes": V
    }

def compute_topology_confidence(descriptor: Dict) -> float:
    """
    Evaluates the trustworthiness of the extracted graph using fragmentation level, 
    largest component ratio, skeleton sparsity, and disconnected component count.
    Fragmented or noisy candidate skeletons contribute less strongly to final reranking.
    """
    conf = 1.0
    
    # Penalize fragmentation
    C = descriptor["connected_component_count"]
    if C > 1:
        # e.g., 1 comp = 1.0, 2 comps = 0.8, 3 comps = 0.6
        conf *= max(0.2, 1.0 - 0.2 * (C - 1))
        
    # Penalize if largest component is small
    lcr = descriptor["largest_component_ratio"]
    if lcr < 0.8:
        conf *= max(0.2, lcr)
        
    # Penalize extreme sparsity
    if descriptor["total_nodes"] < 10:
        conf *= 0.5
        
    return conf

def _normalize_diff(val1: float, val2: float, norm_type="relative") -> float:
    """
    Topology similarity must compare structurally meaningful relative differences, NOT raw descriptor magnitudes.
    """
    if norm_type == "relative":
        denom = max(abs(val1), abs(val2))
        if denom == 0:
            return 0.0
        return abs(val1 - val2) / denom
    elif norm_type == "absolute":
        return abs(val1 - val2)
    return 0.0

def compute_topology_similarity(cand_desc: Dict, templ_desc: Dict) -> Dict:
    """
    Computes a soft bounded score [0, 1] based on normalized structural descriptor differences.
    """
    keys_to_compare = [
        "endpoint_count", 
        "branchpoint_count", 
        "connected_component_count", 
        "estimated_cycle_count",
        "edge_length_mean",
        "graph_density"
    ]
    
    contributions = {}
    total_penalty = 0.0
    
    for k in keys_to_compare:
        v_cand = float(cand_desc[k])
        v_templ = float(templ_desc[k])
        
        # Bounded relative difference [0, 1]
        diff = _normalize_diff(v_cand, v_templ, "relative")
        contributions[k] = diff
        total_penalty += diff
        
    # Find dominant mismatch (highest diff) and strongest alignment (lowest diff)
    dominant_mismatch = max(contributions, key=contributions.get) if contributions else None
    strongest_alignment = min(contributions, key=contributions.get) if contributions else None
    
    # Base similarity: e.g., exp(-2.0 * avg_penalty)
    avg_penalty = total_penalty / len(keys_to_compare) if keys_to_compare else 0.0
    similarity = math.exp(-2.0 * avg_penalty)
    
    # Determine failure attributions if similarity is low
    failure_attributions = []
    if similarity < 0.6:
        if contributions.get("endpoint_count", 0) > 0.4:
            failure_attributions.append("excessive endpoints" if cand_desc["endpoint_count"] > templ_desc["endpoint_count"] else "missing endpoints")
        if contributions.get("connected_component_count", 0) > 0.4:
            failure_attributions.append("fragmentation" if cand_desc["connected_component_count"] > templ_desc["connected_component_count"] else "fused micro-components")
        if contributions.get("branchpoint_count", 0) > 0.4:
            failure_attributions.append("abnormal branchpoint ratio")
        if contributions.get("estimated_cycle_count", 0) > 0.4:
            failure_attributions.append("cycle instability")
        if contributions.get("edge_length_mean", 0) > 0.4:
            failure_attributions.append("edge-length inconsistency")
            
    return {
        "similarity": similarity,
        "descriptor_contributions": contributions,
        "dominant_mismatch": dominant_mismatch,
        "strongest_alignment": strongest_alignment,
        "failure_attributions": failure_attributions
    }

def rerank_with_topology(detections: List[Dict], cand_descs: List[Dict], templ_desc: Dict, w_chamfer=0.60, w_pca=0.25, w_topo=0.15) -> List[Dict]:
    reranked = []
    
    for d, desc in zip(detections, cand_descs):
        sim_data = compute_topology_similarity(desc, templ_desc)
        conf = compute_topology_confidence(desc)
        
        effective_topology_score = conf * sim_data["similarity"]
        
        # Soft reranking: lightweight structural correction term
        chamfer_sim = d.get("chamfer_similarity", 0.0)
        pca_sim = d.get("pca_similarity", 0.0)
        
        # Prevent completely breaking if fields are missing, but they should be there
        final_score = w_chamfer * chamfer_sim + w_pca * pca_sim + w_topo * effective_topology_score
        
        d_copy = d.copy()
        d_copy["topology_similarity"] = sim_data["similarity"]
        d_copy["topology_confidence"] = conf
        d_copy["effective_topology_score"] = effective_topology_score
        d_copy["fused_score_topology"] = final_score
        
        d_copy["topology_descriptor"] = desc
        d_copy["descriptor_contributions"] = sim_data["descriptor_contributions"]
        d_copy["dominant_mismatch"] = sim_data["dominant_mismatch"]
        d_copy["strongest_alignment"] = sim_data["strongest_alignment"]
        d_copy["failure_attributions"] = sim_data["failure_attributions"]
        
        reranked.append(d_copy)
        
    reranked.sort(key=lambda x: x["fused_score_topology"], reverse=True)
    
    for r, d in enumerate(reranked, start=1):
        d["topology_rank"] = r
        
    return reranked
