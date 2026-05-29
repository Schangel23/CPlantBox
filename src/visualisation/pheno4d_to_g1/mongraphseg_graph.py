"""Faithful MonGraphSeg graph-segmentation pipeline (Tobies et al., 2025).

This is a clean re-implementation of the MonGraphSeg MATLAB pipeline
(Engineering-Geodesy-Bonn/MonGraphSeg), phases 2 + 3a-3f + 4, for unbranched
monocots (maize/sorghum). It replaces the earlier divergent Python translation
in ``skeletonizer.py`` + ``graph_refinement.py``, which had two foundational
breaks that made it collapse the whole plant to one path:

  1. ``laplacian_contraction_skeleton`` ended with an MST-diameter extraction
     (``_order_points_mst``) that LINEARISED the plant — every leaf branch was
     thrown away, so the graph had no junctions to segment.
  2. ``build_skeleton_graph`` did a bare kNN(k=3) with no ``rmTriangles``, so on
     the dense medial axis every node became a junction (a blob, not a tree).

Pipeline (MATLAB phase in parentheses):
  contract_point_cloud         (2  Laplacian contraction, Cao/Au)
  farthest_point_resample      (2  branch-preserving resample)
  build_initial_graph          (3a kNN k=3 + rmTriangles + connect components)
  collapse_skeleton_tree       (3a MST + junction/super-edge tracing)
  prune_short_branches         (3f removeSpuriousLeafBranches)
  remove_ground_nodes          (3b verticality start node)
  extract_stem_path            (3e fewest-branching / unbranching metric)
  segment_leaves               (3e leaf instances = branches off the stem)
  segment_point_cloud          (4  nearest panoptic instance — reused from
                                   graph_refinement)

The single driver is :func:`segment_plant_graph`, which returns the same
``organs`` dict shape as the rest of the package.
"""

import collections

import numpy as np
import networkx as nx
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import spsolve
from scipy.sparse.csgraph import dijkstra, minimum_spanning_tree

try:
    from .geodesic_assignment import build_knn_graph
except ImportError:  # direct script import via sys.path
    from geodesic_assignment import build_knn_graph


# ── Phase 2: Laplacian contraction ────────────────────────────────────────


def contract_point_cloud(points, k=10, iterations=8, position_weight=1.0,
                         laplacian_scale=2.0, max_solve_points=2500, seed=0):
    """Contract a point cloud toward its medial axis (Au et al. 2008).

    Solves ``(sl·LᵀL + wh·I) x = wh·p`` repeatedly with the Laplacian weight
    ``sl`` growing each iteration (stronger contraction) while the position
    weight ``wh`` anchors to the original points. Unlike the old port this does
    NOT linearise afterwards — the contracted points keep the branched
    topology, ready for branch-preserving resampling.

    Large clouds are randomly subsampled to ``max_solve_points`` for a
    tractable sparse solve.

    Returns:
        contracted: (M, 3) contracted positions.
        used: (M, 3) the (possibly subsampled) original points, aligned to
            ``contracted`` row-for-row.
    """
    points = np.asarray(points, float)
    n = len(points)
    if n < 4:
        return points.copy(), points.copy()

    if n > max_solve_points:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, max_solve_points, replace=False))
        points = points[idx]
        n = max_solve_points

    k = min(k, n - 1)
    tree = KDTree(points)
    _, indices = tree.query(points, k=k + 1)
    rows = np.repeat(np.arange(n), k)
    cols = indices[:, 1:].ravel()
    adj = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    adj = adj.maximum(adj.T)

    degree = np.asarray(adj.sum(axis=1)).ravel()
    degree[degree == 0] = 1.0
    L = diags(degree) - adj
    LtL = (L.T @ L).tocsc()
    I_n = eye(n, format="csc")

    contracted = points.astype(float).copy()
    sl = 1.0
    for _ in range(iterations):
        sys_mat = (sl * LtL + position_weight * I_n).tocsc()
        contracted = np.column_stack([
            spsolve(sys_mat, position_weight * points[:, d]) for d in range(3)
        ])
        sl *= laplacian_scale

    return contracted, points


def farthest_point_resample(pts, n_samples, start="top"):
    """Greedy farthest-point sampling — branch-preserving (MATLAB 2/FPS).

    Returns indices into ``pts``. ``start='top'`` seeds at the highest point
    (apex) so the sampling is deterministic and the apex is always a node.
    """
    pts = np.asarray(pts, float)
    n = len(pts)
    n_samples = min(n_samples, n)
    sel = np.empty(n_samples, dtype=int)
    sel[0] = int(np.argmax(pts[:, 2])) if start == "top" else 0
    min_d2 = np.sum((pts - pts[sel[0]]) ** 2, axis=1)
    for i in range(1, n_samples):
        sel[i] = int(np.argmax(min_d2))
        min_d2 = np.minimum(min_d2, np.sum((pts - pts[sel[i]]) ** 2, axis=1))
    return np.unique(sel)


# ── Phase 3a: initial graph (kNN + rmTriangles + connect) ─────────────────


def _rm_triangles(G):
    """Remove the longest edge of every triangle (MATLAB rmTriangles.m).

    Turns the over-connected kNN graph on dense medial nodes into a near-tree
    while preserving branch topology.
    """
    changed = True
    while changed:
        changed = False
        for u in list(G.nodes()):
            nbrs = list(G.neighbors(u))
            done = False
            for a in range(len(nbrs)):
                for b in range(a + 1, len(nbrs)):
                    x, y = nbrs[a], nbrs[b]
                    if G.has_edge(x, y):
                        tri = [(u, x), (u, y), (x, y)]
                        longest = max(tri, key=lambda e: G.edges[e]["weight"])
                        G.remove_edge(*longest)
                        changed = done = True
                        break
                if done:
                    break
    return G


def _connect_components(G, positions):
    """Greedily connect disconnected components by the closest node pair."""
    while nx.number_connected_components(G) > 1:
        comps = list(nx.connected_components(G))
        comps.sort(key=len, reverse=True)
        main = comps[0]
        main_arr = np.array([positions[n] for n in main])
        main_ids = list(main)
        best = None
        for comp in comps[1:]:
            for nid in comp:
                d = np.linalg.norm(main_arr - positions[nid], axis=1)
                j = int(np.argmin(d))
                if best is None or d[j] < best[0]:
                    best = (d[j], main_ids[j], nid)
        if best is None:
            break
        G.add_edge(best[1], best[2], weight=float(best[0]))
    return G


def build_initial_graph(nodes, k=3):
    """kNN(k) graph on skeleton nodes, de-triangulated and connected (3a)."""
    nodes = np.asarray(nodes, float)
    n = len(nodes)
    G = nx.Graph()
    positions = {i: nodes[i] for i in range(n)}
    for i in range(n):
        G.add_node(i, pos=nodes[i])
    tree = KDTree(nodes)
    _, idx = tree.query(nodes, k=min(k + 1, n))
    for i in range(n):
        for j in idx[i, 1:]:
            j = int(j)
            if i != j and not G.has_edge(i, j):
                G.add_edge(i, j, weight=float(np.linalg.norm(nodes[i] - nodes[j])))
    _rm_triangles(G)
    _connect_components(G, positions)
    return G


# ── Phase 3a (cont.): MST tree + short-spur pruning ───────────────────────


def collapse_skeleton_tree(G):
    """Reduce the graph to its MST — a cycle-free skeleton tree.

    The MST over Euclidean edge weights follows the medial branches and breaks
    the residual micro-cycles left after ``rmTriangles``. This stands in for
    MATLAB 3c/3d (overlapping-leaf split + leaf-stem cycle removal): both exist
    only to turn the graph into a tree, which the MST does directly.
    """
    nodes = list(G.nodes())
    index = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    rows, cols, w = [], [], []
    for u, v, d in G.edges(data=True):
        rows.append(index[u]); cols.append(index[v]); w.append(d["weight"])
    if not rows:
        return G.copy()
    M = csr_matrix((w, (rows, cols)), shape=(n, n))
    mst = minimum_spanning_tree(M)
    mst = mst + mst.T
    T = nx.Graph()
    for nid in nodes:
        T.add_node(nid, pos=G.nodes[nid]["pos"])
    mst_coo = mst.tocoo()
    for i, j, ww in zip(mst_coo.row, mst_coo.col, mst_coo.data):
        if i < j:
            T.add_edge(nodes[i], nodes[j], weight=float(ww))
    return T


def _angle_deg(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    c = np.clip(float(a @ b) / (na * nb), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def recover_leaf_branches(skeleton_graph, dense_graph, points,
                          min_angle_deg=35.0, min_branch_len_cm=3.0,
                          max_paths_per_junction=2):
    """Recover terminal dense-graph branches that a local MST choice hid.

    The collapsed skeleton is intentionally a tree, but MST tie-breaking around
    pseudostem junctions can route two nearby leaf traces through one branch.
    For each tree junction, inspect dense pre-MST edges that touch the junction
    but are absent from the tree; if the edge leads to a dense terminal through
    a direction distinct from existing tree branches, graft a duplicate terminal
    path onto the tree. Duplicating the recovered path keeps the output acyclic
    while preserving a separate terminal for pseudostem decomposition.
    """
    T = skeleton_graph.copy()
    if T.number_of_nodes() == 0 or dense_graph.number_of_edges() == 0:
        return T

    dense_pos = nx.get_node_attributes(dense_graph, "pos")
    next_id = (max(T.nodes()) + 1) if T.number_of_nodes() else 0
    point_tree = KDTree(np.asarray(points, float)) if len(points) else None

    def path_len(path):
        return sum(
            float(np.linalg.norm(dense_pos[path[i + 1]] - dense_pos[path[i]]))
            for i in range(len(path) - 1)
        )

    def support_count(path):
        if point_tree is None or len(path) < 2:
            return 0
        samples = []
        for i in range(len(path) - 1):
            a = dense_pos[path[i]]
            b = dense_pos[path[i + 1]]
            seg_len = float(np.linalg.norm(b - a))
            n = max(2, int(np.ceil(seg_len / 1.0)))
            t = np.linspace(0.0, 1.0, n)
            samples.append(a + np.outer(t, b - a))
        samples = np.vstack(samples)
        hits = point_tree.query_ball_point(samples, r=2.0)
        return len({idx for group in hits for idx in group})

    def terminal_path_from(junction, first):
        blocked = {junction}
        queue = collections.deque([(first, [junction, first])])
        seen = {junction, first}
        best = None
        while queue:
            cur, path = queue.popleft()
            if dense_graph.degree(cur) == 1:
                best = path
                break
            nbrs = [n for n in dense_graph.neighbors(cur) if n not in blocked]
            for nbr in nbrs:
                if nbr in seen:
                    continue
                # Stay on a branch-like trace; stop expanding through another
                # dense junction unless it is the only route to a terminal.
                if cur != first and dense_graph.degree(cur) >= 3:
                    continue
                seen.add(nbr)
                queue.append((nbr, path + [nbr]))
        return best

    for junction in [n for n in list(T.nodes()) if T.degree(n) >= 3]:
        if junction not in dense_graph:
            continue
        jpos = dense_pos[junction]
        existing_dirs = [
            T.nodes[nbr]["pos"] - jpos
            for nbr in T.neighbors(junction)
            if np.linalg.norm(T.nodes[nbr]["pos"] - jpos) > 1e-9
        ]
        recovered_dirs = []
        candidates = []
        for nbr in dense_graph.neighbors(junction):
            if T.has_edge(junction, nbr):
                continue
            path = terminal_path_from(junction, nbr)
            if path is None or len(path) < 2:
                continue
            length = path_len(path)
            if length < min_branch_len_cm:
                continue
            direction = dense_pos[path[-1]] - jpos
            if any(_angle_deg(direction, d) < min_angle_deg for d in existing_dirs):
                continue
            support = support_count(path)
            if support < 8:
                continue
            candidates.append((support, length, path, direction))

        candidates.sort(reverse=True, key=lambda x: (x[0], x[1]))
        added = 0
        for _, _, path, direction in candidates:
            if added >= max_paths_per_junction:
                break
            if any(_angle_deg(direction, d) < min_angle_deg for d in recovered_dirs):
                continue
            prev = junction
            for old in path[1:]:
                new = next_id
                next_id += 1
                pos = dense_pos[old].copy()
                T.add_node(new, pos=pos)
                w = float(np.linalg.norm(T.nodes[prev]["pos"] - pos))
                T.add_edge(prev, new, weight=w)
                prev = new
            recovered_dirs.append(direction)
            existing_dirs.append(direction)
            added += 1
    return T


def _branch_length(T, path):
    return sum(T.edges[path[i], path[i + 1]]["weight"]
               for i in range(len(path) - 1))


def prune_short_branches(T, min_branch_len_cm=4.0, max_iter=100):
    """Iteratively drop terminal branches shorter than ``min_branch_len_cm``.

    A "branch" is the path from a terminal (degree 1) back to the nearest
    junction (degree ≥ 3). Short terminal spurs are medial-axis noise (MATLAB
    3f removeSpuriousLeafBranches); real leaf branches are long and survive.
    Degree-2 chains are left intact (they are the interior of branches).
    """
    for _ in range(max_iter):
        terminals = [n for n in T.nodes() if T.degree(n) == 1]
        removed = False
        for t in terminals:
            # walk from terminal until we hit a junction (deg>=3) or another terminal
            path = [t]
            prev = None
            cur = t
            while True:
                nbrs = [x for x in T.neighbors(cur) if x != prev]
                if len(nbrs) != 1:
                    break
                prev, cur = cur, nbrs[0]
                path.append(cur)
                if T.degree(cur) != 2:
                    break
            # path[-1] is the junction (or terminal of a bare chain)
            if T.degree(path[-1]) >= 3 and _branch_length(T, path) < min_branch_len_cm:
                T.remove_nodes_from(path[:-1])  # keep the junction
                removed = True
        if not removed:
            break
    # drop any now-isolated nodes
    T.remove_nodes_from([n for n in list(T.nodes()) if T.degree(n) == 0])
    return T


# ── Phase 3b: ground removal (verticality) ────────────────────────────────


def prune_branches_by_support(T, points, min_support_frac=0.02, max_iter=100):
    """Drop terminal branches with little cloud SUPPORT, not just short ones.

    Length-based pruning (:func:`prune_short_branches`) cannot tell a short but
    REAL leaf (small / occluded / partially scanned, yet backed by many cloud
    points) from a medial-axis spur (a noise filament backed by ~no points).
    After contraction + coarse resampling, real leaves are often short, so the
    length rule deletes them and the segmenter under-segments.

    Here each cloud point votes for its nearest skeleton node; a terminal
    branch's support is the point count over its nodes (excluding the junction),
    and a branch is pruned when that support is below ``min_support_frac`` of all
    points. Measured against the synthetic judge this trades count for quality:
    higher ``min_support_frac`` raises per-leaf IoU + point accuracy but
    under-segments more. NOTE: it does NOT lift leaf recall@IoU>=.5 past the
    ~39% ceiling of the contract->MST->prune skeleton -- most leaves are lost at
    the skeleton level (merged / absorbed) before pruning. A learned
    point-instance segmenter (trained on the labelled synthetic set) is the way
    past that ceiling; this remains the best heuristic option.
    """
    import collections
    thresh = max(3, int(min_support_frac * len(points)))
    for _ in range(max_iter):
        node_ids = list(T.nodes())
        if len(node_ids) < 3:
            break
        pos = np.array([T.nodes[n]["pos"] for n in node_ids])
        _, nn = KDTree(pos).query(points)
        votes = collections.Counter(node_ids[i] for i in nn)
        removed = False
        for t in [n for n in T.nodes() if T.degree(n) == 1]:
            path, prev, cur = [t], None, t
            while True:
                nbrs = [x for x in T.neighbors(cur) if x != prev]
                if len(nbrs) != 1:
                    break
                prev, cur = cur, nbrs[0]
                path.append(cur)
                if T.degree(cur) != 2:
                    break
            if T.degree(path[-1]) < 3:
                continue
            support = sum(votes.get(n, 0) for n in path[:-1])
            # prune on support alone: a real leaf is well-supported however short
            # its (contracted) skeleton is; a spur is a filament with ~no points.
            if support < thresh:
                T.remove_nodes_from(path[:-1])
                removed = True
        if not removed:
            break
    T.remove_nodes_from([n for n in list(T.nodes()) if T.degree(n) == 0])
    return T


def remove_ground_nodes(T, angle_threshold_deg=50):
    """Pick the plant base by verticality, drop nodes below it (MATLAB 3b)."""
    if T.number_of_nodes() == 0:
        return T.copy(), None
    positions = nx.get_node_attributes(T, "pos")
    highest = max(positions, key=lambda n: positions[n][2])
    thr = np.radians(angle_threshold_deg)

    candidates = []
    for node in T.nodes():
        if node == highest:
            continue
        try:
            path = nx.shortest_path(T, node, highest, weight="weight")
        except nx.NetworkXNoPath:
            continue
        # MATLAB detectFirstBranch: angle is measured from node i to each
        # DOWNSTREAM path node (cumulative cone), not per-edge — a small
        # horizontal wiggle must not disqualify an otherwise-vertical base.
        p0 = positions[node]
        ok = True
        for p in path[1:]:
            d = positions[p] - p0
            L = np.linalg.norm(d)
            if L < 1e-9:
                continue
            if np.arccos(np.clip(d[2] / L, -1, 1)) >= thr:
                ok = False
                break
        if ok:
            candidates.append(node)

    if candidates:
        start = min(candidates, key=lambda n: positions[n][2])
    else:
        start = min(positions, key=lambda n: positions[n][2])

    start_z = positions[start][2]
    T2 = T.copy()
    T2.remove_nodes_from([n for n in T.nodes() if positions[n][2] < start_z])
    if start not in T2:
        if T2.number_of_nodes() == 0:
            return T2, None
        start = min(T2.nodes(), key=lambda n: positions[n][2])
    # keep the component with the largest vertical span (where the stem lives)
    comps = list(nx.connected_components(T2))
    if len(comps) > 1:
        def zspan(c):
            zs = [positions[n][2] for n in c]
            return max(zs) - min(zs)
        main = max(comps, key=zspan)
        T2 = T2.subgraph(main).copy()
        if start not in T2:
            start = min(T2.nodes(), key=lambda n: positions[n][2])
    return T2, start


# ── Phase 3e: stem path (fewest-branching / unbranching metric) ────────────


def extract_stem_path(T, start):
    """Stem = path from base to the terminal that leaves the fewest branches.

    MATLAB extractStemPath scores each base→terminal candidate by the total
    length of side-branches hanging off it (the "unbranching" metric) and
    picks the minimum, tie-broken by the longest path. A faithful equivalent:
    for each candidate path, sum the length of the subtrees that hang off its
    interior nodes; the true stem has the least hanging mass relative to its
    length (leaves branch off it, not the reverse).
    """
    if start is None or start not in T:
        return []
    terminals = [n for n in T.nodes() if T.degree(n) == 1 and n != start]
    if not terminals:
        return [start]

    best = None
    for term in terminals:
        try:
            path = nx.shortest_path(T, start, term, weight="weight")
        except nx.NetworkXNoPath:
            continue
        path_set = set(path)
        path_len = _branch_length(T, path)
        # mass hanging off the path = everything not reachable along the path
        Tcut = T.copy()
        for i in range(len(path) - 1):
            Tcut.remove_edge(path[i], path[i + 1])
        hanging = 0.0
        for comp in nx.connected_components(Tcut):
            on_path = comp & path_set
            if on_path and len(comp) > len(on_path):
                sub = Tcut.subgraph(comp)
                hanging += sum(d["weight"] for _, _, d in sub.edges(data=True))
        # score: prefer low hanging mass, then long path
        score = (hanging, -path_len)
        if best is None or score < best[0]:
            best = (score, path)
    return best[1] if best else [start]


def segment_leaves(T, stem_path, min_leaf_len_cm=4.0):
    """Leaf instances = branches that hang off the stem path (MATLAB 3e)."""
    stem_set = set(stem_path)
    G = T.copy()
    for i in range(len(stem_path) - 1):
        if G.has_edge(stem_path[i], stem_path[i + 1]):
            G.remove_edge(stem_path[i], stem_path[i + 1])
    leaves = {}
    lid = 1
    for comp in nx.connected_components(G):
        non_stem = comp - stem_set
        if not non_stem:
            continue
        junctions = comp & stem_set
        root = next(iter(junctions)) if junctions else None
        if root is None:
            # attach to nearest stem node
            positions = nx.get_node_attributes(T, "pos")
            stem_arr = np.array([positions[n] for n in stem_path])
            root = min(comp, key=lambda n: np.min(
                np.linalg.norm(stem_arr - positions[n], axis=1)))
        sub = G.subgraph(comp)
        # longest path from the stem junction = the leaf midrib
        far = max(comp, key=lambda n: nx.shortest_path_length(
            sub, root, n, weight="weight") if nx.has_path(sub, root, n) else -1)
        try:
            path = nx.shortest_path(sub, root, far, weight="weight")
        except nx.NetworkXNoPath:
            continue
        if _branch_length(T, path) >= min_leaf_len_cm:
            leaves[lid] = path
            lid += 1
    return leaves


# ── Pseudostem model (no visible stem — young maize) ──────────────────────


def extract_pseudostem_and_leaves(T, start, min_leaf_solo_len_cm=3.0):
    """Decompose a young-maize tree with NO visible stem.

    In these field scans the tall vertical structure is the longest/youngest
    *leaf*, not a stem — the only stem-like part is the basal pseudostem where
    leaf sheaths wrap each other. So instead of electing a privileged stem
    path, every leaf tip is traced down to the base and each skeleton edge is
    classified by how many tip→base paths traverse it:

      * edges on ≥2 paths  -> pseudostem (the shared wrapped-sheath bundle);
      * edges on exactly 1 -> that leaf's emerged blade.

    Each leaf is rooted at its *insertion* (the highest node it still shares
    with another leaf), so the leaf polyline is insertion → tip = sheath-exit
    + blade. The tall erect leaf becomes an ordinary leaf.

    Returns:
        pseudostem_edges: list of (nodeA, nodeB) shared edges.
        leaf_dict: ``{leaf_id: [node_ids]}`` insertion→tip per leaf.
    """
    if start is None or start not in T:
        return [], {}
    terminals = [n for n in T.nodes() if T.degree(n) == 1 and n != start]
    if not terminals:
        return [], {}

    paths = {}
    node_use = collections.Counter()
    edge_use = collections.Counter()
    for t in terminals:
        try:
            p = nx.shortest_path(T, start, t, weight="weight")
        except nx.NetworkXNoPath:
            continue
        paths[t] = p
        for n in p:
            node_use[n] += 1
        for i in range(len(p) - 1):
            edge_use[frozenset((p[i], p[i + 1]))] += 1

    pseudostem_edges = [tuple(e) for e, c in edge_use.items() if c >= 2]

    leaf_dict = {}
    lid = 1
    for t in sorted(paths, key=lambda n: T.nodes[n]["pos"][2]):
        p = paths[t]
        # Prefer the lowest shared node where this tip path turns out of the
        # bundle; fall back to the original highest-shared rule when the path
        # has no clear angular divergence.
        shared_idx = [i for i, n in enumerate(p) if node_use[n] >= 2]
        i0 = max(shared_idx) if shared_idx else 0
        divergence_idx = None
        for i in shared_idx:
            if i <= 0 or i >= len(p) - 1:
                continue
            back = max(0, i - 2)
            fwd = min(len(p) - 1, i + 2)
            vin = T.nodes[p[i]]["pos"] - T.nodes[p[back]]["pos"]
            vout = T.nodes[p[fwd]]["pos"] - T.nodes[p[i]]["pos"]
            lin = float(np.linalg.norm(vin))
            lout = float(np.linalg.norm(vout))
            if lin < 1e-9 or lout < 1e-9:
                continue
            angle = np.degrees(np.arccos(np.clip(
                float(vin @ vout) / (lin * lout), -1.0, 1.0)))
            radial = float(np.linalg.norm(vout[:2]))
            if angle >= 25.0 and radial / max(lout, 1e-9) >= 0.25:
                divergence_idx = i
                break
        if divergence_idx is not None:
            i0 = divergence_idx
        leaf_nodes = p[i0:]              # insertion node + emerged blade
        if len(leaf_nodes) < 2:
            continue
        if _branch_length(T, leaf_nodes) < min_leaf_solo_len_cm:
            continue
        leaf_dict[lid] = leaf_nodes
        lid += 1
    return pseudostem_edges, leaf_dict


def assign_points_to_instances(points, node_positions, instances):
    """Nearest panoptic-instance assignment over explicit edge sets (MATLAB 4).

    ``instances`` is a list of ``(label, edges)`` where ``edges`` is a list of
    ``(nodeA, nodeB)`` id pairs. Each point gets the label of the instance with
    the smallest point-to-segment distance over that instance's edges. Label 0
    is reserved for "unassigned" but every point is reachable here.
    """
    points = np.asarray(points, float)
    n = len(points)
    best_d = np.full(n, np.inf)
    labels = np.zeros(n, dtype=int)
    for label, edges in instances:
        for a_id, b_id in edges:
            a = node_positions[a_id]
            b = node_positions[b_id]
            ab = b - a
            L2 = float(ab @ ab)
            if L2 < 1e-12:
                d = np.linalg.norm(points - a, axis=1)
            else:
                t = np.clip((points - a) @ ab / L2, 0.0, 1.0)
                d = np.linalg.norm(points - (a + np.outer(t, ab)), axis=1)
            upd = d < best_d
            best_d[upd] = d[upd]
            labels[upd] = label
    return labels


def assign_points_geodesic_instances(points, node_positions, leaf_dict,
                                     pseudostem_edges, k=10,
                                     max_edge_cm=3.0, distal_seed_start=0.5,
                                     distal_seed_end=1.0,
                                     fallback_labels=None,
                                     return_debug=False):
    """Assign cloud points by multi-source geodesic distance on a kNN graph.

    Label 0 is the pseudostem. Leaf labels match ``leaf_dict`` ids. Leaf seeds
    are cloud points whose nearest skeleton node lies on the distal portion of
    each insertion-to-tip leaf path; pseudostem seeds are points nearest to any
    skeleton node used by a pseudostem edge.
    """
    points = np.asarray(points, float)
    n = len(points)
    labels = np.zeros(n, dtype=int)
    if n == 0:
        return (labels, {}) if return_debug else labels

    node_ids = list(node_positions)
    if not node_ids:
        return (labels, {}) if return_debug else labels
    node_arr = np.array([node_positions[nid] for nid in node_ids], dtype=float)
    nearest_node_idx = KDTree(node_arr).query(points)[1]
    nearest_node = np.array([node_ids[int(i)] for i in nearest_node_idx])

    ps_nodes = set()
    for a_id, b_id in pseudostem_edges:
        ps_nodes.add(a_id)
        ps_nodes.add(b_id)

    seed_by_label = {0: np.where(np.isin(nearest_node, list(ps_nodes)))[0]}

    for lid, path in sorted(leaf_dict.items()):
        if len(path) < 2:
            continue
        seg_len = np.array([
            np.linalg.norm(node_positions[path[i + 1]] - node_positions[path[i]])
            for i in range(len(path) - 1)
        ])
        total = float(seg_len.sum())
        if total <= 1e-12:
            distal_nodes = {path[-1]}
        else:
            cum = np.concatenate([[0.0], np.cumsum(seg_len)]) / total
            lo = min(distal_seed_start, distal_seed_end)
            hi = max(distal_seed_start, distal_seed_end)
            distal_nodes = {
                node for node, frac in zip(path, cum)
                if lo <= frac <= hi
            }
            if not distal_nodes:
                distal_nodes = {path[-1]}
        seed_by_label[lid] = np.where(np.isin(nearest_node, list(distal_nodes)))[0]

    active = [(lab, seeds) for lab, seeds in seed_by_label.items() if len(seeds)]
    if not active:
        if fallback_labels is not None:
            labels = np.asarray(fallback_labels, dtype=int).copy()
        return (labels, {"seed_by_label": seed_by_label}) if return_debug else labels

    graph = build_knn_graph(points, k=k, max_edge_cm=max_edge_cm)

    # Add one zero-cost super-source per label, then run multi-source Dijkstra.
    aug = graph.tolil()
    aug.resize((n + len(active), n + len(active)))
    source_labels = []
    for src_offset, (lab, seeds) in enumerate(active):
        src = n + src_offset
        source_labels.append(lab)
        aug[src, np.asarray(seeds, dtype=int)] = 1e-9
    aug = aug.tocsr()
    aug = aug.maximum(aug.T)

    dmat = dijkstra(
        aug, directed=False, indices=np.arange(n, n + len(active))
    )[:, :n]
    reachable = np.isfinite(dmat).any(axis=0)
    best = np.argmin(dmat[:, reachable], axis=0)
    labels[reachable] = np.array(source_labels, dtype=int)[best]

    if fallback_labels is not None and np.any(~reachable):
        labels[~reachable] = np.asarray(fallback_labels, dtype=int)[~reachable]

    if return_debug:
        debug = {
            "knn_graph": graph,
            "seed_by_label": seed_by_label,
            "source_labels": source_labels,
            "geodesic_dist": dmat,
            "nearest_node": nearest_node,
        }
        return labels, debug
    return labels


def segment_plant_pseudostem(points, n_skel_nodes=400, min_leaf_len_cm=3.0,
                             angle_threshold_deg=50, prune="length",
                             support_frac=0.02, recover_branches=False,
                             recovery_angle_deg=35.0, contraction_iters=8,
                             assign="segment", geodesic_k=10,
                             geodesic_max_edge_cm=3.0,
                             distal_seed_start=0.5,
                             distal_seed_end=1.0,
                             return_debug=False):
    """No-stem segmentation for young maize: pseudostem bundle + leaves.

    Same skeleton/tree front-end as :func:`segment_plant_graph`, but replaces
    stem-path election with :func:`extract_pseudostem_and_leaves`. Returns an
    ``organs`` dict with a ``pseudostem`` key plus ``leaf_1..N`` (the tall erect
    leaf included).

    ``prune`` selects spur removal: ``"length"`` (drop branches shorter than
    ``min_leaf_len_cm`` -- deletes short real leaves) or ``"support"``
    (:func:`prune_branches_by_support` -- keeps short leaves that have cloud
    support, drops only unsupported tiny spurs).
    """
    points = np.asarray(points, float)
    contracted = contract_point_cloud(points, iterations=contraction_iters)[0]
    sel = farthest_point_resample(contracted, n_skel_nodes)
    nodes = contracted[sel]

    G = build_initial_graph(nodes, k=3)
    T = collapse_skeleton_tree(G)
    if recover_branches:
        T = recover_leaf_branches(
            T, G, points, min_angle_deg=recovery_angle_deg,
            min_branch_len_cm=min_leaf_len_cm)
    if prune == "support":
        T = prune_branches_by_support(T, points, min_support_frac=support_frac)
    else:
        T = prune_short_branches(T, min_branch_len_cm=min_leaf_len_cm)
    T, start = remove_ground_nodes(T, angle_threshold_deg=angle_threshold_deg)
    # support-pruning already removed spurs; don't re-kill short real leaves here
    solo_len = 1.0 if prune == "support" else min_leaf_len_cm
    ps_edges, leaf_dict = extract_pseudostem_and_leaves(
        T, start, min_leaf_solo_len_cm=solo_len)

    node_positions = nx.get_node_attributes(T, "pos")
    instances = [(1, ps_edges)]
    for lid, path in leaf_dict.items():
        edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        instances.append((lid + 1, edges))
    labels_segment = assign_points_to_instances(points, node_positions, instances)
    if assign == "segment":
        labels = labels_segment
        label_for_pseudostem = 1
        label_for_leaf = lambda lid: lid + 1
        geodesic_debug = None
    elif assign == "geodesic":
        labels, geodesic_debug = assign_points_geodesic_instances(
            points, node_positions, leaf_dict, ps_edges,
            k=geodesic_k, max_edge_cm=geodesic_max_edge_cm,
            distal_seed_start=distal_seed_start,
            distal_seed_end=distal_seed_end,
            fallback_labels=np.where(labels_segment == 1, 0,
                                     np.maximum(labels_segment - 1, 0)),
            return_debug=True)
        label_for_pseudostem = 0
        label_for_leaf = lambda lid: lid
    else:
        raise ValueError("assign must be 'segment' or 'geodesic'")

    organs = {"pseudostem": points[labels == label_for_pseudostem]}
    out_id = 0
    for lid in sorted(leaf_dict):
        pts = points[labels == label_for_leaf(lid)]
        if len(pts) == 0:
            continue
        out_id += 1
        organs[f"leaf_{out_id}"] = pts

    if not return_debug:
        return organs
    debug = {"tree": T, "start": start, "pseudostem_edges": ps_edges,
             "leaf_dict": leaf_dict, "labels": labels,
             "node_positions": node_positions, "initial_graph": G,
             "assign": assign}
    if geodesic_debug is not None:
        debug["geodesic"] = geodesic_debug
    return organs, debug


# ── Phase 4: nearest-instance point assignment ────────────────────────────


def segment_point_cloud(points, node_positions, stem_path, leaf_dict):
    """Assign each point to the nearest panoptic instance polyline (MATLAB 4).

    ``node_positions`` is a ``{node_id: (3,) position}`` map covering all nodes
    referenced by ``stem_path`` / ``leaf_dict``. Assignment is point-to-segment
    (edge) distance, matching SegmentationPointCloud.m, not just nearest node.
    """
    points = np.asarray(points, float)
    n = len(points)
    if n == 0:
        return np.zeros(0, dtype=int)

    instances = [("stem", 1, stem_path)]
    for lid, path in leaf_dict.items():
        instances.append((f"leaf_{lid}", lid + 1, path))

    best_d = np.full(n, np.inf)
    labels = np.zeros(n, dtype=int)
    for _, label, path in instances:
        if len(path) < 2:
            if len(path) == 1:
                d = np.linalg.norm(points - node_positions[path[0]], axis=1)
                upd = d < best_d
                best_d[upd] = d[upd]; labels[upd] = label
            continue
        for i in range(len(path) - 1):
            a = node_positions[path[i]]
            b = node_positions[path[i + 1]]
            ab = b - a
            L2 = float(ab @ ab)
            if L2 < 1e-12:
                d = np.linalg.norm(points - a, axis=1)
            else:
                t = np.clip((points - a) @ ab / L2, 0.0, 1.0)
                proj = a + np.outer(t, ab)
                d = np.linalg.norm(points - proj, axis=1)
            upd = d < best_d
            best_d[upd] = d[upd]; labels[upd] = label
    return labels


# ── Driver ────────────────────────────────────────────────────────────────


def segment_plant_graph(points, n_skel_nodes=250, min_leaf_len_cm=4.0,
                        angle_threshold_deg=50, return_debug=False):
    """Full MonGraphSeg graph segmentation of a maize point cloud.

    Returns an ``organs`` dict (``{"stem": (N,3), "leaf_1": ...}``); with
    ``return_debug`` also returns the skeleton tree, stem path and leaf paths.
    """
    points = np.asarray(points, float)
    contracted = contract_point_cloud(points)[0]
    sel = farthest_point_resample(contracted, n_skel_nodes)
    nodes = contracted[sel]

    G = build_initial_graph(nodes, k=3)
    T = collapse_skeleton_tree(G)
    T = prune_short_branches(T, min_branch_len_cm=min_leaf_len_cm)
    T, start = remove_ground_nodes(T, angle_threshold_deg=angle_threshold_deg)
    stem_path = extract_stem_path(T, start)
    leaf_dict = segment_leaves(T, stem_path, min_leaf_len_cm=min_leaf_len_cm)

    node_positions = nx.get_node_attributes(T, "pos")
    labels = segment_point_cloud(points, node_positions, stem_path, leaf_dict)

    organs = {"stem": points[labels == 1]}
    out_id = 0
    for lid in sorted(leaf_dict):
        lbl = lid + 1
        pts = points[labels == lbl]
        if len(pts) == 0:
            continue
        out_id += 1
        organs[f"leaf_{out_id}"] = pts

    if not return_debug:
        return organs
    debug = {
        "contracted": contracted, "nodes": nodes, "tree": T,
        "start": start, "stem_path": stem_path, "leaf_dict": leaf_dict,
        "labels": labels, "node_positions": node_positions,
    }
    return organs, debug
