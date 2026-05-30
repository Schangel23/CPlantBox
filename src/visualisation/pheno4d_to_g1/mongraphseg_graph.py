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


def estimate_normals(points, k=16):
    """Per-point unoriented surface normal = smallest-eigenvalue eigenvector of
    the local covariance over the ``k`` nearest neighbours. Orientation is
    irrelevant here (only the angle between adjacent normals is used)."""
    points = np.asarray(points, float)
    n = len(points)
    if n < 3:
        return np.tile([0.0, 0.0, 1.0], (n, 1))
    kk = min(k, n)
    idx = KDTree(points).query(points, k=kk)[1]
    nb = points[idx]
    nb = nb - nb.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb) / kk
    _, v = np.linalg.eigh(cov)
    return v[:, :, 0]


def assign_points_regiongrow_instances(points, node_positions, leaf_dict,
                                       pseudostem_edges, k=12, max_edge_cm=3.0,
                                       normal_k=16, crease_deg=45.0,
                                       crease_penalty=8.0, distal_seed_start=0.0,
                                       distal_seed_end=1.0, fallback_labels=None,
                                       return_debug=False):
    """Crease-aware multi-source region growing (surface-smoothness watershed).

    Same multi-source flood as the geodesic assigner, but each kNN-graph edge
    cost is multiplied by a CREASE factor from the angle between the two
    endpoints' surface normals: smooth edges stay cheap; edges across a normal
    discontinuity (the seam where two blades touch, or a leaf-stem collar) become
    expensive. A leaf's flood stops at the seam instead of bleeding into a
    touching neighbour -- the direct fix for the contraction's merge ceiling.
    Seeds match the geodesic assigner (leaf midrib path; pseudostem basal nodes).
    """
    from geodesic_assignment import build_knn_graph
    points = np.asarray(points, float)
    n = len(points)
    labels = np.zeros(n, dtype=int)
    if n == 0:
        return (labels, {}) if return_debug else labels
    node_ids = list(node_positions)
    if not node_ids:
        return (labels, {}) if return_debug else labels
    node_arr = np.array([node_positions[nid] for nid in node_ids], dtype=float)
    nearest_node = np.array([node_ids[int(i)]
                             for i in KDTree(node_arr).query(points)[1]])

    ps_nodes = set()
    for a_id, b_id in pseudostem_edges:
        ps_nodes.add(a_id); ps_nodes.add(b_id)
    seed_by_label = {0: np.where(np.isin(nearest_node, list(ps_nodes)))[0]}
    for lid, path in sorted(leaf_dict.items()):
        if len(path) < 2:
            continue
        seg = np.array([
            np.linalg.norm(node_positions[path[i + 1]] - node_positions[path[i]])
            for i in range(len(path) - 1)])
        total = float(seg.sum())
        if total <= 1e-12:
            nodes = {path[-1]}
        else:
            cum = np.concatenate([[0.0], np.cumsum(seg)]) / total
            lo = min(distal_seed_start, distal_seed_end)
            hi = max(distal_seed_start, distal_seed_end)
            nodes = {nd for nd, fr in zip(path, cum) if lo <= fr <= hi} or {path[-1]}
        seed_by_label[lid] = np.where(np.isin(nearest_node, list(nodes)))[0]

    active = [(lab, s) for lab, s in seed_by_label.items() if len(s)]
    if not active:
        if fallback_labels is not None:
            labels = np.asarray(fallback_labels, int).copy()
        return (labels, {}) if return_debug else labels

    normals = estimate_normals(points, k=normal_k)
    G = build_knn_graph(points, k=k, max_edge_cm=max_edge_cm).tocoo()
    cosang = np.abs(np.einsum("ij,ij->i", normals[G.row], normals[G.col]))
    ang = np.degrees(np.arccos(np.clip(cosang, 0.0, 1.0)))      # 0..90, unoriented
    factor = 1.0 + crease_penalty * np.clip(ang / crease_deg, 0.0, None)
    W = csr_matrix((G.data * factor, (G.row, G.col)), shape=(n, n))
    W = W.maximum(W.T)

    aug = W.tolil()
    aug.resize((n + len(active), n + len(active)))
    src_labels = []
    for off, (lab, s) in enumerate(active):
        aug[n + off, np.asarray(s, int)] = 1e-9
        src_labels.append(lab)
    aug = aug.tocsr(); aug = aug.maximum(aug.T)
    dmat = dijkstra(aug, directed=False,
                    indices=np.arange(n, n + len(active)))[:, :n]
    reach = np.isfinite(dmat).any(axis=0)
    labels[reach] = np.array(src_labels, int)[np.argmin(dmat[:, reach], axis=0)]
    if fallback_labels is not None and (~reach).any():
        labels[~reach] = np.asarray(fallback_labels, int)[~reach]
    if return_debug:
        return labels, {"normals": normals, "crease_ang": ang,
                        "seed_by_label": seed_by_label, "source_labels": src_labels}
    return labels


def estimate_tangents(points, k=16):
    """Per-point ribbon direction = LARGEST-eigenvalue eigenvector of the local
    covariance (the long axis of a leaf-blade ribbon). Unoriented."""
    points = np.asarray(points, float)
    n = len(points)
    if n < 3:
        return np.tile([0.0, 0.0, 1.0], (n, 1))
    kk = min(k, n)
    idx = KDTree(points).query(points, k=kk)[1]
    nb = points[idx]
    nb = nb - nb.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb) / kk
    _, v = np.linalg.eigh(cov)
    return v[:, :, 2]


def split_ribbon_leaf_instances(points, leaf_id, max_edge_cm=2.0,
                                tangent_tol_deg=35.0, tangent_k=16,
                                knn_k=12, min_support=50):
    """Split predicted leaf instances that fused CROSSING blades by ribbon
    direction-continuity.

    Overlapping young whorls fail component-split because crossing blades have
    no spatial gap (one connected component). But each blade is a smooth ribbon
    with a continuous local tangent, and two crossing blades meet at a large
    tangent discontinuity. We gate a short-edge kNN graph by tangent alignment:
    an edge survives only if both endpoints' local ribbon directions agree
    within ``tangent_tol_deg``. The crossing is severed (mismatched tangents)
    while each smoothly-curving blade stays connected; connected components are
    the separated ribbons. Transferable cue (holds in sim AND real), unlike
    learned density features.
    """
    from geodesic_assignment import build_knn_graph
    from scipy.sparse.csgraph import connected_components as _cc
    points = np.asarray(points, float)
    leaf_id = np.asarray(leaf_id, int).copy()
    next_id = int(leaf_id.max()) + 1 if leaf_id.max() > 0 else 1
    for lid in [x for x in np.unique(leaf_id) if x > 0]:
        idx = np.where(leaf_id == lid)[0]
        if len(idx) < 2 * min_support:
            continue
        P = points[idx]
        tan = estimate_tangents(P, k=tangent_k)
        G = build_knn_graph(P, k=knn_k, max_edge_cm=max_edge_cm).tocoo()
        cosang = np.abs(np.einsum("ij,ij->i", tan[G.row], tan[G.col]))
        keep = np.degrees(np.arccos(np.clip(cosang, 0.0, 1.0))) <= tangent_tol_deg
        W = csr_matrix((G.data[keep], (G.row[keep], G.col[keep])),
                       shape=(len(P), len(P)))
        W = W.maximum(W.T)
        _, lab = _cc(W, directed=False)
        sizes = np.bincount(lab)
        big = np.where(sizes >= min_support)[0]
        if len(big) < 2:
            continue
        for j, b in enumerate(big):
            leaf_id[idx[lab == b]] = lid if j == 0 else next_id
            if j > 0:
                next_id += 1
        small = idx[~np.isin(lab, big)]
        if len(small):
            trees = [KDTree(points[idx[lab == b]]) for b in big]
            db = np.column_stack([t.query(points[small])[0] for t in trees])
            nearest_big = big[np.argmin(db, axis=1)]
            for s, b in zip(small, nearest_big):
                leaf_id[s] = leaf_id[idx[lab == b][0]]
    return leaf_id


def split_components_leaf_instances(points, leaf_id, max_edge_cm=1.5,
                                    knn_k=12, min_support=50):
    """Split predicted leaf instances by spatial connected components.

    Maize blades don't separate by surface crease (the canopy is one smooth
    manifold) but they ARE separate surfaces joined only at the base, so a kNN
    graph at a short edge length disconnects two merged blades wherever there is
    a physical gap between them. Each connected component >= ``min_support`` of a
    predicted instance becomes its own leaf. Catches spatially-separated merges
    that the geodesic-tip splitter (touching-blade divergence) misses; the
    support floor keeps occlusion fragments from spawning false leaves.
    """
    from geodesic_assignment import build_knn_graph
    from scipy.sparse.csgraph import connected_components as _cc
    points = np.asarray(points, float)
    leaf_id = np.asarray(leaf_id, int).copy()
    next_id = int(leaf_id.max()) + 1 if leaf_id.max() > 0 else 1
    for lid in [x for x in np.unique(leaf_id) if x > 0]:
        idx = np.where(leaf_id == lid)[0]
        if len(idx) < 2 * min_support:
            continue
        G = build_knn_graph(points[idx], k=knn_k, max_edge_cm=max_edge_cm)
        _, lab = _cc(G, directed=False)
        sizes = np.bincount(lab)
        big = np.where(sizes >= min_support)[0]
        if len(big) < 2:
            continue
        for j, b in enumerate(big):
            leaf_id[idx[lab == b]] = lid if j == 0 else next_id
            if j > 0:
                next_id += 1
        small = idx[~np.isin(lab, big)]
        if len(small):
            trees = [KDTree(points[idx[lab == b]]) for b in big]
            db = np.column_stack([t.query(points[small])[0] for t in trees])
            nearest_big = big[np.argmin(db, axis=1)]
            for s, b in zip(small, nearest_big):
                leaf_id[s] = leaf_id[idx[lab == b][0]]
    return leaf_id


def split_merged_leaf_instances(points, leaf_id, ps_mask, knn_k=8,
                                max_edge_cm=None, min_branch_len_cm=6.0,
                                min_tip_sep_cm=5.0, min_support=40,
                                tip_angle_min_deg=35.0, max_subleaves=4):
    """Split predicted leaf instances that fused >=2 real blades.

    The global Laplacian contraction pulls a whorl of touching blades onto one
    skeleton axis, so one predicted instance can own 2-4 GT leaves (the dominant
    under-count channel). The merged instance's RAW points still contain the
    separate blade arcs sharing a base near the insertion. We re-trace each
    instance with a LOCAL kNN graph (no over-contraction): find tips as local
    maxima of geodesic distance from the instance base, merge near-duplicate
    tips, and if >=2 well-separated tips survive, reassign points to the nearest
    tip geodesically. Local-maxima (not farthest-point sampling) avoids cutting
    a single curved blade at its midpoint.

    ``leaf_id``: per-point instance id (0 = non-leaf, 1..L = leaves).
    ``ps_mask``: bool per-point pseudostem mask (defines the insertion anchor).
    Returns a new ``leaf_id`` array with split instances given fresh ids.
    """
    from geodesic_assignment import build_knn_graph

    points = np.asarray(points, float)
    leaf_id = np.asarray(leaf_id, int).copy()
    ps_pts = points[ps_mask] if ps_mask is not None and ps_mask.any() else None
    ps_tree = KDTree(ps_pts) if ps_pts is not None and len(ps_pts) else None
    next_id = int(leaf_id.max()) + 1 if leaf_id.max() > 0 else 1

    for lid in [x for x in np.unique(leaf_id) if x > 0]:
        idx = np.where(leaf_id == lid)[0]
        if len(idx) < 2 * min_support:
            continue
        P = points[idx]
        # local edge scale from median nearest-neighbour spacing (x3)
        if max_edge_cm is None:
            d = KDTree(P).query(P, k=min(4, len(P)))[0]
            edge = float(np.clip(3.0 * np.median(d[:, 1:]), 1.5, 6.0))
        else:
            edge = max_edge_cm
        G = build_knn_graph(P, k=knn_k, max_edge_cm=edge)

        # base = instance point nearest the pseudostem (insertion); else lowest z
        if ps_tree is not None:
            base = int(np.argmin(ps_tree.query(P)[0]))
        else:
            base = int(np.argmin(P[:, 2]))

        geo = dijkstra(G, directed=False, indices=base)
        fin = np.isfinite(geo)
        if fin.sum() < min_support:
            continue

        # tips = local maxima of geodesic distance among reachable points
        Gc = G.tocsr()
        tips = []
        for i in np.where(fin)[0]:
            if geo[i] < min_branch_len_cm:
                continue
            nbrs = Gc.indices[Gc.indptr[i]:Gc.indptr[i + 1]]
            nbrs = [n for n in nbrs if fin[n]]
            if nbrs and all(geo[i] >= geo[n] for n in nbrs):
                tips.append(i)
        # unreachable components: their farthest-from-base point is also a tip
        if (~fin).sum() >= min_support:
            comp_idx = np.where(~fin)[0]
            tips.append(int(comp_idx[np.argmax(P[comp_idx, 2])]))

        if len(tips) < 2:
            continue
        # merge near-duplicate tips (same blade end): greedily keep farthest,
        # drop tips within min_tip_sep_cm euclidean of a kept tip
        tips = sorted(tips, key=lambda t: -(geo[t] if fin[t] else 1e9))
        kept = []
        for t in tips:
            if all(np.linalg.norm(P[t] - P[k]) > min_tip_sep_cm for k in kept):
                kept.append(t)
        if len(kept) < 2:
            continue
        # require tip directions (base->tip) to diverge
        dirs = [P[t] - P[base] for t in kept]
        keep2 = [kept[0]]
        dir2 = [dirs[0]]
        for t, dv in zip(kept[1:], dirs[1:]):
            if all(_angle_deg(dv, d) >= tip_angle_min_deg for d in dir2):
                keep2.append(t); dir2.append(dv)
        kept = keep2[:max_subleaves]
        if len(kept) < 2:
            continue

        # multi-source geodesic from each kept tip; assign point to nearest tip
        dmat = dijkstra(G, directed=False, indices=kept)
        reach = np.isfinite(dmat).any(axis=0)
        sub = np.full(len(P), -1, int)
        sub[reach] = np.argmin(dmat[:, reach], axis=0)
        # unreachable points -> nearest kept tip euclidean
        if (~reach).any():
            ktree = KDTree(P[kept])
            sub[~reach] = ktree.query(P[~reach])[1]

        # enforce support floor: fold tiny sub-instances into the largest
        labels_u, counts = np.unique(sub, return_counts=True)
        big = labels_u[counts >= min_support]
        if len(big) < 2:
            continue
        # relabel: first sub keeps lid, rest get fresh ids; small subs -> nearest big
        if not np.isin(sub, big).all():
            big_tips = [kept[b] for b in big]
            btree = KDTree(P[big_tips])
            small = ~np.isin(sub, big)
            sub[small] = big[btree.query(P[small])[1]]
        for j, b in enumerate(big):
            tgt = lid if j == 0 else next_id
            if j > 0:
                next_id += 1
            leaf_id[idx[sub == b]] = tgt
    return leaf_id


def split_tipseed_leaf_instances(points, leaf_id, ps_mask, knn_k=12,
                                 max_edge_cm=None, min_branch_len_cm=12.0,
                                 min_tip_sep_cm=8.0, min_support=100,
                                 saddle_frac_max=0.6, max_subleaves=4):
    """Split fused blades by TIP-SEEDED competitive growth gated on a
    base-bundle SADDLE test (not base->tip divergence).

    The dominant residual merge is DISTICHOUS same-side leaves: two blades at
    the SAME azimuth, distinct at their tips but fused through the shared basal
    bundle. ``split_merged`` rejects these because its base->tip direction gate
    sees near-parallel vectors; the local cuts (gap / tangent / crease) also
    fail at the bundle, where the blades look identical. Every cue that bites is
    local, and at the bundle there is nothing local to bite on.

    This routine seeds from the distinct TIPS instead (geodesic maxima from the
    base, where the visual shows the blades ARE separable) and decides
    separation by TOPOLOGY rather than angle: rooting a shortest-path tree at
    the base, two tips are accepted as separate leaves when their lowest common
    ancestor sits LOW -- a saddle whose geodesic-from-base is a small fraction
    (``saddle_frac_max``) of the tip distances, i.e. the ribbons diverge down at
    the bundle. A single folded blade yields two maxima that diverge HIGH (large
    saddle) and is left intact. Euclidean tip separation replaces the angular
    gate, so same-azimuth blades are no longer suppressed. Points are then grown
    competitively to the nearest kept tip (multi-source geodesic), which splits
    the shared bundle along its geodesic watershed -- imperfect there, but each
    blade keeps its own tip + lamina, which is what clears the 0.5-IoU bar.
    """
    from geodesic_assignment import build_knn_graph

    points = np.asarray(points, float)
    leaf_id = np.asarray(leaf_id, int).copy()
    ps_pts = points[ps_mask] if ps_mask is not None and ps_mask.any() else None
    ps_tree = KDTree(ps_pts) if ps_pts is not None and len(ps_pts) else None
    next_id = int(leaf_id.max()) + 1 if leaf_id.max() > 0 else 1

    def _lca_geo(a, b, pred, geo, base):
        """Geodesic-from-base of the lowest common ancestor of tips a, b in the
        base-rooted shortest-path tree (the saddle between the two ribbons)."""
        anc = set()
        cur = a
        while cur >= 0 and cur != base:
            anc.add(cur)
            cur = pred[cur]
        anc.add(base)
        cur = b
        while cur >= 0 and cur != base:
            if cur in anc:
                return geo[cur]
            cur = pred[cur]
        return geo[base]

    for lid in [x for x in np.unique(leaf_id) if x > 0]:
        idx = np.where(leaf_id == lid)[0]
        if len(idx) < 2 * min_support:
            continue
        P = points[idx]
        if max_edge_cm is None:
            d = KDTree(P).query(P, k=min(4, len(P)))[0]
            edge = float(np.clip(3.0 * np.median(d[:, 1:]), 1.5, 6.0))
        else:
            edge = max_edge_cm
        G = build_knn_graph(P, k=knn_k, max_edge_cm=edge)

        if ps_tree is not None:
            base = int(np.argmin(ps_tree.query(P)[0]))
        else:
            base = int(np.argmin(P[:, 2]))

        geo, pred = dijkstra(G, directed=False, indices=base,
                             return_predecessors=True)
        fin = np.isfinite(geo)
        if fin.sum() < min_support:
            continue

        # tips = geodesic local maxima reachable from the base
        Gc = G.tocsr()
        tips = []
        for i in np.where(fin)[0]:
            if geo[i] < min_branch_len_cm:
                continue
            nbrs = Gc.indices[Gc.indptr[i]:Gc.indptr[i + 1]]
            nbrs = [n for n in nbrs if fin[n]]
            if nbrs and all(geo[i] >= geo[n] for n in nbrs):
                tips.append(i)
        if len(tips) < 2:
            continue

        # farthest-first, with euclidean dedupe AND the base-bundle saddle gate
        tips = sorted(tips, key=lambda t: -geo[t])
        kept = [tips[0]]
        for t in tips[1:]:
            if any(np.linalg.norm(P[t] - P[k]) <= min_tip_sep_cm for k in kept):
                continue
            diverges_low = True
            for k in kept:
                saddle = _lca_geo(t, k, pred, geo, base)
                denom = max(min(geo[t], geo[k]), 1e-6)
                if saddle / denom > saddle_frac_max:
                    diverges_low = False
                    break
            if diverges_low:
                kept.append(t)
        kept = kept[:max_subleaves]
        if len(kept) < 2:
            continue

        # competitive growth: each point to its nearest kept tip (geodesic)
        dmat = dijkstra(G, directed=False, indices=kept)
        reach = np.isfinite(dmat).any(axis=0)
        sub = np.full(len(P), -1, int)
        sub[reach] = np.argmin(dmat[:, reach], axis=0)
        if (~reach).any():
            sub[~reach] = KDTree(P[kept]).query(P[~reach])[1]

        labels_u, counts = np.unique(sub, return_counts=True)
        big = labels_u[counts >= min_support]
        if len(big) < 2:
            continue
        if not np.isin(sub, big).all():
            big_tips = [kept[b] for b in big]
            btree = KDTree(P[big_tips])
            small = ~np.isin(sub, big)
            sub[small] = big[btree.query(P[small])[1]]
        for j, b in enumerate(big):
            tgt = lid if j == 0 else next_id
            if j > 0:
                next_id += 1
            leaf_id[idx[sub == b]] = tgt
    return leaf_id


def _persistent_tips(Gcsr, geo, persistence_cm, min_branch_len_cm):
    """0-dim topological-persistence peaks of the geodesic field ``geo`` over the
    graph ``Gcsr``. A peak (geodesic local max = blade tip) is kept if it persists
    -- i.e. when its basin merges into a higher one at a saddle, the drop
    ``peak - saddle`` is >= ``persistence_cm``. Scale-adaptive: a short top leaf
    (peak ~10 cm above its collar) and a long bottom leaf (peak ~70 cm) are both
    kept, while shallow within-blade margin bumps die at shallow saddles and are
    discarded. The global maximum never dies and is always a tip. Returns the
    node indices of surviving peaks.
    """
    n = len(geo)
    fin = np.isfinite(geo)
    order = [i for i in np.argsort(-geo) if fin[i]]
    parent = np.full(n, -1)        # union-find: node -> representative
    peak = {}                       # comp root -> its peak node
    peakval = {}                    # comp root -> peak geodesic value
    tips = []

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for i in order:
        nb = Gcsr.indices[Gcsr.indptr[i]:Gcsr.indptr[i + 1]]
        roots = list({int(find(j)) for j in nb if parent[j] != -1})
        parent[i] = i
        if not roots:                       # new basin born at this peak
            peak[i] = i; peakval[i] = geo[i]
            continue
        roots.sort(key=lambda r: -peakval[r])
        main = roots[0]
        parent[i] = main
        for r in roots[1:]:                 # lower basin dies into main at geo[i]
            if peakval[r] - geo[i] >= persistence_cm and geo[peak[r]] >= min_branch_len_cm:
                tips.append(peak[r])
            parent[r] = main                # union
        # main absorbs i; peak/peakval of main unchanged (i is lower)
    # the surviving global basin's peak is always a tip
    if order:
        groot = find(order[0])
        if geo[peak[groot]] >= min_branch_len_cm:
            tips.append(peak[groot])
    return tips


def relabel_leaves_tipdriven(points, leaf_mask, base_pts, knn_k=12,
                             max_edge_cm=None, min_branch_len_cm=8.0,
                             min_tip_geo_sep_cm=20.0, persistence_cm=8.0,
                             source_band_cm=3.0,
                             min_support=80, max_leaves=40):
    """Tip-DRIVEN leaf instance labelling on the non-stem surface.

    The skeleton-contraction core sets the instance count from the medial axis,
    which FUSES touching distichous blades into one branch -> chronic under-count
    that no assign mode or post-hoc split recovers (the split bleeds across the
    shared bundle). This routine inverts the dependency: the instance count comes
    from the leaf TIPS, which the diagnosis showed ARE separable even when the
    bases are fused.

    On the non-stem surface graph we grow a geodesic field from the collar band
    (leaf points adjacent to the stem) and seed tips by GEODESIC farthest-point
    sampling with a minimum geodesic separation: the global farthest point is a
    tip, then the next farthest point whose geodesic distance to every existing
    tip exceeds ``min_tip_geo_sep_cm``, and so on while the farthest qualifying
    point is still a real blade (>= ``min_branch_len_cm`` from the collar). Two
    margin bumps on ONE blade are geodesically close and suppressed; two stacked
    same-azimuth blades are geodesically far (the path dips to the bundle and
    back) so both survive. Every leaf point is then assigned to its nearest tip;
    the stem acts as a hard barrier (it is absent from the graph), so flow from a
    tip cannot leak across the plant. Transferable geometry -- no learned stats.

    ``leaf_mask``: bool per-point, candidate leaf points (non-stem).
    ``base_pts``: (M,3) stem/bundle points (the barrier / collar anchor).
    Returns a per-point ``leaf_id`` over ALL points (0 = non-leaf/unassigned).
    """
    from geodesic_assignment import build_knn_graph

    points = np.asarray(points, float)
    leaf_id = np.zeros(len(points), int)
    idx = np.where(leaf_mask)[0]
    if len(idx) < min_support:
        return leaf_id
    P = points[idx]

    if max_edge_cm is None:
        d = KDTree(P).query(P, k=min(4, len(P)))[0]
        edge = float(np.clip(3.0 * np.median(d[:, 1:]), 1.5, 6.0))
    else:
        edge = max_edge_cm
    G = build_knn_graph(P, k=knn_k, max_edge_cm=edge)

    # collar sources = leaf points adjacent to the stem; else global lowest-z
    if base_pts is not None and len(base_pts):
        dstem = KDTree(np.asarray(base_pts, float)).query(P)[0]
        src = np.where(dstem <= source_band_cm)[0]
        if len(src) == 0:
            src = np.array([int(np.argmin(dstem))])
    else:
        src = np.array([int(np.argmin(P[:, 2]))])

    geo = dijkstra(G, directed=False, indices=src)
    geo = geo.min(axis=0) if geo.ndim == 2 else geo
    fin = np.isfinite(geo)
    # any unreachable component (occlusion / weak link): seed it from its lowest pt
    extra_src = []
    if (~fin).sum() >= min_support:
        comp = np.where(~fin)[0]
        extra_src.append(int(comp[np.argmin(P[comp, 2])]))
        if extra_src:
            g2 = dijkstra(G, directed=False, indices=extra_src)
            g2 = g2.min(axis=0) if g2.ndim == 2 else g2
            geo = np.where(np.isfinite(geo), geo, g2)
            fin = np.isfinite(geo)

    # tip seeding: scale-adaptive topological-persistence peaks of the geo field
    tips = _persistent_tips(G.tocsr(), geo, persistence_cm, min_branch_len_cm)
    # collapse any peaks that ended up geodesically near each other (one blade,
    # twin margin maxima that both cleared persistence): keep the farther one
    if len(tips) > 1 and min_tip_geo_sep_cm > 0:
        tips = sorted(tips, key=lambda t: -geo[t])
        kept = [tips[0]]
        dk = dijkstra(G, directed=False, indices=kept[0])
        for t in tips[1:]:
            if np.isfinite(dk[t]) and dk[t] <= min_tip_geo_sep_cm:
                continue
            kept.append(t)
            dt = dijkstra(G, directed=False, indices=t)
            dk = np.minimum(dk, np.where(np.isfinite(dt), dt, np.inf))
        tips = kept
    tips = tips[:max_leaves]
    if not tips:
        return leaf_id

    # assign each leaf point to its nearest tip (multi-source geodesic)
    dmat = dijkstra(G, directed=False, indices=tips)
    dmat = dmat if dmat.ndim == 2 else dmat[None, :]
    reach = np.isfinite(dmat).any(axis=0)
    sub = np.full(len(P), -1, int)
    sub[reach] = np.argmin(np.where(np.isfinite(dmat), dmat, np.inf)[:, reach],
                           axis=0)
    if (~reach).any():
        sub[~reach] = KDTree(P[tips]).query(P[~reach])[1]

    # support floor: fold tiny instances into the nearest surviving tip
    labels_u, counts = np.unique(sub, return_counts=True)
    big = labels_u[counts >= min_support]
    if len(big) == 0:
        big = labels_u[[np.argmax(counts)]]
    if not np.isin(sub, big).all():
        big_tips = [tips[b] for b in big]
        btree = KDTree(P[big_tips])
        small = ~np.isin(sub, big)
        sub[small] = big[btree.query(P[small])[1]]

    for new_id, b in enumerate(big, start=1):
        leaf_id[idx[sub == b]] = new_id
    return leaf_id


def segment_plant_pseudostem(points, n_skel_nodes=400, min_leaf_len_cm=3.0,
                             angle_threshold_deg=50, prune="length",
                             support_frac=0.02, recover_branches=False,
                             recovery_angle_deg=35.0, contraction_iters=8,
                             assign="segment", geodesic_k=10,
                             geodesic_max_edge_cm=3.0,
                             distal_seed_start=0.0,
                             distal_seed_end=1.0,
                             normal_k=16,
                             crease_deg=45.0,
                             crease_penalty=8.0,
                             pseudostem_basal_only=False,
                             pseudostem_top_margin_cm=2.0,
                             split_merged=False,
                             split_components=False,
                             split_component_max_edge_cm=1.5,
                             split_ribbon=False,
                             split_ribbon_max_edge_cm=2.0,
                             split_ribbon_tangent_tol_deg=35.0,
                             split_tipseed=False,
                             split_tipseed_saddle_frac_max=0.6,
                             split_tipseed_min_tip_sep_cm=8.0,
                             tipdriven=False,
                             tipdriven_min_tip_geo_sep_cm=20.0,
                             tipdriven_min_branch_len_cm=8.0,
                             tipdriven_persistence_cm=8.0,
                             split_min_branch_len_cm=12.0,
                             split_min_support=100,
                             split_tip_angle_min_deg=60.0,
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

    # Optionally confine the pseudostem to the basal bundle (base -> lowest leaf
    # collar). Above the lowest insertion the shared bundle runs parallel to
    # every blade; a tall pseudostem polyline there steals lamina points by 3-D
    # proximity (the dominant IoU-loss channel). Truncating to below the lowest
    # insertion lets mid/high blade points fall to their own leaf midrib.
    if pseudostem_basal_only and leaf_dict and ps_edges:
        ins_z = [node_positions[path[0]][2] for path in leaf_dict.values()
                 if len(path) >= 1]
        if ins_z:
            z_cut = min(ins_z) + pseudostem_top_margin_cm
            ps_edges = [(a, b) for (a, b) in ps_edges
                        if max(node_positions[a][2], node_positions[b][2]) <= z_cut]

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
    elif assign == "regiongrow":
        labels, geodesic_debug = assign_points_regiongrow_instances(
            points, node_positions, leaf_dict, ps_edges,
            k=geodesic_k, max_edge_cm=geodesic_max_edge_cm,
            normal_k=normal_k, crease_deg=crease_deg, crease_penalty=crease_penalty,
            distal_seed_start=distal_seed_start, distal_seed_end=distal_seed_end,
            fallback_labels=np.where(labels_segment == 1, 0,
                                     np.maximum(labels_segment - 1, 0)),
            return_debug=True)
        label_for_pseudostem = 0
        label_for_leaf = lambda lid: lid
    else:
        raise ValueError("assign must be 'segment', 'geodesic' or 'regiongrow'")

    # normalised per-point instance id: 0 = pseudostem/unassigned, 1..L leaves
    ps_mask = labels == label_for_pseudostem
    leaf_id = np.zeros(len(points), dtype=int)
    for lid in sorted(leaf_dict):
        leaf_id[labels == label_for_leaf(lid)] = lid

    # Tip-driven override: discard the skeleton's (merge-capped) leaf instances
    # and relabel the non-stem surface from leaf tips. Instance count then comes
    # from the separable tips, not the fusing medial axis. The pseudostem mask is
    # the barrier; everything not pseudostem is candidate leaf surface.
    if tipdriven:
        leaf_id = relabel_leaves_tipdriven(
            points, ~ps_mask, points[ps_mask],
            min_tip_geo_sep_cm=tipdriven_min_tip_geo_sep_cm,
            min_branch_len_cm=tipdriven_min_branch_len_cm,
            persistence_cm=tipdriven_persistence_cm,
            min_support=max(split_min_support // 2, 40))

    if leaf_id.max() > 0 and not tipdriven:
        if split_merged:
            leaf_id = split_merged_leaf_instances(
                points, leaf_id, ps_mask,
                min_branch_len_cm=split_min_branch_len_cm,
                min_support=split_min_support,
                tip_angle_min_deg=split_tip_angle_min_deg)
        if split_components:
            leaf_id = split_components_leaf_instances(
                points, leaf_id, max_edge_cm=split_component_max_edge_cm,
                min_support=split_min_support)
        if split_ribbon:
            leaf_id = split_ribbon_leaf_instances(
                points, leaf_id, max_edge_cm=split_ribbon_max_edge_cm,
                tangent_tol_deg=split_ribbon_tangent_tol_deg,
                min_support=split_min_support)
        if split_tipseed:
            leaf_id = split_tipseed_leaf_instances(
                points, leaf_id, ps_mask,
                min_branch_len_cm=split_min_branch_len_cm,
                min_support=split_min_support,
                saddle_frac_max=split_tipseed_saddle_frac_max,
                min_tip_sep_cm=split_tipseed_min_tip_sep_cm)

    organs = {"pseudostem": points[ps_mask]}
    out_id = 0
    for lid in sorted(x for x in np.unique(leaf_id) if x > 0):
        pts = points[leaf_id == lid]
        if len(pts) == 0:
            continue
        out_id += 1
        organs[f"leaf_{out_id}"] = pts

    if not return_debug:
        return organs
    debug = {"tree": T, "start": start, "pseudostem_edges": ps_edges,
             "leaf_dict": leaf_dict, "labels": labels, "leaf_id": leaf_id,
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
