import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.datasets import make_classification, make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


st.set_page_config(page_title="Decision Tree Visual Lab", layout="wide")


def impurity_from_counts(counts: np.ndarray, metric: str) -> float:
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    if metric == "gini":
        return float(1 - np.sum(p**2))
    return float(-np.sum(p * np.log2(p)))


def impurity(y: np.ndarray, metric: str) -> float:
    if y.size == 0:
        return 0.0
    counts = np.bincount(y, minlength=2)
    return impurity_from_counts(counts, metric)


def info_gain(y_parent: np.ndarray, y_left: np.ndarray, y_right: np.ndarray, metric: str) -> float:
    n = len(y_parent)
    if n == 0:
        return 0.0
    parent_imp = impurity(y_parent, metric)
    w = len(y_left) / n
    child_imp = w * impurity(y_left, metric) + (1 - w) * impurity(y_right, metric)
    return parent_imp - child_imp


@st.cache_data
def generate_dataset(kind: str, add_noise: bool, noise_std: float, random_state: int = 42):
    if kind == "Simple":
        X, y = make_moons(n_samples=350, noise=0.15, random_state=random_state)
    elif kind == "Complex":
        X, y = make_classification(
            n_samples=450,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            n_clusters_per_class=2,
            class_sep=1.1,
            random_state=random_state,
        )
    else:
        X, y = make_moons(n_samples=400, noise=0.35, random_state=random_state)
        rng = np.random.default_rng(random_state)
        flip_idx = rng.choice(len(y), size=int(0.08 * len(y)), replace=False)
        y[flip_idx] = 1 - y[flip_idx]

    if add_noise and noise_std > 0:
        rng = np.random.default_rng(random_state + 7)
        X = X + rng.normal(0, noise_std, X.shape)

    return X, y


def fit_tree(X, y, criterion, max_depth, min_samples_split, ccp_alpha=0.0):
    clf = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42,
        ccp_alpha=ccp_alpha,
    )
    clf.fit(X, y)
    return clf


def get_bounds(X):
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    pad_x = 0.2 * (x_max - x_min + 1e-9)
    pad_y = 0.2 * (y_max - y_min + 1e-9)
    return (x_min - pad_x, x_max + pad_x, y_min - pad_y, y_max + pad_y)


def extract_regions_and_splits(clf, bounds):
    tree = clf.tree_
    children_left = tree.children_left
    children_right = tree.children_right
    features = tree.feature
    thresholds = tree.threshold

    regions = {}
    splits = []

    def recurse(node, xmin, xmax, ymin, ymax, depth):
        regions[node] = (xmin, xmax, ymin, ymax, depth)
        left = children_left[node]
        right = children_right[node]
        if left == right:
            return

        f = features[node]
        thr = thresholds[node]
        split = {
            "node": node,
            "feature": int(f),
            "threshold": float(thr),
            "depth": depth,
            "left": int(left),
            "right": int(right),
            "region": (xmin, xmax, ymin, ymax),
        }
        if f == 0:
            split["line"] = (thr, ymin, thr, ymax)
            recurse(left, xmin, thr, ymin, ymax, depth + 1)
            recurse(right, thr, xmax, ymin, ymax, depth + 1)
        else:
            split["line"] = (xmin, thr, xmax, thr)
            recurse(left, xmin, xmax, ymin, thr, depth + 1)
            recurse(right, xmin, xmax, thr, ymax, depth + 1)
        splits.append(split)

    recurse(0, *bounds, 0)
    splits_sorted = sorted(splits, key=lambda d: (d["depth"], d["node"]))
    return regions, splits_sorted


def candidate_splits(X, y, metric):
    rows = []
    for f in range(X.shape[1]):
        xs = np.unique(np.round(X[:, f], 8))
        if len(xs) < 2:
            continue
        thresholds = (xs[:-1] + xs[1:]) / 2.0
        if len(thresholds) > 80:
            idx = np.linspace(0, len(thresholds) - 1, 80).astype(int)
            thresholds = thresholds[idx]

        for thr in thresholds:
            mask = X[:, f] <= thr
            if mask.sum() == 0 or mask.sum() == len(y):
                continue
            gain = info_gain(y, y[mask], y[~mask], metric)
            rows.append(
                {
                    "feature": f,
                    "threshold": float(thr),
                    "info_gain": float(gain),
                    "left_n": int(mask.sum()),
                    "right_n": int((~mask).sum()),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["feature", "threshold", "info_gain", "left_n", "right_n"])
    df = pd.DataFrame(rows).sort_values("info_gain", ascending=False).reset_index(drop=True)
    return df


def plot_boundary(ax, clf, X, y, bounds, title, alpha=0.25):
    x_min, x_max, y_min, y_max = bounds
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
    zz = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5], alpha=alpha, cmap="coolwarm")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k", s=30)
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")


def build_progressive_dot(clf, expanded_count):
    tree = clf.tree_
    left = tree.children_left
    right = tree.children_right
    feat = tree.feature
    thr = tree.threshold
    vals = tree.value.squeeze(axis=1)

    internal_nodes = [i for i in range(tree.node_count) if left[i] != right[i]]
    if expanded_count <= 0:
        expanded = set()
        current = None
    else:
        expanded = set(internal_nodes[:expanded_count])
        current = internal_nodes[min(expanded_count - 1, len(internal_nodes) - 1)] if internal_nodes else None

    lines = ["digraph Tree {", "node [shape=box, style=filled, color=black, fontname=Helvetica];"]

    def node_label(node):
        counts = vals[node]
        pred = int(np.argmax(counts))
        if left[node] == right[node] or node not in expanded:
            return f"Leaf\\nnode={node}\\nclass={pred}\\ncounts={counts.astype(int).tolist()}"
        return f"node={node}\\nX{feat[node]+1} <= {thr[node]:.3f}"

    def recurse(node):
        is_leaf = left[node] == right[node]
        is_expanded = node in expanded
        fill = "#cfe8ff"
        if node == current:
            fill = "#ffd166"
        elif is_leaf or not is_expanded:
            fill = "#d8f3dc"

        lines.append(f'{node} [label="{node_label(node)}", fillcolor="{fill}"];')
        if is_leaf or not is_expanded:
            return
        l, r = left[node], right[node]
        recurse(l)
        recurse(r)
        lines.append(f"{node} -> {l} [label='True'];")
        lines.append(f"{node} -> {r} [label='False'];")

    recurse(0)
    lines.append("}")
    return "\n".join(lines), internal_nodes


def explanation_block(text, observe):
    st.markdown(text)
    st.info(f"**What to observe:** {observe}")


st.title("🌳 Decision Tree Interactive Visualizer")

with st.sidebar:
    st.header("Controls")
    dataset_type = st.selectbox("Dataset", ["Simple", "Complex", "Noisy"])
    max_depth = st.slider("Max Depth", 1, 10, 4)
    min_samples_split = st.slider("Min Samples per Split", 2, 20, 4)
    impurity_measure = st.selectbox("Impurity Measure", ["gini", "entropy"], format_func=lambda x: x.capitalize())
    show_impurity = st.toggle("Show Impurity Values", value=True)
    add_noise = st.toggle("Add Noise to Dataset", value=False)
    enable_pruning = st.toggle("Enable Pruning", value=True)
    manual_feature = st.selectbox("Manual Split Feature", [0, 1], format_func=lambda x: f"Feature {x+1}")
    noise_std = st.slider("Gaussian Noise Std", 0.0, 1.0, 0.25, 0.05)


X, y = generate_dataset(dataset_type, add_noise, noise_std, random_state=42)
bounds = get_bounds(X)
fmin, fmax = float(X[:, manual_feature].min()), float(X[:, manual_feature].max())
split_threshold = st.sidebar.slider(
    "Split Threshold",
    min_value=float(fmin),
    max_value=float(fmax),
    value=float((fmin + fmax) / 2.0),
    step=float((fmax - fmin) / 200.0 if fmax > fmin else 0.01),
)

clf = fit_tree(X, y, impurity_measure, max_depth, min_samples_split)
regions, split_list = extract_regions_and_splits(clf, bounds)

tabs = st.tabs(
    [
        "1) Data & Splits",
        "2) Impurity Measures",
        "3) Split Selection",
        "4) Tree Growth",
        "5) Prediction Path",
        "6) Overfitting & Depth",
        "7) Noise & Pruning",
    ]
)


with tabs[0]:
    explanation_block(
        "Decision trees split feature space into rectangular regions using axis-aligned rules. "
        "Each split divides one region into two child regions recursively.",
        "As you move through split steps, regions become smaller and class purity in each region usually increases.",
    )

    step = st.slider("Split step", 0, len(split_list), min(3, len(split_list)), key="tab1_step")
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_boundary(ax, clf, X, y, bounds, "Data, Decision Regions, and Split Lines", alpha=0.22)

    for i, sp in enumerate(split_list[:step]):
        x1, y1, x2, y2 = sp["line"]
        ax.plot([x1, x2], [y1, y2], color="black", linewidth=1.4, alpha=0.9)

    if step > 0:
        sp = split_list[step - 1]
        for child, color in [(sp["left"], "#8ecae6"), (sp["right"], "#ffb703")]:
            xmin, xmax, ymin, ymax, _ = regions[child]
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, facecolor=color, alpha=0.20, edgecolor="none")
            ax.add_patch(rect)
    st.pyplot(fig)

    st.subheader("Manual split preview")
    mask = X[:, manual_feature] <= split_threshold
    y_left, y_right = y[mask], y[~mask]
    g_gain = info_gain(y, y_left, y_right, "gini")
    e_gain = info_gain(y, y_left, y_right, "entropy")

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k", s=30)
    if manual_feature == 0:
        ax2.axvline(split_threshold, color="purple", linewidth=2)
    else:
        ax2.axhline(split_threshold, color="purple", linewidth=2)
    ax2.set_title("Manual Feature-Threshold Split")
    ax2.set_xlabel("Feature 1")
    ax2.set_ylabel("Feature 2")
    st.pyplot(fig2)

    c1, c2, c3 = st.columns(3)
    c1.metric("Left samples", int(mask.sum()))
    c2.metric("Right samples", int((~mask).sum()))
    c3.metric("Info gain (selected metric)", f"{(g_gain if impurity_measure == 'gini' else e_gain):.4f}")


with tabs[1]:
    explanation_block(
        "Impurity measures quantify class mixing in a node. "
        "A good split reduces weighted child impurity relative to the parent.",
        "Compare Gini and Entropy on the same node: they usually agree on strong splits, but score them differently.",
    )

    tree = clf.tree_
    internal_nodes = [i for i in range(tree.node_count) if tree.children_left[i] != tree.children_right[i]]
    selected_node = st.selectbox("Node to analyze", internal_nodes if internal_nodes else [0], key="tab2_node")

    path = clf.decision_path(X)
    node_idx = path[:, selected_node].nonzero()[0]
    Xn, yn = X[node_idx], y[node_idx]

    if len(yn) == 0:
        st.warning("Selected node has no samples.")
    else:
        f = int(tree.feature[selected_node])
        t = float(tree.threshold[selected_node])
        m = Xn[:, f] <= t
        y_l, y_r = yn[m], yn[~m]

        counts_parent = np.bincount(yn, minlength=2)
        counts_left = np.bincount(y_l, minlength=2)
        counts_right = np.bincount(y_r, minlength=2)

        dist_df = pd.DataFrame(
            {
                "Node": ["Parent", "Parent", "Left", "Left", "Right", "Right"],
                "Class": ["0", "1", "0", "1", "0", "1"],
                "Count": [counts_parent[0], counts_parent[1], counts_left[0], counts_left[1], counts_right[0], counts_right[1]],
            }
        )
        st.plotly_chart(px.bar(dist_df, x="Node", y="Count", color="Class", barmode="group", title="Class Distribution"), use_container_width=True)

        g_parent = impurity(yn, "gini")
        e_parent = impurity(yn, "entropy")
        g_gain = info_gain(yn, y_l, y_r, "gini")
        e_gain = info_gain(yn, y_l, y_r, "entropy")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Gini (parent)", f"{g_parent:.4f}")
        m2.metric("Entropy (parent)", f"{e_parent:.4f}")
        m3.metric("Info gain (Gini)", f"{g_gain:.4f}")
        m4.metric("Info gain (Entropy)", f"{e_gain:.4f}")

        cand = candidate_splits(Xn, yn, impurity_measure)
        if len(cand):
            best = cand.iloc[0]
            fig_c = go.Figure()
            fig_c.add_trace(go.Bar(x=np.arange(len(cand)), y=cand["info_gain"], name="Information Gain"))
            fig_c.add_trace(
                go.Scatter(
                    x=[0],
                    y=[best["info_gain"]],
                    mode="markers+text",
                    text=["Best split"],
                    textposition="top center",
                    marker=dict(size=12, color="red"),
                    name="Best",
                )
            )
            fig_c.update_layout(title="Candidate Splits in Selected Node", xaxis_title="Rank", yaxis_title="Information Gain")
            st.plotly_chart(fig_c, use_container_width=True)
            st.caption(f"Best split: Feature {int(best['feature'])+1}, threshold {best['threshold']:.4f}")


with tabs[2]:
    explanation_block(
        "A tree tests many feature-threshold combinations and chooses the split with highest impurity reduction. "
        "This is the core optimization at each node.",
        "Observe how changing threshold shifts child class balance and the gain score immediately.",
    )

    split_df = candidate_splits(X, y, impurity_measure)
    if len(split_df) == 0:
        st.warning("No valid candidate split found.")
    else:
        st.dataframe(split_df.head(25), use_container_width=True)
        best = split_df.iloc[0]

        fig_rank = px.scatter(
            split_df.head(80),
            x=split_df.head(80).index,
            y="info_gain",
            color=split_df.head(80)["feature"].astype(str),
            title="Candidate Split Ranking by Information Gain",
            labels={"x": "Rank", "info_gain": "Impurity Reduction", "color": "Feature"},
        )
        fig_rank.add_scatter(x=[0], y=[best["info_gain"]], mode="markers", marker=dict(size=14, color="red"), name="Best")
        st.plotly_chart(fig_rank, use_container_width=True)

        st.subheader("Best split highlighted")
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        plot_boundary(ax3, clf, X, y, bounds, "Best split overlay", alpha=0.18)
        if int(best["feature"]) == 0:
            ax3.axvline(best["threshold"], color="red", linewidth=2.5)
        else:
            ax3.axhline(best["threshold"], color="red", linewidth=2.5)
        st.pyplot(fig3)

        f_sel = st.selectbox("Feature for threshold sensitivity", [0, 1], format_func=lambda x: f"Feature {x+1}", key="tab3_feat")
        f_lo, f_hi = float(X[:, f_sel].min()), float(X[:, f_sel].max())
        thr_live = st.slider("Adjust threshold", f_lo, f_hi, float((f_lo + f_hi) / 2), key="tab3_thr")
        m = X[:, f_sel] <= thr_live
        live_gain = info_gain(y, y[m], y[~m], impurity_measure)
        st.metric("Real-time impurity reduction", f"{live_gain:.4f}")


with tabs[3]:
    explanation_block(
        "Tree growth is recursive: pick best split at a node, create children, then repeat. "
        "Stopping criteria create leaves.",
        "As expansion grows, note depth increases and leaves represent more specific local rules.",
    )

    dot_all, internal_nodes = build_progressive_dot(clf, len([i for i in range(clf.tree_.node_count) if clf.tree_.children_left[i] != clf.tree_.children_right[i]]))
    max_expand = len(internal_nodes)
    expand_step = st.slider("Expand nodes one by one", 0, max_expand, min(4, max_expand) if max_expand > 0 else 0)
    dot_step, _ = build_progressive_dot(clf, expand_step)
    st.graphviz_chart(dot_step)

    node_depth = clf.tree_.max_depth
    depth_counts = {}
    stack = [(0, 0)]
    while stack:
        node, depth = stack.pop()
        depth_counts[depth] = depth_counts.get(depth, 0) + 1
        l, r = clf.tree_.children_left[node], clf.tree_.children_right[node]
        if l != r:
            stack.append((l, depth + 1))
            stack.append((r, depth + 1))
    depth_df = pd.DataFrame({"Depth": list(depth_counts.keys()), "Nodes": list(depth_counts.values())}).sort_values("Depth")
    st.plotly_chart(px.bar(depth_df, x="Depth", y="Nodes", title="Depth levels in current tree"), use_container_width=True)
    st.caption(f"Max depth reached: {node_depth}")


with tabs[4]:
    explanation_block(
        "Prediction follows one root-to-leaf route based on feature tests. "
        "Each step narrows the possible class outcomes.",
        "Track the traversed rules: the final leaf stores class counts used for prediction.",
    )

    mode = st.radio("Point source", ["Pick existing sample", "Manual input"], horizontal=True)
    if mode == "Pick existing sample":
        idx = st.slider("Sample index", 0, len(X) - 1, 0)
        point = X[idx]
        true_label = int(y[idx])
    else:
        c1, c2 = st.columns(2)
        x1 = c1.number_input("Feature 1", value=float(np.mean(X[:, 0])))
        x2 = c2.number_input("Feature 2", value=float(np.mean(X[:, 1])))
        point = np.array([x1, x2])
        true_label = None

    path_nodes = clf.decision_path(point.reshape(1, -1)).indices.tolist()
    pred = int(clf.predict(point.reshape(1, -1))[0])

    step_path = st.slider("Traversal step", 1, len(path_nodes), 1)
    traversed = set(path_nodes[:step_path])

    fig4, ax4 = plt.subplots(figsize=(8, 6))
    plot_boundary(ax4, clf, X, y, bounds, "Prediction traversal")
    split_map = {s["node"]: s for s in split_list}
    for n in traversed:
        if n in split_map:
            x1, y1, x2, y2 = split_map[n]["line"]
            ax4.plot([x1, x2], [y1, y2], color="gold", linewidth=3)
    ax4.scatter([point[0]], [point[1]], c=["lime"], edgecolors="black", s=140, marker="*")
    st.pyplot(fig4)

    st.subheader("Decision rules")
    for i, n in enumerate(path_nodes[:-1], start=1):
        f = clf.tree_.feature[n]
        t = clf.tree_.threshold[n]
        direction = "left (True)" if point[f] <= t else "right (False)"
        st.write(f"Step {i}: if Feature {f+1} <= {t:.4f} → go {direction}")

    if true_label is not None:
        st.write(f"True class: **{true_label}**")
    st.success(f"Final predicted class at leaf: **{pred}**")


with tabs[5]:
    explanation_block(
        "Greater depth can fit training data better but may hurt validation performance. "
        "That gap indicates overfitting.",
        "Look for the depth where validation accuracy peaks; beyond that, complexity often hurts generalization.",
    )

    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    shallow_depth = max(1, min(3, max_depth // 2 if max_depth > 1 else 1))
    deep_depth = max_depth

    shallow = fit_tree(Xtr, ytr, impurity_measure, shallow_depth, min_samples_split)
    deep = fit_tree(Xtr, ytr, impurity_measure, deep_depth, min_samples_split)

    fig5, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
    plot_boundary(axes[0], shallow, Xtr, ytr, bounds, f"Shallow tree (max_depth={shallow_depth})")
    plot_boundary(axes[1], deep, Xtr, ytr, bounds, f"Deep tree (max_depth={deep_depth})")
    st.pyplot(fig5)

    d_range = list(range(1, 11))
    train_acc, val_acc = [], []
    for d in d_range:
        m = fit_tree(Xtr, ytr, impurity_measure, d, min_samples_split)
        train_acc.append(accuracy_score(ytr, m.predict(Xtr)))
        val_acc.append(accuracy_score(yva, m.predict(Xva)))

    curve_df = pd.DataFrame({"Depth": d_range, "Train Accuracy": train_acc, "Validation Accuracy": val_acc})
    fig_curve = go.Figure()
    fig_curve.add_trace(go.Scatter(x=curve_df["Depth"], y=curve_df["Train Accuracy"], mode="lines+markers", name="Train"))
    fig_curve.add_trace(go.Scatter(x=curve_df["Depth"], y=curve_df["Validation Accuracy"], mode="lines+markers", name="Validation"))
    fig_curve.update_layout(title="Overfitting Curve", xaxis_title="Tree Depth", yaxis_title="Accuracy")
    st.plotly_chart(fig_curve, use_container_width=True)


with tabs[6]:
    explanation_block(
        "Noise can cause unstable, overly specific splits. Pruning removes weak branches to improve robustness. "
        "Cost-complexity pruning uses $ccp\_alpha$ to penalize complexity.",
        "Compare boundary smoothness and accuracy: pruned trees are often simpler and more stable on noisy data.",
    )

    noise_for_tab = st.slider("Extra noise for this comparison", 0.0, 1.0, 0.35, 0.05, key="tab7_noise")
    Xn, yn = generate_dataset(dataset_type, True, noise_for_tab, random_state=123)
    b2 = get_bounds(Xn)

    alpha = st.slider("Pruning strength (ccp_alpha)", 0.0, 0.05, 0.01, 0.001)
    unpruned = fit_tree(Xn, yn, impurity_measure, max_depth, min_samples_split, ccp_alpha=0.0)
    pruned_alpha = alpha if enable_pruning else 0.0
    pruned = fit_tree(Xn, yn, impurity_measure, max_depth, min_samples_split, ccp_alpha=pruned_alpha)

    fig7, ax7 = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
    plot_boundary(ax7[0], unpruned, Xn, yn, b2, "Unpruned tree")
    plot_boundary(ax7[1], pruned, Xn, yn, b2, f"Pruned tree (ccp_alpha={pruned_alpha:.3f})")
    st.pyplot(fig7)

    acc_un = accuracy_score(yn, unpruned.predict(Xn))
    acc_pr = accuracy_score(yn, pruned.predict(Xn))

    comp = pd.DataFrame(
        {
            "Model": ["Unpruned", "Pruned"],
            "Accuracy": [acc_un, acc_pr],
            "Node Count": [unpruned.tree_.node_count, pruned.tree_.node_count],
            "Leaf Count": [unpruned.tree_.n_leaves, pruned.tree_.n_leaves],
            "Depth": [unpruned.tree_.max_depth, pruned.tree_.max_depth],
        }
    )
    st.dataframe(comp, use_container_width=True)

    fig_comp = px.bar(
        comp.melt(id_vars="Model", value_vars=["Node Count", "Leaf Count", "Depth"]),
        x="variable",
        y="value",
        color="Model",
        barmode="group",
        title="Decision Boundary Complexity Comparison",
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    if not enable_pruning:
        st.warning("Pruning toggle is OFF, so both models are effectively unpruned.")


if show_impurity:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Current Root Impurity")
    st.sidebar.write(f"Gini: {impurity(y, 'gini'):.4f}")
    st.sidebar.write(f"Entropy: {impurity(y, 'entropy'):.4f}")