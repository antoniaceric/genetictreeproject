import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, linregress
from pathlib import Path
import plotly.graph_objects as go


def plot_multiple_distance_correlations(gen_dist, distances_dict, ncols=3, save_path=None, filter_high_gen_dist=False):
    """
    Plot scatterplots of genetic distance vs. other distance types, with Spearman correlation and regression lines.

    Parameters:
    - gen_dist: NxN genetic distance matrix
    - distances_dict: dict {label: distance_matrix}
    - ncols: number of columns in the subplot grid
    - save_path: path to save the figure
    - filter_high_gen_dist: if True, only include pairs with genetic distance > 0.7
    """
    nplots = len(distances_dict)
    nrows = (nplots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes = axes.flatten()

    # Extract upper triangle (excluding diagonal)
    i_upper = np.triu_indices_from(gen_dist, k=1)
    flat_gen = gen_dist[i_upper]

    for ax, (label, other_dist) in zip(axes, distances_dict.items()):
        flat_other = other_dist[i_upper]

        # Optional filtering for high genetic distance
        if filter_high_gen_dist:
            mask = flat_gen > 0.7
            x = flat_gen[mask]
            y = flat_other[mask]
        else:
            x = flat_gen
            y = flat_other

        # Compute correlation
        rho, pval = spearmanr(x, y)
        ax.scatter(x, y, alpha=0.4, s=10)
        
        # Fit and plot regression line
        slope, intercept, *_ = linregress(x, y)
        x_fit = np.linspace(min(x), max(x), 100)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, color='red', linestyle='--', label='Linear fit')
        ax.legend()

        ax.set_title(f"{label}\nρ = {rho:.2f}, p = {pval:.2g}")
        ax.set_xlabel("Genetic distance")
        ax.set_ylabel(f"{label} distance")

    # Hide any unused subplots
    for ax in axes[nplots:]:
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved multi-panel plot to: {save_path}")
    else:
        plt.show()
    plt.close()

def plot_sankey_agreement(labels_gen, labels_other, method_name, save_dir):
    """
    Plot Sankey-style cluster flow from genetic clustering to another clustering method.

    Parameters:
    - labels_gen: array-like, genetic cluster assignments
    - labels_other: array-like, comparison method cluster assignments
    - method_name: str, name of the other clustering method (e.g., "Embedding Euclidean")
    - save_dir: Path, directory to save the .png file
    """
    sources = []
    targets = []
    values = []
    label_map = {}
    counter = 0

    def get_label_id(label):
        nonlocal counter
        if label not in label_map:
            label_map[label] = counter
            counter += 1
        return label_map[label]

    # Build flow from genetic to method clusters
    for g in np.unique(labels_gen):
        for m in np.unique(labels_other):
            count = np.sum((labels_gen == g) & (labels_other == m))
            if count > 0:
                src_label = f"Gen {g}"
                tgt_label = f"{method_name} {m}"
                src = get_label_id(src_label)
                tgt = get_label_id(tgt_label)
                sources.append(src)
                targets.append(tgt)
                values.append(count)

    # Create ordered label list
    index_to_label = [None] * len(label_map)
    for label, idx in label_map.items():
        index_to_label[idx] = label

    # Build Sankey plot
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=index_to_label,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values
        )
    )])

    fig.update_layout(
        title_text=f"Cluster Flow: Genetic → {method_name}",
        font_size=10,
        margin=dict(t=30, l=10, r=10, b=10)
    )

    # Save to PNG
    filename = f"sankey_genetic_to_{method_name.replace(' ', '_').lower()}.png"
    out_path = Path(save_dir) / filename
    fig.write_image(str(out_path), format="png", scale=2)
    print(f"Saved Sankey diagram to: {out_path}")

    import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path

def plot_multi_sankey_grid(ref_labels, label_dict, method_names, save_path):
    import plotly.subplots as sp
    import plotly.graph_objects as go

    n_methods = len(method_names)
    ncols = 3
    nrows = (n_methods + ncols - 1) // ncols

    fig = sp.make_subplots(
        rows=nrows, cols=ncols,
        specs=[[{"type": "domain"}] * ncols for _ in range(nrows)],
        subplot_titles=method_names
    )

    for idx, name in enumerate(method_names):
        target_labels = label_dict[name]
        ref = ref_labels
        target = target_labels
        ref_unique = sorted(set(ref))
        tgt_unique = sorted(set(target))

        label_names = [f"G{i}" for i in ref_unique] + [f"{name[:3]}{j}" for j in tgt_unique]
        label_idx_map = {v: i for i, v in enumerate(label_names)}

        counts = {}
        source_sums = {}
        for g, t in zip(ref, target):
            src_label = f"G{g}"
            tgt_label = f"{name[:3]}{t}"
            src = label_idx_map[src_label]
            tgt = label_idx_map[tgt_label]
            key = (src, tgt)
            counts[key] = counts.get(key, 0) + 1
            source_sums[src] = source_sums.get(src, 0) + 1

        sources, targets, values, hovertexts = [], [], [], []
        for (src, tgt), count in counts.items():
            sources.append(src)
            targets.append(tgt)
            values.append(count)
            pct = 100 * count / source_sums[src]
            hovertexts.append(f"{count} ({pct:.1f}%)")

        sankey = go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                label=label_names,
                line=dict(color="black", width=0.5),
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                label=hovertexts
            )
        )

        row = idx // ncols + 1
        col = idx % ncols + 1
        fig.add_trace(sankey, row=row, col=col)

    fig.update_layout(
        title=dict(
            text="Clustering Agreement with Genetic Clustering",
            x=0.5,                # ← Center the title
            xanchor="center",
            font=dict(size=16)
        ),
        font_size=11,
        height=380 * nrows,
        width=1200,
        margin=dict(t=100, l=20, r=20, b=40)  # ← more space under title
    )

    fig.write_image(str(save_path), scale=2)
    fig.write_html(str(save_path).replace(".png", ".html"))
    print(f"Sankey grid saved to: {save_path}")