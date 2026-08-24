from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import py3Dmol
from IPython.display import HTML, display
from rdkit import Chem
from rdkit.Chem import Mol


def visualize_conformers_grid(
    mol: Mol,
    weights: Sequence[float],
    key_conformers: Iterable[int],
    top_n: int = 5,
    style: str = "stick",
    n_cols: int = 4,
    width: int = 250,
    height: int = 250,
    show_all: bool = False,
    sort_by_weight: bool = True,
) -> None:
    """Render a molecule's conformers as a grid of 3D viewers in a notebook.

    Highlights the true key conformers and the top predicted ones by color,
    so you can eyeball how well predicted weights line up with the known key
    instances. Displays the grid directly via IPython; nothing is returned.

    Args:
        mol (rdkit.Chem.Mol): Molecule with one embedded conformer per entry
            in ``weights``.
        weights (Sequence[float]): Predicted weight for each conformer, in
            conformer-index order.
        key_conformers (Iterable[int]): Indices of the true key conformers,
            highlighted in red.
        top_n (int): Number of highest-weighted conformers to highlight in
            blue as predictions.
        style (str): py3Dmol rendering style (e.g. ``"stick"``).
        n_cols (int): Number of viewers per grid row.
        width (int): Width in pixels of each viewer.
        height (int): Height in pixels of each viewer.
        show_all (bool): If True, show every conformer instead of only the
            key and top-predicted ones.
        sort_by_weight (bool): If True, order the shown conformers by
            descending predicted weight.
    """

    num_confs = mol.GetNumConformers()
    if num_confs != len(weights):
        raise ValueError("Number of weights must equal number of conformers")

    # top-N predicted indices
    top_indices = set(np.argsort(weights)[-top_n:][::-1])
    key_conformers = set(key_conformers)

    if show_all:
        conf_indices = list(range(num_confs))
    else:
        conf_indices = sorted(key_conformers.union(top_indices))

    # sort conformers by weight if requested
    if sort_by_weight:
        conf_indices = sorted(conf_indices, key=lambda i: weights[i], reverse=True)

    viewers_html = []
    for i in conf_indices:
        conf = mol.GetConformer(int(i))
        block = Chem.MolToMolBlock(mol, confId=conf.GetId())

        color = "0xAAAAAA"  # default grey
        label = f"Conf {i} (w={weights[i]:.2f})"
        if i in key_conformers:
            color = "0xFF0000"  # red
            label += " [TRUE]"
        elif i in top_indices:
            color = "0x0000FF"  # blue
            label += " [PRED]"

        viewer = py3Dmol.view(width=width, height=height)
        viewer.addModel(block, "sdf")
        viewer.setStyle({style: {"color": color}})
        viewer.zoomTo()

        html = viewer._make_html()
        viewers_html.append(f"<div style='display:inline-block; text-align:center;'>{html}<br>{label}</div>")

    # arrange into grid
    rows = []
    for i in range(0, len(viewers_html), n_cols):
        row_html = "".join(viewers_html[i : i + n_cols])
        rows.append(f"<div style='margin-bottom:20px'>{row_html}</div>")

    # add legend
    legend_html = """
    <div style='margin:10px 0;'>
      <b>Legend:</b> 
      <span style='color:red;'>[TRUE]=Ground truth</span> | 
      <span style='color:blue;'>[PRED]=Top predicted</span> | 
      <span style='color:gray;'>Others</span>
    </div>
    """

    display(HTML(legend_html + "".join(rows)))
