# streamlit_master_app_v5.py
# -----------------------------------------------------------------------------
# Dynamic Flow Designer (v5)
#
# Changes from v4:
# - Results plots (flows, stoichiometry, residence time, mixer requirements)
#   are now Plotly, so they’re interactive (zoom, hover).
#
# Other features kept from v4:
# 1. Interactive DAG preview using Plotly with hover tooltips:
#    - Node type color
#    - Thick black border = reactive path
#    - Magenta border = control junction
#    - Split ratios annotated
#    - Legend
#
# 2. Pump feasibility constraints (max flow, max |dQ/dt|)
#
# 3. Defaults updated per your latest design:
#    In1, P → M0 → Plate → Gradient → R3 → D → Out
#    With In2 merging at Plate
#
# Run:
#   streamlit run streamlit_master_app_v5.py --server.address 0.0.0.0 --server.port 8501
# -----------------------------------------------------------------------------

import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ========== Safe math for user expressions ==========

SAFE = {
    "np": np,
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
    "abs": np.abs, "maximum": np.maximum, "minimum": np.minimum,
    "pi": math.pi, "e": math.e,
    "H": lambda x: 1.0 * (np.array(x) >= 0.0),  # Heaviside
    "clip": np.clip,
    "min": np.minimum, "max": np.maximum,
}

def eval_time_expr(expr: str,
                   t: np.ndarray,
                   extra: Dict[str, float],
                   clip_min=None,
                   clip_max=None) -> np.ndarray:
    """
    Evaluate expr on t, broadcast scalars, clamp, check finite.
    """
    expr = (expr or "").strip()
    loc = dict(SAFE); loc.update(extra); loc["t"] = t
    val = eval(expr, {"__builtins__": {}}, loc)
    v = np.asarray(val, dtype=float)
    if v.shape == ():  # scalar
        v = np.full_like(t, float(v), dtype=float)
    else:
        v = np.broadcast_to(v, t.shape).astype(float)

    if clip_min is not None or clip_max is not None:
        v = np.clip(
            v,
            -np.inf if clip_min is None else clip_min,
            np.inf if clip_max is None else clip_max
        )
    if not np.all(np.isfinite(v)):
        raise ValueError(f"Expression '{expr}' produced non-finite values.")
    return v


# ========== DAG helpers ==========

def topo_sort(node_ids: List[str], edges: List[Tuple[str, str]]) -> List[str]:
    """
    Topological sort; raises if cycle.
    """
    from collections import defaultdict, deque

    indeg = {n: 0 for n in node_ids}
    adj = defaultdict(list)
    for u, v in edges:
        if u not in indeg or v not in indeg:
            raise ValueError(f"Edge {u}->{v} references unknown node.")
        adj[u].append(v)
        indeg[v] += 1

    q = deque([n for n in node_ids if indeg[n] == 0])
    out = []
    while q:
        u = q.popleft()
        out.append(u)
        for w in adj[u]:
            indeg[w] -= 1
            if indeg[w] == 0:
                q.append(w)
    if len(out) != len(node_ids):
        raise ValueError("Graph has a cycle or disconnected mutual dependency.")
    return out


def compute_depths(node_ids: List[str],
                   edges: List[Tuple[str, str]]) -> Dict[str, int]:
    """
    Assign each node a 'depth' = longest path length from any source.
    We'll use this for x-position in the diagram.
    """
    from collections import defaultdict, deque
    parents = defaultdict(list)
    children = defaultdict(list)
    indeg = {n: 0 for n in node_ids}

    for u, v in edges:
        parents[v].append(u)
        children[u].append(v)
        indeg[v] += 1

    depth = {n: 0 for n in node_ids}
    q = deque([n for n in node_ids if indeg[n] == 0])

    while q:
        u = q.popleft()
        for w in children[u]:
            depth[w] = max(depth.get(w, 0), depth[u] + 1)
            indeg[w] -= 1
            if indeg[w] == 0:
                q.append(w)
    return depth


def sum_reactive_volume(nodes_df: pd.DataFrame,
                        path_nodes: List[str]) -> float:
    """
    Sum V_uL of nodes in the specified reactive path subset
    that have reactive == True.
    """
    id2row = {r["id"]: r for _, r in nodes_df.iterrows()}
    V_total = 0.0
    for nid in path_nodes:
        if nid not in id2row:
            continue
        row = id2row[nid]
        V = float(row.get("V_uL", 0.0) or 0.0)
        reactive_flag = bool(row.get("reactive", True))
        if reactive_flag and V > 0:
            V_total += V
    return V_total


# ========== Mixer/reactor dynamics + inverse math ==========

def integrate_cstr_fraction(t: np.ndarray,
                            Q_out: np.ndarray,
                            fA_in: np.ndarray,
                            V_hold: float) -> np.ndarray:
    """
    CSTR dynamics for species A fraction:
        dC_A/dt = (fA_in - C_A)/tauI,
        tauI = V_hold / Q_out.
    We use Heun RK2.
    """
    C = np.empty_like(t, dtype=float)
    C[0] = float(fA_in[0])  # assume it starts at inlet comp

    for i in range(len(t) - 1):
        dt = t[i+1] - t[i]
        tauI_i = max(V_hold / max(Q_out[i], 1e-12), 1e-9)
        k1 = (fA_in[i] - C[i]) / tauI_i
        C_pred = C[i] + dt * k1

        tauI_ip1 = max(V_hold / max(Q_out[i+1], 1e-12), 1e-9)
        k2 = (fA_in[i+1] - C_pred) / tauI_ip1

        C[i+1] = C[i] + 0.5 * dt * (k1 + k2)

    return C


def check_pump_constraints(t: np.ndarray,
                           flow: np.ndarray,
                           max_flow: float,
                           max_slew: float,
                           label: str) -> List[str]:
    """
    Check simple constraints for one pump:
    - flow(t) <= max_flow
    - |d(flow)/dt| <= max_slew
    Returns list of violation messages (strings). Empty if OK.
    """
    issues = []
    # flow limit
    if np.any(flow > max_flow + 1e-9):
        issues.append(f"{label} exceeds max flow ({max_flow} µL/min).")

    # slew limit
    dQdt = np.gradient(flow, t)  # µL/min^2
    if np.any(np.abs(dQdt) > max_slew + 1e-9):
        issues.append(f"{label} exceeds max |dQ/dt| ({max_slew} µL/min²).")

    # negativity check
    if np.any(flow < -1e-9):
        issues.append(f"{label} went negative.")

    return issues


def inverse_design_single_junction(
    t: np.ndarray,
    tau_target_expr: str,
    phi_expr: str,
    R_expr: str,
    V_reactive_total: float,
    V_M: float,
    extra_symbols: Dict[str, float],
    max_flow: float,
    max_slew: float,
):
    """
    Inverse design math for a single junction:
    - tau_target(t): residence time target for downstream reactive subset.
    - phi(t): fraction of downstream flow contributed by the 'aged' branch M.
    - R_target(t): desired A/(A+B) at the junction where M and the late-A branch meet.
    - V_reactive_total: total holdup in the reactive subset downstream.
    - V_M: holdup of the upstream mixer/coil M (A+B hold-up before junction).
    - max_flow, max_slew: feasibility constraints for pumps.
    """

    # (Targets)
    tau_target = eval_time_expr(tau_target_expr, t, extra_symbols, clip_min=1e-6)
    phi        = np.clip(eval_time_expr(phi_expr, t, extra_symbols), 1e-6, 1.0)
    R_target   = np.clip(eval_time_expr(R_expr, t, extra_symbols), 1e-6, 1.0-1e-6)

    # (1) downstream total flow from tau_target
    Q_reactive = V_reactive_total / tau_target  # µL/min

    # (2) split at the control junction
    Q_M_out  = phi * Q_reactive          # aged branch (A+B from M)
    Q_A_byp  = (1.0 - phi) * Q_reactive  # 'late A' branch, pure A

    # (3) required A fraction at M outlet to hit R_target after merge
    #     C_req_raw = 1 - (1-R)/phi
    C_req_raw = 1.0 - (1.0 - R_target)/np.maximum(phi, 1e-9)
    feas_phi  = phi >= (1.0 - R_target)
    C_req     = np.clip(C_req_raw, 0.0, 1.0)

    # (4) invert M's lag
    tauI_M = V_M / np.maximum(Q_M_out, 1e-12)
    dCdt   = np.gradient(C_req, t)
    fA_in_req_raw = C_req + tauI_M * dCdt
    fA_in_req     = np.clip(fA_in_req_raw, 0.0, 1.0)
    feas_fA_in_bounds = np.logical_and(
        fA_in_req_raw >= -1e-6,
        fA_in_req_raw <= 1.0 + 1e-6
    )

    # (5) convert inlet fraction to actual pump flows feeding M
    # upstream of M:
    #   In1 = pure A
    #   P   = pure B
    Q_In1 = fA_in_req * Q_M_out
    Q_P   = (1.0 - fA_in_req) * Q_M_out

    # late A branch:
    Q_In2 = Q_A_byp  # pure A added right at junction

    # (6) forward simulate M to verify actual ratio at junction
    fA_in_forward = np.divide(Q_In1, np.maximum(Q_In1 + Q_P, 1e-12))
    Q_M_out_forward = Q_In1 + Q_P  # should match Q_M_out

    C_A_out = integrate_cstr_fraction(
        t,
        Q_M_out_forward,
        fA_in_forward,
        V_hold=V_M
    )
    F_A_M = Q_M_out_forward * C_A_out
    F_B_M = Q_M_out_forward * (1.0 - C_A_out)

    R_achieved = (F_A_M + Q_In2) / np.maximum(F_A_M + Q_In2 + F_B_M, 1e-12)

    # (7) Feasibility: physical, dynamic, pump limits
    issues = []
    if np.any(~feas_phi):
        issues.append("phi(t) < 1 - R_target(t) somewhere (C_req would go negative).")
    if np.any(~feas_fA_in_bounds):
        issues.append("fA_in_req(t) leaves [0,1] pre-clipping -> M can't change that fast.")

    # pump constraints per source
    issues += check_pump_constraints(t, Q_In1, max_flow, max_slew, "In1 (A upstream of M)")
    issues += check_pump_constraints(t, Q_P,   max_flow, max_slew, "P (B upstream of M)")
    issues += check_pump_constraints(t, Q_In2, max_flow, max_slew, "In2 (late A branch)")

    # no negative flow after clipping check
    if np.any(Q_In1 < -1e-9) or np.any(Q_P < -1e-9) or np.any(Q_In2 < -1e-9):
        issues.append("One or more computed flows < 0. (Unphysical.)")

    return {
        "t": t,
        "tau_target": tau_target,
        "phi": phi,
        "R_target": R_target,
        "Q_reactive": Q_reactive,
        "Q_M_out": Q_M_out,
        "Q_In2": Q_In2,
        "C_req": C_req,
        "fA_in_req": fA_in_req,
        "Q_In1": Q_In1,
        "Q_P": Q_P,
        "R_achieved": R_achieved,
        "issues": issues,
    }


# ========== DAG diagram (Plotly, interactive hover) ==========

def compute_layout_coords(node_ids: List[str],
                          edges: List[Tuple[str, str]]) -> Dict[str, Tuple[float, float]]:
    """
    Node coordinates: x = depth (longest distance from any source),
    y = evenly spaced within that depth.
    """
    depth = compute_depths(node_ids, edges)

    # bucket by depth
    buckets = {}
    for nid in node_ids:
        d = depth[nid]
        buckets.setdefault(d, []).append(nid)

    coords = {}
    for d, bucket in buckets.items():
        # sort to make layout stable
        bucket_sorted = sorted(bucket)
        n = len(bucket_sorted)
        if n == 1:
            ys = [0.0]
        else:
            ys = np.linspace(0.0, 1.0, n)
        for y_val, nid in zip(ys, bucket_sorted):
            coords[nid] = (float(d), float(y_val))
    return coords


def build_splitter_ratio_map(edges_df: pd.DataFrame,
                             nodes_df: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    """
    Return {(u,v): ratio} for edges that originate at splitter nodes.
    """
    id2type = {str(r["id"]): str(r["type"]) for _, r in nodes_df.iterrows()}
    ratio_map = {}
    for _, row in edges_df.iterrows():
        u = str(row["from"])
        v = str(row["to"])
        if id2type.get(u, "") == "splitter":
            ratio_val = row.get("ratio")
            if pd.isna(ratio_val):
                ratio_val = None
            ratio_map[(u, v)] = ratio_val
    return ratio_map


def make_dag_figure_plotly(nodes_df: pd.DataFrame,
                           edges_df: pd.DataFrame,
                           reactive_path_nodes: List[str],
                           junction_node: str) -> go.Figure:
    """
    Create a Plotly figure for the DAG.
    """

    node_ids = nodes_df["id"].astype(str).tolist()
    edges = [(str(r["from"]), str(r["to"]))
             for _, r in edges_df.iterrows()
             if str(r["from"]) and str(r["to"])]

    coords = compute_layout_coords(node_ids, edges)
    ratio_map = build_splitter_ratio_map(edges_df, nodes_df)

    # color by type
    color_map = {
        "source":   "#4f6bed",  # blue
        "mixer":    "#ff9933",  # orange
        "reactor":  "#e74c3c",  # red
        "splitter": "#2ecc71",  # green
    }
    default_color = "#aaaaaa"

    # build edge traces (lines)
    edge_x = []
    edge_y = []
    annotations = []

    for (u, v) in edges:
        if u not in coords or v not in coords:
            continue
        x1, y1 = coords[u]
        x2, y2 = coords[v]

        # line segment
        edge_x.extend([x1, x2, None])
        edge_y.extend([y1, y2, None])

        # arrow-ish annotation
        annotations.append(
            dict(
                ax=x1, ay=y1,
                x=x2,  y=y2,
                xref='x', yref='y',
                axref='x', ayref='y',
                showarrow=True,
                arrowhead=3,
                arrowsize=1,
                arrowwidth=1.5,
                arrowcolor="black",
                standoff=2,
            )
        )

        # splitter ratio label
        r = ratio_map.get((u, v), None)
        if r is not None:
            xm = 0.5*(x1+x2)
            ym = 0.5*(y1+y2)
            annotations.append(
                dict(
                    x=xm, y=ym,
                    xref='x', yref='y',
                    text=f"{r:.2f}",
                    showarrow=False,
                    font=dict(size=10, color="black"),
                    align="center",
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=0.5,
                )
            )

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode='lines',
        line=dict(color='black', width=1.5),
        hoverinfo='none',
        showlegend=False,
    )

    # build node scatter
    x_nodes = []
    y_nodes = []
    text_nodes = []
    hover_nodes = []
    marker_colors = []
    marker_line_colors = []
    marker_line_widths = []

    id2row = {str(r["id"]): r for _, r in nodes_df.iterrows()}

    for nid in node_ids:
        row = id2row[nid]
        ntype = str(row["type"])
        V     = float(row.get("V_uL", 0.0) or 0.0)
        reactive_flag = bool(row.get("reactive", False))

        A_frac = row.get("species_A_frac", "")
        B_frac = row.get("species_B_frac", "")

        x, y = coords.get(nid, (0.0, 0.0))
        x_nodes.append(x)
        y_nodes.append(y)

        text_nodes.append(nid)

        hover_txt = [
            f"id: {nid}",
            f"type: {ntype}",
            f"V = {V:.2f} µL",
            f"reactive: {reactive_flag}",
        ]
        if ntype == "source":
            hover_txt.append(f"A_frac: {A_frac}")
            hover_txt.append(f"B_frac: {B_frac}")
        if nid == junction_node:
            hover_txt.append("JUNCTION: ratio control point")
        if nid in reactive_path_nodes:
            hover_txt.append("IN REACTIVE PATH")

        hover_nodes.append("<br>".join(str(s) for s in hover_txt))

        fill = color_map.get(ntype, default_color)

        outline_color = "black"
        outline_width = 1.5
        if nid in reactive_path_nodes:
            outline_color = "black"
            outline_width = 3.0
        if nid == junction_node:
            outline_color = "#ff00aa"  # magenta
            outline_width = 3.0

        marker_colors.append(fill)
        marker_line_colors.append(outline_color)
        marker_line_widths.append(outline_width)

    node_trace = go.Scatter(
        x=x_nodes,
        y=y_nodes,
        mode='markers+text',
        text=text_nodes,
        textposition='middle center',
        hoverinfo='text',
        hovertext=hover_nodes,
        showlegend=False,
        marker=dict(
            symbol='square',
            size=40,
            color=marker_colors,
            line=dict(
                color=marker_line_colors,
                width=marker_line_widths
            )
        ),
        textfont=dict(color='white', size=10),
    )

    # legend: dummy traces
    legend_traces = []
    legend_items = [
        ("source",   color_map["source"],   "Source"),
        ("mixer",    color_map["mixer"],    "Mixer / junction"),
        ("reactor",  color_map["reactor"],  "Reactor / hold-up"),
        ("splitter", color_map["splitter"], "Splitter"),
        ("reactive_path", "white",          "Reactive path node (thick black border)"),
        ("junction", "#ff00aa",             "Control junction (magenta border)"),
    ]

    for key, col, name in legend_items:
        if key in ("reactive_path","junction"):
            legend_traces.append(
                go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(
                        symbol='square',
                        size=16,
                        color='white',
                        line=dict(
                            color='black' if key=="reactive_path" else "#ff00aa",
                            width=3
                        )
                    ),
                    showlegend=True,
                    name=name,
                    hoverinfo='skip'
                )
            )
        else:
            legend_traces.append(
                go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(
                        symbol='square',
                        size=16,
                        color=col,
                        line=dict(color='black', width=1)
                    ),
                    showlegend=True,
                    name=name,
                    hoverinfo='skip'
                )
            )

    fig = go.Figure(data=[edge_trace, node_trace] + legend_traces)

    fig.update_layout(
        annotations=annotations,
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            scaleanchor="x",
            scaleratio=1.0,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(
            x=1.02,
            y=1.0,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=0.5,
        )
    )

    if len(x_nodes) > 0 and len(y_nodes) > 0:
        fig.update_xaxes(range=[min(x_nodes)-0.5, max(x_nodes)+0.5])
        fig.update_yaxes(range=[min(y_nodes)-0.5, max(y_nodes)+0.5])

    return fig


# ========== Plotly helpers for result plots ==========

def plot_flows_plotly(t,
                      Q_reactive,
                      Q_M_out,
                      Q_In2,
                      Q_In1,
                      Q_P,
                      max_flow):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=Q_reactive,
                             mode="lines",
                             name="Q_reactive (downstream)"))
    fig.add_trace(go.Scatter(x=t, y=Q_M_out,
                             mode="lines",
                             name="Q_M (A+B branch)"))
    fig.add_trace(go.Scatter(x=t, y=Q_In2,
                             mode="lines",
                             name="Q_In2 (late A)"))
    fig.add_trace(go.Scatter(x=t, y=Q_In1,
                             mode="lines",
                             line=dict(dash="dash"),
                             name="In1 (A upstream of M)"))
    fig.add_trace(go.Scatter(x=t, y=Q_P,
                             mode="lines",
                             line=dict(dash="dash"),
                             name="P (B upstream of M)"))
    # pump limit overlay
    fig.add_trace(go.Scatter(
        x=[t[0], t[-1]],
        y=[max_flow, max_flow],
        mode="lines",
        line=dict(color="gray", dash="dot"),
        name="max_flow limit",
    ))

    fig.update_layout(
        xaxis_title="time (min)",
        yaxis_title="flow (µL/min)",
        template="simple_white",
        legend=dict(font=dict(size=10)),
        margin=dict(l=30,r=30,t=30,b=30),
    )
    return fig


def plot_ratio_plotly(t, R_target, R_achieved):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=R_target,
                             mode="lines",
                             name="R_target(t) = A/(A+B)"))
    fig.add_trace(go.Scatter(x=t, y=R_achieved,
                             mode="lines",
                             line=dict(dash="dash"),
                             name="R_achieved(t)"))

    fig.update_yaxes(range=[-0.02, 1.02])

    fig.update_layout(
        xaxis_title="time (min)",
        yaxis_title="A/(A+B)",
        template="simple_white",
        legend=dict(font=dict(size=10)),
        margin=dict(l=30,r=30,t=30,b=30),
    )
    return fig


def plot_tau_vs_flow_plotly(t, tau_target, Q_reactive):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=tau_target,
                             mode="lines",
                             name="τ_target(t) [min]"))
    fig.add_trace(go.Scatter(x=t, y=Q_reactive,
                             mode="lines",
                             line=dict(dash="dash"),
                             name="Q_reactive(t) = V_reactive/τ_target"))

    fig.update_layout(
        xaxis_title="time (min)",
        yaxis_title="value",
        template="simple_white",
        legend=dict(font=dict(size=10)),
        margin=dict(l=30,r=30,t=30,b=30),
    )
    return fig


def plot_mixer_requirements_plotly(t, C_req, fA_in_req):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=C_req,
                             mode="lines",
                             name="C_req (A frac @ M outlet)"))
    fig.add_trace(go.Scatter(x=t, y=fA_in_req,
                             mode="lines",
                             line=dict(dash="dash"),
                             name="fA_in_req (A frac @ M inlet)"))

    fig.update_yaxes(range=[-0.1, 1.1])

    fig.update_layout(
        xaxis_title="time (min)",
        yaxis_title="fraction A",
        template="simple_white",
        legend=dict(font=dict(size=10)),
        margin=dict(l=30,r=30,t=30,b=30),
    )
    return fig


# ========== STREAMLIT APP ==========

st.set_page_config(
    page_title="Dynamic Flow Designer (v5)",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Dynamic Flow Designer (v5)")
st.markdown("""
You're designing a dynamic flow experiment.

**Workflow:**
1. Define your network (sources, mixers, reactors, splitters).
2. Pick which downstream region is counted as 'chemistry time' (reactive path).
3. Pick the control junction where you care about A:B stoichiometry.
4. Give us:
   - Target residence time profile τ_target(t) for that reactive zone,
   - Desired ratio A:(A+B) at the control junction,
   - Split policy φ(t) (how much of the total flow at that junction comes from the aged branch),
   - Physical volumes.
5. We compute required syringe pump programs.
6. We check feasibility (mixing limits and pump limits).
""")

# ---- Sidebar: global time + pump constraints ----
st.sidebar.header("Time grid")
t_end = st.sidebar.number_input("End time (min)", 1.0, 240.0, 24.0, 0.5)
N_pts = st.sidebar.slider("Number of time points", 300, 3000, 1001)
t = np.linspace(0.0, t_end, N_pts)

st.sidebar.header("Pump constraints")
max_flow = st.sidebar.number_input(
    "Max flow per source (µL/min)",
    min_value=0.0,
    max_value=1e6,
    value=500.0,
    step=1.0
)
max_slew = st.sidebar.number_input(
    "Max |dQ/dt| per source (µL/min²)",
    min_value=0.0,
    max_value=1e6,
    value=50.0,
    step=1.0
)

# ---- Section 1: DAG Builder ----
st.subheader("1️⃣ Build your flow DAG")

st.caption(
    "Nodes table:\n"
    "- `type`: source / mixer / reactor / splitter\n"
    "- `V_uL`: holdup volume (µL); 0 means ideal tee/no holdup.\n"
    "- `reactive`: if True, this node's volume is counted toward chemistry time.\n"
    "- `species_A_frac`, `species_B_frac`: for sources, what's inside that syringe."
)

default_nodes = pd.DataFrame([
    {"id":"In1","type":"source","V_uL":0.0,"reactive":True,"species_A_frac":1.0,"species_B_frac":0.0},
    {"id":"P",  "type":"source","V_uL":0.0,"reactive":True,"species_A_frac":0.0,"species_B_frac":1.0},
    {"id":"M0", "type":"mixer","V_uL":10.0,"reactive":True,"species_A_frac":"","species_B_frac":""},
    {"id":"In2","type":"source","V_uL":0.0,"reactive":True,"species_A_frac":1.0,"species_B_frac":0.0},
    {"id":"Plate", "type":"reactor","V_uL":100.0,"reactive":True,"species_A_frac":"","species_B_frac":""},
    {"id":"Gradient", "type":"reactor","V_uL":100.0,"reactive":True,"species_A_frac":"","species_B_frac":""},
    {"id":"R3",  "type":"mixer","V_uL":200.0,"reactive":True,"species_A_frac":"","species_B_frac":""},
    {"id":"D","type":"mixer","V_uL":50,"reactive":False,"species_A_frac":"","species_B_frac":""},
    {"id":"Out","type":"mixer","V_uL":0.0,"reactive":False,"species_A_frac":"","species_B_frac":""},
], dtype=object)

nodes_df = st.data_editor(
    default_nodes,
    num_rows="dynamic",
    use_container_width=True,
    key="nodes_editor"
)

st.caption(
    "Edges table:\n"
    "- Each row defines a directed connection `from` -> `to`.\n"
    "- For splitter nodes, fill `ratio` on outgoing edges (ratios out of that splitter should sum to 1).\n"
    "- Mixers/reactors must NOT have >1 child unless it's a splitter.\n"
)

default_edges = pd.DataFrame([
    {"from":"In1","to":"M0","ratio":np.nan},
    {"from":"P",  "to":"M0","ratio":np.nan},
    {"from":"M0","to":"Plate","ratio":np.nan},
    {"from":"In2","to":"Plate","ratio":np.nan},
    {"from":"Plate","to":"Gradient","ratio":np.nan},
    {"from":"Gradient","to":"R3", "ratio":np.nan},
    {"from":"R3","to":"D", "ratio":np.nan},
    {"from":"D","to":"Out","ratio":np.nan},
], dtype=object)

edges_df = st.data_editor(
    default_edges,
    num_rows="dynamic",
    use_container_width=True,
    key="edges_editor"
)

# DAG validation
node_ids = nodes_df["id"].astype(str).tolist()
edges_list = [
    (str(r["from"]), str(r["to"]))
    for _, r in edges_df.iterrows()
    if str(r["from"]) and str(r["to"])
]

try:
    order = topo_sort(node_ids, edges_list)
    st.success(f"✅ Graph is a DAG. Topological order: {order}")
except Exception as e:
    st.error(f"❌ DAG check failed: {e}")
    order = []


# ---- Section 2: Reactive path + τ_target ----
st.subheader("2️⃣ Reactive subset and residence-time target")

st.caption(
    "List (in downstream order) the nodes that define the chemistry zone where you "
    "care about residence time. We'll sum their holdup volumes if `reactive=True` "
    "to get V_reactive, and then interpret τ_target(t) as the desired residence time "
    "in that combined zone."
)

path_text = st.text_input(
    "Reactive path nodes (comma-separated, downstream order)",
    "Plate,Gradient,R3"
)

tau_expr = st.text_input(
    "τ_target(t) [min] (expression in t)",
    "4.0+0.4*t"
)

path_nodes = [p.strip() for p in path_text.split(",") if p.strip()]
V_reactive_total = sum_reactive_volume(nodes_df, path_nodes)
st.write(f"→ Computed V_reactive = {V_reactive_total:.3f} µL from that path.")


# ---- Section 3: Junction ratio control ----
st.subheader("3️⃣ Junction ratio control (A vs B)")

st.caption(
    "Pick the control junction node. At that node, two branches combine:\n"
    "- Branch M: already-mixed A+B (aged in some holdup volume V_M).\n"
    "- A_bypass: a late 'pure A' source.\n\n"
    "You define:\n"
    "- φ(t): what fraction of the downstream total flow should come from M.\n"
    "- R_target(t): desired A/(A+B) immediately after those branches merge.\n"
    "- V_M: holdup volume of the last mixer/coil that produced the A+B stream feeding that junction.\n"
)

junction_node = st.text_input(
    "Control junction node id (e.g. 'Plate')",
    "R2"
)

V_M = st.number_input(
    "Holdup volume V_M (µL) for the branch that arrives pre-mixed (A+B) at this junction",
    min_value=0.0,
    max_value=1e6,
    value=50.0,
    step=1.0
)

phi_expr = st.text_input(
    "φ(t) in (0,1] = fraction of total downstream flow contributed by the pre-mixed branch",
    "0.55"
)

R_expr = st.text_input(
    "R_target(t) in (0,1) = desired A/(A+B) at this junction",
    "0.8+0.15*sin(1/5*t)"
)


# ---- Section 4: DAG Preview (interactive) ----
st.subheader("4️⃣ DAG Preview (interactive)")

st.caption(
    "Hover a node to see:\n"
    "- volume (µL)\n"
    "- reactive flag (does it count toward chemistry time?)\n"
    "- source composition if it's a pump\n\n"
    "Thick black outline = part of your reactive path.\n"
    "Magenta outline = control junction node.\n"
    "Edge labels on splitters = split ratio.\n"
    "Legend is on the right."
)

try:
    fig_preview = make_dag_figure_plotly(
        nodes_df,
        edges_df,
        reactive_path_nodes=path_nodes,
        junction_node=junction_node
    )
    st.plotly_chart(fig_preview, use_container_width=True)
except Exception as e:
    st.error(f"Couldn't render DAG preview: {e}")


# ---- Section 5: Solve inverse design & check feasibility ----
st.subheader("5️⃣ Inverse design, pump schedules, feasibility")

st.caption(
    "We now do the inverse calculation:\n"
    " - Use τ_target(t) to get total downstream flow through your reactive path.\n"
    " - Use φ(t) to split that total between the 'aged A+B branch' and the 'late A' branch.\n"
    " - Use R_target(t) to solve how A-rich that aged branch's outlet must be.\n"
    " - Invert the mixer/coil holdup (V_M) to get In1(t) and P(t).\n"
    "Then we:\n"
    " - Forward-simulate the mixer to verify the achieved A:(A+B).\n"
    " - Check pump limits (max flow, max |dQ/dt|).\n"
)

if st.button("Solve inverse design"):
    try:
        results = inverse_design_single_junction(
            t=t,
            tau_target_expr=tau_expr,
            phi_expr=phi_expr,
            R_expr=R_expr,
            V_reactive_total=V_reactive_total,
            V_M=V_M,
            extra_symbols={"t_end": t_end},
            max_flow=max_flow,
            max_slew=max_slew,
        )

        tau_target   = results["tau_target"]
        phi_out      = results["phi"]
        R_target_out = results["R_target"]
        Q_reactive   = results["Q_reactive"]
        Q_M_out      = results["Q_M_out"]
        Q_In2        = results["Q_In2"]
        C_req        = results["C_req"]
        fA_in_req    = results["fA_in_req"]
        Q_In1        = results["Q_In1"]
        Q_P          = results["Q_P"]
        R_achieved   = results["R_achieved"]
        issues       = results["issues"]

        col1, col2 = st.columns([1.1, 1.0])

        with col1:
            st.subheader("Flows (interactive)")
            flows_fig = plot_flows_plotly(
                t,
                Q_reactive,
                Q_M_out,
                Q_In2,
                Q_In1,
                Q_P,
                max_flow,
            )
            st.plotly_chart(flows_fig, use_container_width=True)

        with col2:
            st.subheader("Stoichiometry at control junction")
            ratio_fig = plot_ratio_plotly(
                t,
                R_target_out,
                R_achieved
            )
            st.plotly_chart(ratio_fig, use_container_width=True)

        st.subheader("Residence time vs implied flow")
        tau_fig = plot_tau_vs_flow_plotly(
            t,
            tau_target,
            Q_reactive
        )
        st.plotly_chart(tau_fig, use_container_width=True)

        st.subheader("Mixer M requirements")
        mixer_fig = plot_mixer_requirements_plotly(
            t,
            C_req,
            fA_in_req
        )
        st.plotly_chart(mixer_fig, use_container_width=True)

        if len(issues) == 0:
            st.success("Feasible design over the full time horizon ✅ (dynamics + pump limits).")
        else:
            st.warning("Feasibility issues:\n- " + "\n- ".join(issues))

        st.subheader("📥 Download pump schedules")
        out_df = pd.DataFrame({
            "t_min": t,
            "tau_target_min": tau_target,
            "phi": phi_out,
            "R_target": R_target_out,
            "Q_reactive_uLmin": Q_reactive,
            "Q_M_out_uLmin": Q_M_out,
            "Q_In2_uLmin": Q_In2,
            "fA_in_req": fA_in_req,
            "Q_In1_uLmin": Q_In1,
            "Q_P_uLmin": Q_P,
            "R_achieved": R_achieved,
        })
        st.download_button(
            "Download CSV",
            out_df.to_csv(index=False).encode("utf-8"),
            "design_schedule.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(f"Error during inverse solve: {e}")




# # streamlit_inverse_controller_v1.py
# # -----------------------------------------------------------------------------
# # Inverse design (3-stream tutorial):
# # - User specifies:
# #     * Total reactive holdup V_reactive (µL) for the series after the In2 join
# #     * Target τ_total(t) for those reactive sections (linear "mapped" case: τ = V_total/Q)
# #     * Fraction φ(t) of reactive flow that comes from M0 (0<φ≤1), the rest (1-φ) from In2
# #     * Target junction ratio R(t) = In/(In+P) at the R2 entrance
# #     * M0 holdup V_M0 (µL)
# # - We compute:
# #     * Q_reactive(t) = V_reactive / τ_total_target(t)
# #     * Q_M0(t) = φ(t) Q_reactive(t),  Q_In2(t) = (1-φ) Q_reactive(t)
# #     * Required M0 outlet In fraction: C_req(t) = 1 - (1 - R(t))/φ(t)
# #     * Required inlet fraction into M0: f_in(t) = C_req(t) + τI_M0(t) * dC_req/dt,  τI_M0 = V_M0/Q_M0
# #     * Then In1(t) = f_in * Q_M0,  P(t) = (1 - f_in) * Q_M0
# # - We verify by forward-simulating the M0 CSTR and recomputing the achieved R(t).
# # -----------------------------------------------------------------------------

# import math
# from typing import Dict
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import streamlit as st

# # ---------------- Utilities ----------------
# SAFE = {
#     "np": np,
#     "sin": np.sin, "cos": np.cos, "tan": np.tan,
#     "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
#     "abs": np.abs, "maximum": np.maximum, "minimum": np.minimum,
#     "pi": math.pi, "e": math.e,
#     "H": lambda x: 1.0 * (np.array(x) >= 0.0),
#     "clip": np.clip,
#     "min": np.minimum, "max": np.maximum,
# }


# def eval_time_expr(expr: str, t: np.ndarray, extra: Dict[str, float], clip_min=None, clip_max=None) -> np.ndarray:
#     loc = dict(SAFE)
#     loc.update(extra)
#     loc["t"] = t
#     val = eval(expr, {"__builtins__": {}}, loc)
#     v = np.asarray(val, dtype=float)
#     if v.shape == ():
#         v = np.full_like(t, float(v), dtype=float)
#     else:
#         v = np.broadcast_to(v, t.shape).astype(float)
#     if clip_min is not None or clip_max is not None:
#         v = np.clip(v, -np.inf if clip_min is None else clip_min,
#                     np.inf if clip_max is None else clip_max)
#     if not np.all(np.isfinite(v)):
#         raise ValueError(f"Expression '{expr}' produced non-finite values.")
#     return v


# def integrate_cstr_fraction(t: np.ndarray, Q: np.ndarray, fin_in: np.ndarray, C0: float, V: float) -> np.ndarray:
#     C = np.empty_like(t, dtype=float)
#     C[0] = C0
#     for i in range(len(t)-1):
#         dt = t[i+1] - t[i]
#         tauI_i = max(V / max(Q[i], 1e-12), 1e-9)
#         k1 = (fin_in[i] - C[i]) / tauI_i
#         C_pred = C[i] + dt*k1
#         tauI_ip1 = max(V / max(Q[i+1], 1e-12), 1e-9)
#         k2 = (fin_in[i+1] - C_pred) / tauI_ip1
#         C[i+1] = C[i] + 0.5*dt*(k1 + k2)
#     return C


# # ---------------- UI ----------------
# st.set_page_config(
#     page_title="Inverse Residence-Time + Ratio Controller", page_icon="🧪", layout="wide")
# st.title("🧪 Inverse Design: τ_total & Junction Ratio → Source Flows")

# with st.expander("📘 Method summary"):
#     st.markdown(r"""
# We start from **targets** and solve for the **source flows**:
# 1. You specify **total reactive holdup** \(V_{\mathrm{reactive}}\) (sum of volumes in the reactive subset **after** the In2 join), and a target **mapped residence time** \(\tau_{\mathrm{target}}(t)\).  
#    In the linear case (no internal delay mapping), \(Q_{\mathrm{reactive}}(t)=V_{\mathrm{reactive}}/\tau_{\mathrm{target}}(t)\).
# 2. Choose a **split policy** \(\phi(t)\): the fraction of that reactive flow that comes from the **M0** branch (the rest comes from **In2**).
# 3. Choose a **junction composition target** \(R(t)=\frac{\text{In}}{\text{In}+\text{P}}\) at the entrance to R2.
# 4. The required **M0 outlet In-fraction** is \(C_{\mathrm{req}}(t)=1-\frac{1-R(t)}{\phi(t)}\) (feasible only if \(\phi(t)\ge 1-R(t)\)).
# 5. Accounting for **M0 dynamics** (CSTR with holdup \(V_{M0}\)), the inlet fraction must be  
#    \(f_{\mathrm{in}}(t)=C_{\mathrm{req}}(t)+\tau_{I,M0}(t)\,\frac{dC_{\mathrm{req}}}{dt}\), where \(\tau_{I,M0}(t)=V_{M0}/Q_{M0}(t)\).
# 6. Finally, \(Q_{\mathrm{In1}}(t)=f_{\mathrm{in}}(t)\,Q_{M0}(t)\), \(Q_{P}(t)=(1-f_{\mathrm{in}}(t))\,Q_{M0}(t)\), and \(Q_{In2}(t)=(1-\phi(t))\,Q_{\mathrm{reactive}}(t)\).
# We then **simulate forward** to verify that the achieved ratio matches the target (subject to feasibility).
# """)

# # Time grid
# st.sidebar.header("Time grid")
# t_end = st.sidebar.number_input("End time (min)", 1.0, 240.0, 24.0, 0.5)
# N = st.sidebar.slider("Points", 400, 3000, 1201)
# t = np.linspace(0.0, t_end, N)

# # Volumes
# st.sidebar.header("Volumes (µL)")
# V_reactive = st.sidebar.number_input(
#     "V_reactive (after In2 joins)", 1.0, 20000.0, 150.0, 1.0)
# V_M0 = st.sidebar.number_input(
#     "V_M0 (Mixer holdup before join)", 1.0, 10000.0, 50.0, 1.0)

# # Targets
# st.sidebar.header("Targets")
# tau_expr = st.sidebar.text_input(
#     "τ_target(t) in minutes", "lin = 3 + 2*(t>1)*(t-1)/(t_end-1); lin")
# phi_expr = st.sidebar.text_input("φ(t) in (0,1]", "0.7")
# R_expr = st.sidebar.text_input("R_target(t) in (0,1)", "0.6")

# # Evaluate targets
# extra = {"t_end": t_end}
# tau_target = eval_time_expr(tau_expr, t, extra, clip_min=1e-6)
# phi = np.clip(eval_time_expr(phi_expr, t, extra), 1e-6, 1.0)
# R = np.clip(eval_time_expr(R_expr,   t, extra), 1e-6, 1.0-1e-6)

# # Reactive flow from τ_target
# Q_reactive = V_reactive / tau_target  # µL/min

# # Split policy
# Q_M0 = phi * Q_reactive
# Q_In2 = (1.0 - phi) * Q_reactive

# # Required outlet fraction at M0 for target R
# C_req = 1.0 - (1.0 - R) / np.maximum(phi, 1e-9)

# # Feasibility checks
# feas_phi = phi >= (1.0 - R)  # otherwise C_req < 0
# C_req = np.clip(C_req, 0.0, 1.0)
# tauI_M0 = V_M0 / np.maximum(Q_M0, 1e-12)
# dCdt = np.gradient(C_req, t)
# f_in = C_req + tauI_M0 * dCdt

# feas_fin_lo = f_in >= -1e-6
# feas_fin_hi = f_in <= 1.0 + 1e-6
# f_in_clipped = np.clip(f_in, 0.0, 1.0)

# # Source flows from required f_in
# Q_In1 = f_in_clipped * Q_M0
# Q_P = (1.0 - f_in_clipped) * Q_M0

# # Forward simulate M0 to verify
# fin_in_forward = np.divide(Q_In1, np.maximum(Q_M0, 1e-12))
# COut = integrate_cstr_fraction(
#     t, Q_M0, fin_in_forward, float(fin_in_forward[0]), V_M0)
# I_M0 = Q_M0 * COut
# P_M0 = Q_M0 * (1.0 - COut)

# # Achieved junction ratio with computed In2
# R_achieved = (I_M0 + Q_In2) / np.maximum(I_M0 + Q_In2 + P_M0, 1e-12)

# # ---------------- Plots ----------------
# c1, c2 = st.columns([1.1, 1.0])
# with c1:
#     st.subheader("Reactive flow & split")
#     fig, ax = plt.subplots(figsize=(8, 3.0))
#     ax.plot(t, Q_reactive, lw=2, label="Q_reactive = Vreactive/τ_target")
#     ax.plot(t, Q_M0,      lw=2, label="Q_M0 = φ Q_reactive")
#     ax.plot(t, Q_In2,     lw=2, label="Q_In2 = (1-φ) Q_reactive")
#     ax.set_xlabel("time (min)")
#     ax.set_ylabel("flow (µL/min)")
#     ax.grid(True, alpha=0.3)
#     ax.legend()
#     st.pyplot(fig)

# with c2:
#     st.subheader(
#         "Required M0 outlet In-fraction C_req and inlet fraction f_in")
#     fig2, ax2 = plt.subplots(figsize=(8, 3.0))
#     ax2.plot(t, C_req, lw=2, label="C_req (M0 out)")
#     ax2.plot(t, f_in,  lw=1.5, ls="--", label="f_in raw (may violate [0,1])")
#     ax2.plot(t, f_in_clipped, lw=2, label="f_in clipped")
#     ax2.set_xlabel("time (min)")
#     ax2.set_ylabel("fraction")
#     ax2.grid(True, alpha=0.3)
#     ax2.legend()
#     # Feasibility shading
#     bad = ~(feas_phi & feas_fin_lo & feas_fin_hi)
#     if np.any(bad):
#         ax2.fill_between(t, 0, 1, where=bad, color="red",
#                          alpha=0.12, label="infeasible region")
#     st.pyplot(fig2)

# st.subheader("Forward verification: junction ratio")
# fig3, ax3 = plt.subplots(figsize=(9, 3.0))
# ax3.plot(t, R, lw=2, label="Target R(t)")
# ax3.plot(t, R_achieved, lw=2, ls="--", label="Achieved R(t)")
# ax3.set_xlabel("time (min)")
# ax3.set_ylabel("R = In/(In+P)")
# ax3.set_ylim(-0.02, 1.02)
# ax3.grid(True, alpha=0.3)
# ax3.legend()
# st.pyplot(fig3)

# st.subheader("Source flows (results)")
# fig4, ax4 = plt.subplots(figsize=(9, 3.0))
# ax4.plot(t, Q_In1, lw=2, label="In1(t)")
# ax4.plot(t, Q_P,   lw=2, label="P(t)")
# ax4.plot(t, Q_In2, lw=2, label="In2(t)")
# ax4.set_xlabel("time (min)")
# ax4.set_ylabel("flow (µL/min)")
# ax4.grid(True, alpha=0.3)
# ax4.legend()
# st.pyplot(fig4)

# # ---------------- Download ----------------
# st.subheader("📥 Download design schedule")
# df = pd.DataFrame({
#     "t_min": t,
#     "tau_target_min": tau_target,
#     "phi": phi,
#     "R_target": R,
#     "Q_reactive_uLmin": Q_reactive,
#     "Q_M0_uLmin": Q_M0,
#     "Q_In2_uLmin": Q_In2,
#     "C_req": C_req,
#     "f_in_raw": f_in,
#     "f_in_clipped": f_in_clipped,
#     "In1_uLmin": Q_In1,
#     "P_uLmin": Q_P,
#     "R_achieved": R_achieved,
# })
# st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
#                    "inverse_design_schedule.csv", "text/csv")

# # Feasibility messages
# issues = []
# if np.any(phi < (1.0 - R) - 1e-9):
#     issues.append("φ(t) < 1 - R(t) at some times (C_req would be negative).")
# if np.any(f_in < -1e-6) or np.any(f_in > 1.0 + 1e-6):
#     issues.append(
#         "Required f_in(t) falls outside [0,1] at some times (too fast a change for given V_M0 & φ).")
# if len(issues) == 0:
#     st.success("Feasible design for all times (within numeric tolerance).")
# else:
#     st.warning("Feasibility issues:\n- " + "\n- ".join(issues))
