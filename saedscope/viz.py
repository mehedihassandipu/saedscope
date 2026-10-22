"""Figures: interactive Plotly views for the app and a Matplotlib figure for export."""

from __future__ import annotations

import io

import numpy as np

from .crystal import fmt_index, ring_lines

PALETTE = ["#0F766E", "#C2410C", "#7C3AED", "#2563EB", "#BE185D"]


def display_image(img: np.ndarray, max_px: int = 900) -> tuple[np.ndarray, float]:
    """Log-scaled, percentile-clipped 8-bit image (downsampled if large). Returns (image, scale)."""
    im = np.log1p(np.clip(img, 0, None))
    lo, hi = np.percentile(im, [1, 99.7])
    im = np.clip((im - lo) / max(hi - lo, 1e-9), 0, 1)
    h, w = im.shape
    f = max(1, int(np.ceil(max(h, w) / max_px)))
    if f > 1:
        im = im[: h - h % f, : w - w % f].reshape(h // f, f, w // f, f).mean(axis=(1, 3))
    return (im * 255).astype(np.uint8), float(f)


def _circle(cy, cx, r, n=180):
    t = np.linspace(0, 2 * np.pi, n)
    return cx + r * np.cos(t), cy + r * np.sin(t)


def pattern_figure(res, show_overlay: bool = True, height: int = 560):
    import plotly.express as px
    import plotly.graph_objects as go

    im, f = display_image(res.image)
    fig = px.imshow(im, binary_string=True)
    cy, cx = res.centre[0] / f, res.centre[1] / f
    fig.add_trace(go.Scatter(x=[cx], y=[cy], mode="markers", marker=dict(symbol="cross-thin", size=14, line=dict(width=2, color="#F59E0B")),
                             name="centre", hovertext=[f"centre ({res.centre[1]:.1f}, {res.centre[0]:.1f}) px"], hoverinfo="text"))
    if show_overlay and res.ring_result is not None:
        rr = res.ring_result
        phase_color = {}
        for a in rr.assignments:
            col = "#9CA3AF"
            if a.phase:
                if a.phase not in phase_color:
                    phase_color[a.phase] = PALETTE[len(phase_color) % len(PALETTE)]
                col = phase_color[a.phase]
            x, y = _circle(cy, cx, a.radius_px / f)
            label = f"{a.hkl}" if a.hkl and a.ring_index < 8 else ""
            hover = f"ring {a.ring_index + 1}: d = {a.d_meas:.3f} Å" + (f"<br>{a.phase} ({a.hkl}), ref {a.d_ref:.3f} Å, err {a.error_pct:+.2f}%" if a.phase else "<br>unexplained")
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=col, width=1.6, dash="solid" if a.phase else "dot"),
                                     hovertext=hover, hoverinfo="text", showlegend=False))
            ang = -np.pi / 4 + 0.22 * (a.ring_index % 4)
            fig.add_trace(go.Scatter(x=[cx + a.radius_px / f * np.cos(ang)], y=[cy + a.radius_px / f * np.sin(ang)], mode="text",
                                     text=[label], textfont=dict(color=col, size=12), hoverinfo="skip", showlegend=False))
        for name, col in phase_color.items():
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=col, width=3), name=name))
    if show_overlay and res.spot_result is not None and res.spot_result.solutions:
        sol = res.spot_result.solutions[0]
        c0 = res.spot_result.centre
        ccy, ccx = c0[0] / f, c0[1] / f
        idx = [s for s in sol.indexed if s["hkl"]]
        un = [s for s in sol.indexed if not s["hkl"]]
        H, W = res.image.shape
        pred_un = [p for p in sol.predicted if not p["observed"]
                   and 0 <= c0[0] + p["dy"] < H and 0 <= c0[1] + p["dx"] < W]  # only where the detector could see them
        if pred_un:
            fig.add_trace(go.Scatter(x=[ccx + p["dx"] / f for p in pred_un], y=[ccy + p["dy"] / f for p in pred_un], mode="markers",
                                     marker=dict(symbol="circle-open", size=9, color="#60A5FA", line=dict(width=1)),
                                     hovertext=[f"predicted {p['hkl']} (not observed)" for p in pred_un], hoverinfo="text", name="predicted, not seen"))
        fig.add_trace(go.Scatter(x=[ccx + s["dx"] / f for s in idx], y=[ccy + s["dy"] / f for s in idx], mode="markers+text",
                                 marker=dict(symbol="circle-open", size=12, color=PALETTE[0], line=dict(width=2)),
                                 text=[s["hkl"] for s in idx], textposition="top center", textfont=dict(color="#5EEAD4", size=10),
                                 hovertext=[f"{s['hkl']}: d = {s['d_meas']:.3f} Å (ref {s['d_ref']:.3f})" + ("" if s["allowed"] else "<br>kinematically forbidden: double diffraction?") for s in idx],
                                 hoverinfo="text", name=f"{sol.phase.name} {fmt_index(sol.uvw_family, '[]')}"))
        if un:
            fig.add_trace(go.Scatter(x=[ccx + s["dx"] / f for s in un], y=[ccy + s["dy"] / f for s in un], mode="markers",
                                     marker=dict(symbol="x", size=8, color="#F87171"), hovertext=["unindexed spot"] * len(un), hoverinfo="text", name="unindexed"))
        fit = res.spot_result.fit
        if fit is not None:
            for vec, nm in ((fit.a, "g1"), (fit.b, "g2")):
                fig.add_annotation(x=ccx + vec[1] / f, y=ccy + vec[0] / f, ax=ccx, ay=ccy, xref="x", yref="y", axref="x", ayref="y",
                                   showarrow=True, arrowhead=3, arrowwidth=2, arrowcolor="#F59E0B", text="")
                fig.add_annotation(x=ccx + 1.18 * vec[1] / f, y=ccy + 1.18 * vec[0] / f, xref="x", yref="y", showarrow=False,
                                   text=nm, font=dict(color="#F59E0B", size=12))
    if show_overlay and res.amorphous and res.amorphous.get("halo_radius_px") and not res.amorphous.get("note"):
        x, y = _circle(cy, cx, res.amorphous["halo_radius_px"] / f)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color="#A78BFA", width=2, dash="dash"), name="amorphous halo",
                                 hovertext=f"halo d ≈ {res.amorphous.get('halo_d_A', '?')} Å", hoverinfo="text"))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=-0.02, x=0, font=dict(size=11)),
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      plot_bgcolor="black", paper_bgcolor="white", hovermode="closest", dragmode="pan")
    return fig


def zone_figure(res, height: int = 560):
    """Simulated zero-order Laue zone of the best solution, with observed spots filled."""
    import plotly.graph_objects as go

    sol = res.spot_result.solutions[0]
    pr = sol.predicted
    fig = go.Figure()
    obs = [p for p in pr if p["observed"]]
    nob = [p for p in pr if not p["observed"]]
    for group, filled, name in ((nob, False, "predicted, not observed"), (obs, True, "predicted and observed")):
        if not group:
            continue
        fig.add_trace(go.Scatter(
            x=[p["dx"] for p in group], y=[p["dy"] for p in group], mode="markers+text",
            marker=dict(size=[8 + 26 * p["rel_I"] ** 0.5 for p in group], color=PALETTE[0] if filled else "rgba(0,0,0,0)",
                        line=dict(color=PALETTE[0], width=1.5)),
            text=[p["hkl"] if (p["rel_I"] > 0.05 or filled) else "" for p in group], textposition="top center",
            textfont=dict(size=10, color="#374151"),
            hovertext=[f"{p['hkl']}  rel. intensity {p['rel_I']:.2f}" + ("" if p["allowed"] else " (forbidden)") for p in group],
            hoverinfo="text", name=name))
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", marker=dict(symbol="cross-thin", size=14, line=dict(width=2, color="#F59E0B")),
                             name="direct beam", hoverinfo="skip"))
    for hh, nm in ((sol.h1, "g1"), (sol.h2, "g2")):
        lab = fmt_index(hh)
        tgt = next((p for p in pr if p["hkl"] == lab), None)
        if tgt is not None:
            fig.add_annotation(x=tgt["dx"], y=tgt["dy"], ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                               arrowhead=3, arrowwidth=2, arrowcolor="#B45309", text="")
            fig.add_annotation(x=0.55 * tgt["dx"], y=0.55 * tgt["dy"], xref="x", yref="y", showarrow=False, text=nm,
                               font=dict(color="#B45309", size=12), bgcolor="rgba(255,255,255,0.8)")
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=36, b=10), plot_bgcolor="white",
                      title=dict(text=f"Simulated {fmt_index(sol.uvw_family, '[]')} zone of {sol.phase.name}", font=dict(size=14), x=0.01),
                      legend=dict(orientation="h", y=-0.02, x=0, font=dict(size=11)))
    fig.update_xaxes(visible=False, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(visible=False, autorange="reversed")
    return fig


def tilt_figure(res, item: dict, height: int | None = None, width_hint: int = 1040):
    """Measured pattern now, and what each competing answer predicts after the suggested tilt (same image frame)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from .synth import oriented_spots

    adv = res.advice or {}
    hyps = adv.get("hypotheses", [])
    sol = res.spot_result.solutions[0]
    c = sol.calib
    obs = [s for s in sol.indexed if s["hkl"]] or list(sol.indexed)  # indexed spots set the scale (outliers would shrink it)
    gx = np.array([s["dx"] * c for s in obs])
    gy = np.array([s["dy"] * c for s in obs])
    g_vis = float(np.max(np.hypot(gx, gy))) * 1.08 if len(obs) else 0.8
    by_name = {p.name: p for p in res.candidates}
    R = np.array(item["rotation"])
    ax = np.array(item["axis_image_xy"])
    n = 1 + len(hyps)
    titles = ["Now (measured)"]
    for h, p in zip(hyps, item["predictions"]):
        titles.append(f"If {p['phase']}: {p['zone'] if p.get('zone') else 'off-axis'}")
    fig = make_subplots(rows=1, cols=n, subplot_titles=titles, horizontal_spacing=0.02)
    L = g_vis * 1.02
    for col in range(1, n + 1):
        fig.add_trace(go.Scatter(x=[-L * ax[0], L * ax[0]], y=[-L * ax[1], L * ax[1]], mode="lines",
                                 line=dict(color="#B45309", width=1.5, dash="dash"), hoverinfo="skip", showlegend=False), row=1, col=col)
        fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", marker=dict(symbol="cross-thin", size=10, line=dict(width=2, color="#F59E0B")),
                                 hoverinfo="skip", showlegend=False), row=1, col=col)
    if len(obs):
        fig.add_trace(go.Scatter(x=gx, y=gy, mode="markers", marker=dict(size=7, color="#111827"),
                                 hovertext=[s["hkl"] or "unindexed" for s in obs], hoverinfo="text", showlegend=False), row=1, col=1)
    for k, h in enumerate(hyps):
        ph = by_name.get(h["phase"])
        if ph is None:
            continue
        U = R @ np.array(h["orientation"])
        xy, w, hkl, s = oriented_spots(ph, U, g_max=g_vis, voltage_kV=res.calibration.voltage_kV or 200.0)
        inside = np.hypot(xy[:, 0], xy[:, 1]) <= g_vis if len(xy) else []
        if len(xy):
            xy, w, hkl = xy[inside], w[inside], hkl[inside]
        fig.add_trace(go.Scatter(x=xy[:, 0] if len(xy) else [], y=xy[:, 1] if len(xy) else [], mode="markers",
                                 marker=dict(size=4 + 12 * np.sqrt(w) if len(w) else 4, color=PALETTE[k % len(PALETTE)]),
                                 hovertext=[fmt_index(tuple(int(v) for v in hh)) for hh in hkl], hoverinfo="text", showlegend=False),
                      row=1, col=k + 2)
    for col in range(1, n + 1):
        fig.update_xaxes(visible=False, range=[-L, L], scaleanchor=f"y{'' if col == 1 else col}", scaleratio=1, constrain="domain", row=1, col=col)
        fig.update_yaxes(visible=False, range=[L, -L], constrain="domain", row=1, col=col)
    fig.update_annotations(font=dict(size=12))
    if height is None:  # square panels
        height = int(min(420, width_hint / n * 0.98) + 46)
    fig.update_layout(height=height, margin=dict(l=6, r=6, t=40, b=6), plot_bgcolor="#F8FAFC", paper_bgcolor="white")
    return fig


def profile_figure(res, height: int = 560):
    import plotly.graph_objects as go

    p = res.profile
    c = res.calib_used
    fig = go.Figure()
    if c:
        x = p.r * c
        xlab = "g = 1/d (Å⁻¹)"
    else:
        x = p.r
        xlab = "radius (px)"
    fig.add_trace(go.Scatter(x=x, y=p.median, mode="lines", name="radial profile", line=dict(color="#111827", width=1.4)))
    fig.add_trace(go.Scatter(x=x, y=p.background, mode="lines", name="background", line=dict(color="#9CA3AF", width=1, dash="dot")))
    ymax = float(np.max(p.median)) if len(p.median) else 1
    rr = res.ring_result
    if rr and rr.primary and c:
        for k, ps in enumerate([rr.primary] + ([rr.secondary] if rr.secondary else [])):
            col = PALETTE[k]
            lines = [l for l in ring_lines(ps.phase, g_max=float(x.max()) * 1.02) if l.g <= x.max()]
            sx, sy, txt = [], [], []
            base = float(np.median(p.background)) if k == 0 else float(np.median(p.background)) * 0.6
            for l in lines:
                h = base + (ymax - base) * 0.6 * l.intensity / 100
                sx += [l.g, l.g, None]
                sy += [base * 0.5, h, None]
                txt += [f"{l.label}: d={l.d:.3f} Å, I={l.intensity:.0f}"] * 3
            fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines", line=dict(color=col, width=2), name=f"{ps.phase.name} (ref.)",
                                     hovertext=txt, hoverinfo="text"))
    for r in res.rings:
        xr = r.radius * c if c else r.radius
        fig.add_vline(x=xr, line=dict(color="#F59E0B" if r.kind == "sharp" else "#A78BFA", width=1, dash="dot"))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), xaxis_title=xlab, yaxis_title="intensity (log)",
                      yaxis_type="log", legend=dict(orientation="h", y=1.08, x=0, font=dict(size=11)), plot_bgcolor="white")
    fig.update_xaxes(showgrid=True, gridcolor="#F3F4F6")
    fig.update_yaxes(showgrid=True, gridcolor="#F3F4F6")
    return fig


def export_png(res, dpi: int = 160) -> bytes:
    """Publication-style two-panel figure (pattern with overlay + profile)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    im, f = display_image(res.image, max_px=1400)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw=dict(width_ratios=[1, 1.15]))
    ax = axes[0]
    ax.imshow(im, cmap="gray")
    ax.set_axis_off()
    cy, cx = res.centre[0] / f, res.centre[1] / f
    if res.ring_result:
        cols = {}
        for a in res.ring_result.assignments:
            col = "#9CA3AF"
            if a.phase:
                cols.setdefault(a.phase, PALETTE[len(cols) % len(PALETTE)])
                col = cols[a.phase]
            ax.add_patch(plt.Circle((cx, cy), a.radius_px / f, fill=False, color=col, lw=0.9, ls="-" if a.phase else ":"))
            if a.hkl and a.ring_index < 10:
                ang = -np.pi / 4 + 0.22 * (a.ring_index % 4)
                ax.text(cx + a.radius_px / f * np.cos(ang), cy + a.radius_px / f * np.sin(ang), a.hkl, color=col, fontsize=8)
    if res.spot_result and res.spot_result.solutions:
        sol = res.spot_result.solutions[0]
        c0 = res.spot_result.centre
        for s in sol.indexed:
            if s["hkl"]:
                x, y = c0[1] / f + s["dx"] / f, c0[0] / f + s["dy"] / f
                ax.add_patch(plt.Circle((x, y), 6, fill=False, color="#14B8A6", lw=1))
                ax.text(x, y - 9, s["hkl"], color="#5EEAD4", fontsize=6.5, ha="center")
        ax.set_title(f"{sol.phase.name}  ·  zone axis {fmt_index(sol.uvw_family, '[]')}", fontsize=10, pad=8)
    elif res.ring_result and res.ring_result.primary:
        ax.set_title(res.ring_result.primary.phase.name, fontsize=10)
    else:
        ax.set_title(res.classification.kind, fontsize=10)
    ax2 = axes[1]
    c = res.calib_used
    if res.spot_result and res.spot_result.solutions and res.classification.kind in ("single crystal", "mixed"):
        sol = res.spot_result.solutions[0]
        pr = sol.predicted
        ax2.scatter([p["dx"] for p in pr], [-p["dy"] for p in pr], s=[20 + 220 * p["rel_I"] for p in pr],
                    facecolors=["#0F766E" if p["observed"] else "none" for p in pr], edgecolors="#0F766E", lw=1)
        for p in pr:
            if p["rel_I"] > 0.08 or p["observed"]:
                ax2.text(p["dx"], -p["dy"] + 9, p["hkl"], fontsize=6.5, ha="center", color="#374151")
        ax2.scatter([0], [0], marker="+", s=80, color="#F59E0B")
        ax2.set_aspect("equal")
        ax2.set_title(f"Simulated {fmt_index(sol.uvw_family, '[]')} pattern (filled = observed)", fontsize=10, pad=8)
        ax2.set_axis_off()
    else:
        x = res.profile.r * c if c else res.profile.r
        ax2.semilogy(x, np.maximum(res.profile.median, 1e-3), color="#111827", lw=1)
        if res.ring_result and res.ring_result.primary and c:
            for k, ps in enumerate([res.ring_result.primary] + ([res.ring_result.secondary] if res.ring_result.secondary else [])):
                for l in ring_lines(ps.phase, g_max=float(x.max())):
                    ax2.axvline(l.g, color=PALETTE[k], lw=0.6 + 1.2 * l.intensity / 100, alpha=0.6)
                ax2.plot([], [], color=PALETTE[k], label=ps.phase.name)
            ax2.legend(fontsize=8, frameon=False)
        ax2.set_xlabel("g = 1/d (Å⁻¹)" if c else "radius (px)")
        ax2.set_ylabel("intensity")
        ax2.set_title("Radial profile with reference lines", fontsize=10, pad=8)
        ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    return buf.getvalue()
