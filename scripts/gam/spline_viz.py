# app.py
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.interpolate import make_lsq_spline, BSpline

st.set_page_config(page_title="Interactive LSQ B-splines", layout="wide")

st.title("Interactive B-splines with `make_lsq_spline`")
st.write(
    "Pick **degree**, **knot count**, and **coefficients** to see how weighting B-spline basis functions changes the spline shape. "
    "The bold curve is the spline produced by your coefficient vector."
)

# ----------------------------
# Sidebar controls
# ----------------------------
with st.sidebar:
    st.header("Spline settings")

    k = st.slider("Degree (k)", min_value=0, max_value=5, value=3, step=1)
    n_internal = st.slider(
        "Number of interior knots", min_value=0, max_value=20, value=8, step=1
    )

    n_grid = st.slider(
        "Plot resolution (x points)", min_value=50, max_value=1000, value=300, step=50
    )

    st.divider()
    st.header("Demo data (for baseline LSQ fit)")
    show_data = st.checkbox("Show sample noisy data + baseline fit", value=True)
    noise = st.slider("Noise level", 0.0, 0.5, 0.10, 0.01)
    seed = st.number_input(
        "Random seed", min_value=0, max_value=10_000, value=0, step=1
    )

    st.divider()
    st.header("Coefficient input")
    mode = st.radio(
        "How to set coefficients?", ["Preset pattern", "Manual (edit vector)"], index=0
    )

# ----------------------------
# Construct x, knots, and spline space
# ----------------------------
x = np.linspace(0.0, 1.0, int(n_grid))

# Open/clamped knot vector: repeat endpoints k+1 times
t_internal = (
    np.linspace(0.0, 1.0, n_internal + 2)[1:-1] if n_internal > 0 else np.array([])
)
t = np.r_[np.zeros(k + 1), t_internal, np.ones(k + 1)]

n_coeff = len(t) - k - 1
if n_coeff <= 0:
    st.error(
        "Invalid configuration: number of coefficients <= 0. Try increasing knots or lowering degree."
    )
    st.stop()

# ----------------------------
# Create demo data and baseline LSQ spline (using make_lsq_spline)
# ----------------------------
rng = np.random.default_rng(int(seed))
y_true = np.sin(2 * np.pi * x)
y_data = y_true + float(noise) * rng.normal(size=x.size)

# make_lsq_spline expects x in ascending order (it is), and returns a BSpline with coeffs
s_fit = make_lsq_spline(x, y_data, t, k)
c_fit = s_fit.c.copy()


def spline_from_coeffs(c: np.ndarray) -> BSpline:
    return BSpline(t, np.asarray(c, dtype=float), k, extrapolate=False)


# ----------------------------
# Build coefficient vector from UI
# ----------------------------
if mode == "Preset pattern":
    with st.sidebar:
        preset = st.selectbox(
            "Preset",
            [
                "Baseline LSQ fit",
                "Scaled baseline (0.4×)",
                "Single spike",
                "Smooth ramp",
                "Alternating signs",
                "Two bumps",
                "All zeros",
            ],
            index=0,
        )
        if preset == "Single spike":
            spike_i = st.slider("Spike index", 0, n_coeff - 1, n_coeff // 2, 1)
            spike_amp = st.slider("Spike amplitude", -5.0, 5.0, 1.0, 0.1)

    if preset == "Baseline LSQ fit":
        c = c_fit
    elif preset == "Scaled baseline (0.4×)":
        c = 0.4 * c_fit
    elif preset == "Single spike":
        c = np.zeros(n_coeff)
        c[int(spike_i)] = float(spike_amp)
    elif preset == "Smooth ramp":
        c = np.linspace(0.0, 1.0, n_coeff)
    elif preset == "Alternating signs":
        c = ((-1.0) ** np.arange(n_coeff)) * 0.8
    elif preset == "Two bumps":
        idx = np.arange(n_coeff)
        c = np.exp(-0.5 * ((idx - n_coeff * 0.30) / 1.2) ** 2) - 0.8 * np.exp(
            -0.5 * ((idx - n_coeff * 0.75) / 1.2) ** 2
        )
    elif preset == "All zeros":
        c = np.zeros(n_coeff)
    else:
        c = c_fit

else:
    # Manual editing:
    # Provide a text area with comma/space-separated numbers AND optional quick tools
    with st.sidebar:
        st.caption(f"Coefficient vector length = {n_coeff}")
        init = ", ".join([f"{v:.6g}" for v in c_fit])
        text = st.text_area(
            "Edit coefficients (comma or space separated)",
            value=init,
            height=160,
        )

        colA, colB = st.columns(2)
        with colA:
            zero_all = st.button("Set all to 0")
        with colB:
            use_fit = st.button("Reset to baseline fit")

        if zero_all:
            text = ", ".join(["0"] * n_coeff)
        if use_fit:
            text = init

        highlight_i = st.slider(
            "Highlight basis index", 0, n_coeff - 1, n_coeff // 2, 1
        )
        highlight_weight = st.checkbox(
            "Show highlighted basis * coefficient", value=True
        )

    # Parse
    raw = text.replace("\n", " ").replace(",", " ").split()
    try:
        vals = [float(v) for v in raw]
    except ValueError:
        st.error(
            "Could not parse coefficients. Please use numbers separated by commas/spaces."
        )
        st.stop()

    if len(vals) != n_coeff:
        st.error(f"Expected {n_coeff} coefficients, got {len(vals)}.")
        st.stop()

    c = np.array(vals, dtype=float)

# ----------------------------
# Compute spline and basis (for highlighting)
# ----------------------------
s_user = spline_from_coeffs(c)
y_user = s_user(x)

# Build basis functions on the same knot vector / degree
# (Each basis is spline with c[i]=1, others 0)
basis = []
for i in range(n_coeff):
    ci = np.zeros(n_coeff)
    ci[i] = 1.0
    basis.append(BSpline(t, ci, k, extrapolate=False)(x))
basis = np.array(basis)  # (n_coeff, len(x))

# Highlight controls (always available)
with st.sidebar:
    st.divider()
    st.header("Highlighting")
    hi_i = st.slider(
        "Basis index to highlight", 0, n_coeff - 1, n_coeff // 2, 1, key="hi_i_main"
    )
    show_hi_basis = st.checkbox("Show highlighted basis function", value=True)
    show_hi_contrib = st.checkbox(
        "Show highlighted contribution (c[i]*basis[i])", value=True
    )
    show_all_basis = st.checkbox("Show all basis functions (faint)", value=False)

# ----------------------------
# Layout: plots + coefficient readout
# ----------------------------
left, right = st.columns([2, 1], gap="large")

with left:
    st.subheader("Spline shape")
    fig, ax = plt.subplots(figsize=(10, 5))

    if show_all_basis:
        for i in range(n_coeff):
            ax.plot(x, basis[i], alpha=0.15, linewidth=1)

    if show_data:
        ax.plot(x, y_data, ".", ms=3, alpha=0.35, label="demo data")
        ax.plot(x, s_fit(x), linewidth=2, alpha=0.7, label="baseline LSQ fit")

    # Main spline (bold)
    ax.plot(x, y_user, linewidth=3, label="your spline (from coefficients)")

    # Highlight basis + contribution
    if show_hi_basis:
        ax.plot(
            x, basis[int(hi_i)], linewidth=2, linestyle="--", label=f"basis[{hi_i}]"
        )

    if show_hi_contrib:
        ax.plot(
            x,
            c[int(hi_i)] * basis[int(hi_i)],
            linewidth=2,
            linestyle=":",
            label=f"c[{hi_i}]*basis[{hi_i}]",
        )

    ax.set_xlabel("x")
    ax.set_ylabel("value")
    ax.set_title(f"Degree k={k} | interior knots={n_internal} | #coeff={n_coeff}")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.2)

    st.pyplot(fig, clear_figure=True)

with right:
    st.subheader("Coefficient vector")
    st.caption(
        "Each coefficient weights one B-spline basis function (local influence)."
    )

    # Simple stem plot of coefficients
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.stem(np.arange(n_coeff), c, basefmt=" ")
    ax2.plot([hi_i], [c[int(hi_i)]], marker="o", markersize=10)  # highlight point
    ax2.set_xlabel("index i")
    ax2.set_ylabel("c[i]")
    ax2.set_title(f"Coefficients (highlight i={hi_i})")
    ax2.grid(True, alpha=0.2)
    st.pyplot(fig2, clear_figure=True)

    st.divider()
    st.markdown("**Knot vector (t)**")
    st.code(np.array2string(t, precision=4, separator=", "))

    st.markdown("**Selected coefficient**")
    st.write({"i": int(hi_i), "c[i]": float(c[int(hi_i)])})

st.info("Run with:  streamlit run spline_viz.py")
