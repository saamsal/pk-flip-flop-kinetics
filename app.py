import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


LN2 = math.log(2.0)


@dataclass(frozen=True)
class PKMetrics:
    kel_h: float
    ka_h: float
    elimination_half_life_h: float
    absorption_half_life_h: float
    tmax_h: float
    cmax_mg_l: float
    auc_mg_h_l: float
    apparent_terminal_half_life_h: float
    time_90_absorbed_h: float


def concentration_extravascular(
    time_h: np.ndarray,
    dose_mg: float,
    bioavailability: float,
    volume_l: float,
    ka_h: float,
    kel_h: float,
) -> np.ndarray:
    """One-compartment extravascular concentration with first-order absorption."""
    time_h = np.asarray(time_h, dtype=float)
    if math.isclose(ka_h, kel_h, rel_tol=1e-7, abs_tol=1e-10):
        # Limit as ka approaches kel.
        return (bioavailability * dose_mg / volume_l) * ka_h * time_h * np.exp(-ka_h * time_h)

    scale = bioavailability * dose_mg * ka_h / (volume_l * (ka_h - kel_h))
    conc = scale * (np.exp(-kel_h * time_h) - np.exp(-ka_h * time_h))
    return np.maximum(conc, 0.0)


def pk_metrics(
    dose_mg: float,
    bioavailability: float,
    volume_l: float,
    ka_h: float,
    kel_h: float,
) -> PKMetrics:
    if math.isclose(ka_h, kel_h, rel_tol=1e-7, abs_tol=1e-10):
        tmax_h = 1.0 / ka_h
    else:
        tmax_h = math.log(ka_h / kel_h) / (ka_h - kel_h)

    cmax = float(
        concentration_extravascular(
            np.array([tmax_h]), dose_mg, bioavailability, volume_l, ka_h, kel_h
        )[0]
    )

    return PKMetrics(
        kel_h=kel_h,
        ka_h=ka_h,
        elimination_half_life_h=LN2 / kel_h,
        absorption_half_life_h=LN2 / ka_h,
        tmax_h=tmax_h,
        cmax_mg_l=cmax,
        auc_mg_h_l=bioavailability * dose_mg / (volume_l * kel_h),
        apparent_terminal_half_life_h=LN2 / min(ka_h, kel_h),
        time_90_absorbed_h=math.log(10.0) / ka_h,
    )


def format_duration(hours: float) -> str:
    if hours >= 48:
        return f"{hours / 24:.2f} days"
    return f"{hours:.2f} h"


def make_time_grid(abs_half_h: float, elim_half_1_h: float, elim_half_2_h: float, scenario: str) -> np.ndarray:
    slowest_half_life = max(abs_half_h, elim_half_1_h, elim_half_2_h)
    if scenario == "usual":
        end_h = max(24.0, 8.0 * slowest_half_life)
    else:
        end_h = max(7.0 * 24.0, 8.0 * slowest_half_life)
    return np.linspace(0.0, end_h, 1600)


def add_profile_trace(
    fig: go.Figure,
    x: np.ndarray,
    y: np.ndarray,
    label: str,
    dash: str,
    hover_time_unit: str,
) -> None:
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=label,
            line={"width": 3, "dash": dash},
            hovertemplate=f"Time: %{{x:.3g}} {hover_time_unit}<br>Concentration: %{{y:.4g}} mg/L<extra>{label}</extra>",
        )
    )


def make_concentration_plot(
    time_h: np.ndarray,
    profiles: list[tuple[str, np.ndarray, str]],
    title: str,
    x_in_days: bool,
    log_y: bool,
) -> go.Figure:
    x = time_h / 24.0 if x_in_days else time_h
    unit = "days" if x_in_days else "hours"
    fig = go.Figure()

    for label, concentration, dash in profiles:
        plotted = np.maximum(concentration, 1e-10) if log_y else concentration
        add_profile_trace(fig, x, plotted, label, dash, unit)

    fig.update_layout(
        title=title,
        xaxis_title=f"Time ({unit})",
        yaxis_title="Concentration (mg/L)",
        hovermode="x unified",
        legend_title_text="Elimination setting",
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
        height=430,
    )
    if log_y:
        fig.update_yaxes(type="log")
    return fig


def make_rates_plot(
    time_h: np.ndarray,
    dose_mg: float,
    bioavailability: float,
    volume_l: float,
    ka_h: float,
    kel_h: float,
    label: str,
    x_in_days: bool,
) -> go.Figure:
    # Input rate into the central compartment and elimination rate from it.
    absorption_rate = bioavailability * dose_mg * ka_h * np.exp(-ka_h * time_h)
    concentration = concentration_extravascular(time_h, dose_mg, bioavailability, volume_l, ka_h, kel_h)
    amount_central = concentration * volume_l
    elimination_rate = kel_h * amount_central

    x = time_h / 24.0 if x_in_days else time_h
    unit = "days" if x_in_days else "hours"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=absorption_rate,
            mode="lines",
            name="Absorption input rate",
            line={"width": 3},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=elimination_rate,
            mode="lines",
            name="Elimination output rate",
            line={"width": 3, "dash": "dash"},
        )
    )
    fig.update_layout(
        title=f"Input and output rates: {label}",
        xaxis_title=f"Time ({unit})",
        yaxis_title="Rate (mg/h)",
        hovermode="x unified",
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
        height=410,
    )
    return fig


def metric_table(metrics_a: PKMetrics, metrics_b: PKMetrics, label_a: str, label_b: str) -> pd.DataFrame:
    rows = [
        ("kel", metrics_a.kel_h, metrics_b.kel_h, "h^-1"),
        ("Elimination half-life", metrics_a.elimination_half_life_h, metrics_b.elimination_half_life_h, "h"),
        ("ka", metrics_a.ka_h, metrics_b.ka_h, "h^-1"),
        ("Absorption half-life", metrics_a.absorption_half_life_h, metrics_b.absorption_half_life_h, "h"),
        ("Tmax", metrics_a.tmax_h, metrics_b.tmax_h, "h"),
        ("Cmax", metrics_a.cmax_mg_l, metrics_b.cmax_mg_l, "mg/L"),
        ("AUC(0-inf)", metrics_a.auc_mg_h_l, metrics_b.auc_mg_h_l, "mg.h/L"),
        (
            "Apparent terminal half-life",
            metrics_a.apparent_terminal_half_life_h,
            metrics_b.apparent_terminal_half_life_h,
            "h",
        ),
        ("Time to 90% absorbed", metrics_a.time_90_absorbed_h, metrics_b.time_90_absorbed_h, "h"),
    ]
    return pd.DataFrame(rows, columns=["Metric", label_a, label_b, "Unit"])


def scenario_panel(
    scenario_name: str,
    explanation: str,
    absorption_half_life_h: float,
    elimination_half_life_a_h: float,
    elimination_half_life_b_h: float,
    dose_mg: float,
    bioavailability: float,
    volume_l: float,
    x_in_days: bool,
) -> None:
    ka_h = LN2 / absorption_half_life_h
    kel_a_h = LN2 / elimination_half_life_a_h
    kel_b_h = LN2 / elimination_half_life_b_h

    scenario_key = "flip" if x_in_days else "usual"
    time_h = make_time_grid(
        absorption_half_life_h,
        elimination_half_life_a_h,
        elimination_half_life_b_h,
        scenario_key,
    )

    label_a = f"Setting A: t1/2,elim {elimination_half_life_a_h:g} h (kel {kel_a_h:.4f} h^-1)"
    label_b = f"Setting B: t1/2,elim {elimination_half_life_b_h:g} h (kel {kel_b_h:.4f} h^-1)"

    conc_a = concentration_extravascular(time_h, dose_mg, bioavailability, volume_l, ka_h, kel_a_h)
    conc_b = concentration_extravascular(time_h, dose_mg, bioavailability, volume_l, ka_h, kel_b_h)

    metrics_a = pk_metrics(dose_mg, bioavailability, volume_l, ka_h, kel_a_h)
    metrics_b = pk_metrics(dose_mg, bioavailability, volume_l, ka_h, kel_b_h)

    st.subheader(scenario_name)
    st.caption(explanation)

    tab_linear, tab_log, tab_rates, tab_metrics = st.tabs(
        ["Linear concentration", "Semi-log concentration", "Input/output rates", "Metrics"]
    )

    with tab_linear:
        fig = make_concentration_plot(
            time_h,
            [(label_a, conc_a, "solid"), (label_b, conc_b, "dash")],
            f"{scenario_name}: concentration-time profiles",
            x_in_days=x_in_days,
            log_y=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_log:
        fig = make_concentration_plot(
            time_h,
            [(label_a, conc_a, "solid"), (label_b, conc_b, "dash")],
            f"{scenario_name}: terminal phase on a log scale",
            x_in_days=x_in_days,
            log_y=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_rates:
        selected = st.radio(
            "Rate plot setting",
            [label_a, label_b],
            horizontal=True,
            key=f"rate_setting_{scenario_key}",
        )
        selected_kel = kel_a_h if selected == label_a else kel_b_h
        fig = make_rates_plot(
            time_h,
            dose_mg,
            bioavailability,
            volume_l,
            ka_h,
            selected_kel,
            selected,
            x_in_days=x_in_days,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_metrics:
        table = metric_table(metrics_a, metrics_b, "Setting A", "Setting B")
        st.dataframe(
            table.style.format({"Setting A": "{:.4g}", "Setting B": "{:.4g}"}),
            use_container_width=True,
            hide_index=True,
        )

        ratio_a = ka_h / kel_a_h
        ratio_b = ka_h / kel_b_h
        st.markdown(
            f"**Rate-order check:** ka/kel = **{ratio_a:.2f}** for Setting A and "
            f"**{ratio_b:.2f}** for Setting B. "
            "Values above 1 favour the usual situation; values below 1 indicate flip-flop kinetics."
        )

        if ka_h > max(kel_a_h, kel_b_h):
            st.info(
                "Here ka is faster than kel. The slow terminal exponential is therefore governed by kel, "
                "so changing kel visibly changes the terminal slope and apparent terminal half-life."
            )
        elif ka_h < min(kel_a_h, kel_b_h):
            st.warning(
                "Here ka is slower than kel. The terminal exponential is governed by ka, so changing kel "
                "can change exposure, peak concentration and the early profile while the terminal slope remains absorption-limited."
            )
        else:
            st.warning(
                "The two elimination settings fall on opposite sides of ka. This is a transition region: "
                "one profile behaves conventionally and the other displays flip-flop kinetics."
            )


st.set_page_config(page_title="Changing kel: usual vs flip-flop PK", layout="wide")

st.title("Changing kel in usual and flip-flop pharmacokinetics")
st.markdown(
    "This teaching app uses a **one-compartment model with first-order absorption and elimination**. "
    "The same two elimination settings are applied to a usual absorption scenario and to a slow, flip-flop scenario."
)

with st.sidebar:
    st.header("Shared PK inputs")
    dose_mg = st.slider("Dose (mg)", min_value=10.0, max_value=1000.0, value=100.0, step=10.0)
    bioavailability = st.slider("Bioavailability, F", min_value=0.1, max_value=1.0, value=1.0, step=0.05)
    volume_l = st.slider("Apparent volume, V (L)", min_value=2.0, max_value=200.0, value=20.0, step=1.0)

    st.divider()
    st.header("Change elimination")
    elimination_half_life_a_h = st.slider(
        "Native/baseline elimination half-life (h)", min_value=0.5, max_value=48.0, value=0.5, step=0.5
    )
    elimination_half_life_b_h = st.slider(
        "Comparison elimination half-life (h)", min_value=0.5, max_value=48.0, value=18.0, step=0.5
    )
    st.caption(
        f"Setting A kel = {LN2 / elimination_half_life_a_h:.4f} h^-1\n\n"
        f"Setting B kel = {LN2 / elimination_half_life_b_h:.4f} h^-1"
    )

    st.divider()
    st.header("Absorption scenarios")
    usual_abs_half_h = st.slider(
        "Usual absorption half-life (h)", min_value=0.1, max_value=12.0, value=0.1, step=0.1
    )
    flip_abs_half_days = st.slider(
        "Flip-flop absorption half-life (days)", min_value=1.0, max_value=30.0, value=10.0, step=0.25
    )
    st.caption(
        f"Usual ka = {LN2 / usual_abs_half_h:.4f} h^-1\n\n"
        f"Flip-flop ka = {LN2 / (flip_abs_half_days * 24.0):.5f} h^-1"
    )

usual_col, flip_col = st.columns(2, gap="large")

with usual_col:
    scenario_panel(
        scenario_name="1) Usual PK: absorption faster than elimination",
        explanation="Default absorption half-life is 6 minutes, faster than the 30-minute native elimination half-life.",
        absorption_half_life_h=usual_abs_half_h,
        elimination_half_life_a_h=elimination_half_life_a_h,
        elimination_half_life_b_h=elimination_half_life_b_h,
        dose_mg=dose_mg,
        bioavailability=bioavailability,
        volume_l=volume_l,
        x_in_days=False,
    )

with flip_col:
    scenario_panel(
        scenario_name="2) Flip-flop PK: absorption slower than elimination",
        explanation="Default absorption half-life is 10 days, while the native intrinsic elimination half-life is 30 minutes.",
        absorption_half_life_h=flip_abs_half_days * 24.0,
        elimination_half_life_a_h=elimination_half_life_a_h,
        elimination_half_life_b_h=elimination_half_life_b_h,
        dose_mg=dose_mg,
        bioavailability=bioavailability,
        volume_l=volume_l,
        x_in_days=True,
    )

st.divider()
st.subheader("What the app is designed to show")
st.markdown(
    "- In **usual PK**, the terminal phase is elimination-limited: the apparent terminal half-life follows the elimination half-life.\n"
    "- In **flip-flop PK**, the terminal phase is absorption-limited: the apparent terminal half-life follows the absorption half-life, even when kel changes.\n"
    "- Changing kel still changes clearance and therefore AUC in both scenarios because, in this simple model, CL = kel x V and AUC = F x Dose / CL.\n"
    "- The model is intentionally simple: linear one-compartment disposition, a single first-order absorption process, and no lag time or distribution phase."
)

with st.expander("Model equations"):
    st.latex(r"C(t)=\frac{F\,Dose\,k_a}{V(k_a-k_{el})}\left(e^{-k_{el}t}-e^{-k_at}\right)")
    st.latex(r"t_{1/2,el}=\frac{\ln 2}{k_{el}},\qquad t_{1/2,abs}=\frac{\ln 2}{k_a}")
    st.latex(r"AUC_{0-\infty}=\frac{F\,Dose}{V\,k_{el}}=\frac{F\,Dose}{CL}")
    st.latex(r"\lambda_z=\min(k_a,k_{el})")
