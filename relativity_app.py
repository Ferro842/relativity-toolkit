import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import re
import math


# ==========================
# Fysische constanten
# ==========================
SPEED_OF_LIGHT = 299_792_458  # m/s


# ==========================
# Format helpers
# ==========================

def fmt(value: float, decimals: int = 2) -> str:
    """Nederlands getalformaat: punt → komma."""
    if math.isnan(value) or math.isinf(value):
        return "∞" if math.isinf(value) else "NaN"
    return f"{value:.{decimals}f}".replace(".", ",")


# ==========================
# Validatie
# ==========================

def assert_valid_beta(beta: float) -> None:
    if not np.isfinite(beta) or abs(beta) >= 1:
        raise ValueError(
            f"Ongeldige beta: {beta}. Beta moet |beta| < 1 zijn (snelheid < lichtsnelheid)."
        )


def assert_non_negative(name: str, value: float) -> None:
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"Ongeldige waarde voor {name}: {value}. Verwacht ≥ 0.")


# ==========================
# Fysica functies
# ==========================

def beta_from_velocity(v: float, c: float = SPEED_OF_LIGHT) -> float:
    if not np.isfinite(v):
        raise ValueError(f"Ongeldige snelheid v: {v}.")
    if not np.isfinite(c) or c <= 0:
        raise ValueError(f"Ongeldige lichtsnelheid c: {c}.")
    return v / c


def gamma_from_beta(beta: float) -> float:
    assert_valid_beta(beta)
    return 1.0 / np.sqrt(1.0 - beta * beta)


def time_dilation(proper_time: float, beta: float) -> float:
    assert_non_negative("eigen tijd", proper_time)
    return gamma_from_beta(beta) * proper_time


def length_contraction(proper_length: float, beta: float) -> float:
    assert_non_negative("eigen lengte", proper_length)
    return proper_length / gamma_from_beta(beta)


def relativistic_velocity_add(v1: float, v2: float, c: float = SPEED_OF_LIGHT) -> float:
    numerator = v1 + v2
    denominator = 1.0 + (v1 * v2) / (c * c)
    if abs(denominator) < 1e-15:
        raise ValueError("Deling door nul bij snelheidsoptelling.")
    return numerator / denominator


def transform_beta_to_frame(beta_obj: float, beta_ref: float) -> float:
    numerator = beta_obj - beta_ref
    denominator = 1.0 - beta_obj * beta_ref
    if abs(denominator) < 1e-9:
        return float("nan")
    return numerator / denominator


def doppler_factor(beta: float, approaching: bool = True) -> float:
    """
    Relativistisch Doppler-effect:
    Naderend:   f_obs = f_0 * sqrt((1+β)/(1-β))
    Verwijderend: f_obs = f_0 * sqrt((1-β)/(1+β))
    """
    assert_valid_beta(beta)
    if approaching:
        return math.sqrt((1 + beta) / (1 - beta))
    else:
        return math.sqrt((1 - beta) / (1 + beta))


def wavelength_shift(lambda0: float, beta: float, approaching: bool = True) -> float:
    """Verschoven golflengte λ_obs = λ_0 / doppler_factor (naderend = blauwverschuiving)."""
    factor = doppler_factor(beta, approaching)
    return lambda0 / factor


def mass_energy(mass_kg: float) -> float:
    """E = mc², geeft energie in Joule."""
    return mass_kg * SPEED_OF_LIGHT ** 2


def relativistic_kinetic_energy(mass_kg: float, beta: float) -> float:
    """Kinetische energie = (γ-1)mc²"""
    gamma = gamma_from_beta(beta)
    return (gamma - 1) * mass_kg * SPEED_OF_LIGHT ** 2


def rest_energy_mev(mass_kg: float) -> float:
    """Rustenergie in MeV (handig voor deeltjesfysica)."""
    joule = mass_energy(mass_kg)
    return joule / 1.602176634e-13


# ==========================
# Wavelength → RGB (voor Doppler-visualisatie)
# ==========================

def wavelength_to_rgb(wavelength_nm: float) -> tuple:
    """Converteert golflengte in nm naar een (R, G, B) tuple (0-1 bereik)."""
    wl = wavelength_nm
    if wl < 380:
        return (0.5, 0.0, 0.5)   # UV: paars
    elif wl < 440:
        r = -(wl - 440) / (440 - 380)
        return (r, 0.0, 1.0)
    elif wl < 490:
        g = (wl - 440) / (490 - 440)
        return (0.0, g, 1.0)
    elif wl < 510:
        b = -(wl - 510) / (510 - 490)
        return (0.0, 1.0, b)
    elif wl < 580:
        r = (wl - 510) / (580 - 510)
        return (r, 1.0, 0.0)
    elif wl < 645:
        g = -(wl - 645) / (645 - 580)
        return (1.0, g, 0.0)
    elif wl <= 780:
        return (1.0, 0.0, 0.0)
    else:
        return (0.5, 0.0, 0.0)   # IR: donkerrood


# ==========================
# Plot stijl
# ==========================

BG = "#0a0a0f"
BG2 = "#111118"
GRID = "#1e1e2e"
TEXT = "#e2e8f0"
ACCENT1 = "#60a5fa"   # blauw
ACCENT2 = "#f472b6"   # roze
ACCENT3 = "#34d399"   # groen
ACCENT4 = "#fbbf24"   # geel
ACCENT5 = "#a78bfa"   # paars


def apply_style(ax, fig=None):
    if fig:
        fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG2)
    for spine in ax.spines.values():
        spine.set_color("#2d2d3d")
        spine.set_linewidth(0.8)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    ax.grid(True, alpha=0.15, color=GRID, linestyle="-", linewidth=0.6)


# ==========================
# Streamlit configuratie
# ==========================

st.set_page_config(
    page_title="Spacetime Forge",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    .stApp {
        background-color: #0a0a0f;
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Alle tekst lichtgrijs */
    html, body, [class*="css"], .stMarkdown, p, span, label {
        color: #e2e8f0 !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #07070d;
        border-right: 1px solid #1e1e2e;
    }

    /* Knoppen — alle knoppen donker, geen geel */
    div.stButton > button {
        background: #1e1e2e !important;
        color: #e2e8f0 !important;
        border: 1px solid #2d2d4e !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        padding: 0.5rem 1.2rem !important;
        transition: all 0.2s ease !important;
        letter-spacing: 0.02em;
    }
    div.stButton > button:hover {
        background: #2d2d4e !important;
        border-color: #60a5fa !important;
        color: #60a5fa !important;
    }
    /* Primary knoppen blauw ipv geel */
    div.stButton > button[kind="primary"],
    div.stButton > button[data-testid="baseButton-primary"] {
        background: #1e3a5f !important;
        color: #60a5fa !important;
        border-color: #60a5fa !important;
    }
    div.stButton > button[kind="primary"]:hover,
    div.stButton > button[data-testid="baseButton-primary"]:hover {
        background: #243f6a !important;
        color: #93c5fd !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #0f0f1a;
        border-bottom: 1px solid #1e1e2e;
        gap: 0px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #94a3b8 !important;
        border-radius: 0;
        font-weight: 500;
        font-size: 0.85rem;
        padding: 0.6rem 1rem;
        border-bottom: 2px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent !important;
        color: #60a5fa !important;
        border-bottom: 2px solid #60a5fa !important;
    }

    /* Inputs */
    .stTextInput input, .stTextArea textarea {
        background-color: #111118 !important;
        border: 1px solid #2d2d4e !important;
        color: #e2e8f0 !important;
        border-radius: 6px !important;
        font-size: 0.9rem !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #60a5fa !important;
        box-shadow: 0 0 0 1px #60a5fa22 !important;
    }

    /* Sliders */
    .stSlider [data-baseweb="slider"] {
        padding: 0.2rem 0;
    }

    /* Alerts */
    .stSuccess {
        background-color: #0d2518 !important;
        border: 1px solid #166534 !important;
        border-radius: 8px !important;
        color: #86efac !important;
    }
    .stInfo {
        background-color: #0c1929 !important;
        border: 1px solid #1e40af !important;
        border-radius: 8px !important;
    }
    .stError {
        background-color: #1c0d0d !important;
        border: 1px solid #991b1b !important;
        border-radius: 8px !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: #111118 !important;
        border: 1px solid #1e1e2e !important;
        border-radius: 6px !important;
        color: #94a3b8 !important;
    }

    /* Tabellen */
    .stTable table {
        background-color: #111118;
        border: 1px solid #1e1e2e;
        border-radius: 8px;
        overflow: hidden;
    }
    .stTable thead tr th {
        background-color: #0f0f1a !important;
        color: #60a5fa !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stTable tbody tr td {
        color: #cbd5e1 !important;
        font-size: 0.85rem !important;
        border-color: #1e1e2e !important;
    }
    .stTable tbody tr:hover td {
        background-color: #1a1a2e !important;
    }

    /* Radio */
    .stRadio label {
        color: #94a3b8 !important;
        font-size: 0.88rem !important;
    }

    /* Metric cards */
    .metric-card {
        background: #111118;
        border: 1px solid #1e1e2e;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.4rem 0;
    }
    .metric-label {
        color: #64748b;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        color: #60a5fa;
        font-size: 1.4rem;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    .metric-unit {
        color: #475569;
        font-size: 0.8rem;
        margin-left: 0.3rem;
    }

    /* Hero section */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 600;
        letter-spacing: -0.04em;
        line-height: 1.1;
        color: #f1f5f9 !important;
    }
    .hero-sub {
        font-size: 1.05rem;
        color: #64748b !important;
        line-height: 1.7;
        max-width: 600px;
    }
    .highlight {
        color: #60a5fa !important;
    }

    /* Feature cards */
    .feature-card {
        background: #0f0f1a;
        border: 1px solid #1e1e2e;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        transition: border-color 0.2s;
    }
    .feature-card:hover {
        border-color: #2d2d4e;
    }
    .feature-icon {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .feature-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #e2e8f0 !important;
        margin-bottom: 0.3rem;
    }
    .feature-desc {
        font-size: 0.82rem;
        color: #475569 !important;
        line-height: 1.5;
    }

    /* Divider */
    hr {
        border: none;
        border-top: 1px solid #1e1e2e;
        margin: 1.5rem 0;
    }

    /* Select box */
    .stSelectbox > div > div {
        background-color: #111118 !important;
        border-color: #2d2d4e !important;
        color: #e2e8f0 !important;
    }

    /* iPad Pro optimalisatie — touch-vriendelijk */
    @media (max-width: 1200px) {
        .block-container { padding: 1rem 1.2rem !important; }
        .metric-value { font-size: 1.6rem !important; }
        .hero-title { font-size: 2.2rem !important; }
    }

    /* Grotere touch targets voor sliders en knoppen */
    .stSlider > div { padding: 0.6rem 0 !important; }
    .stSlider [data-baseweb="slider"] { height: 28px !important; }
    .stSlider [role="slider"] {
        width: 28px !important;
        height: 28px !important;
    }

    /* Sidebar knoppen groter voor touch */
    section[data-testid="stSidebar"] div.stButton > button {
        padding: 0.7rem 1rem !important;
        font-size: 0.9rem !important;
        min-height: 44px !important;
        text-align: left !important;
        justify-content: flex-start !important;
    }

    /* Presentatie navigatieknoppen groter */
    .pres-nav button {
        min-height: 52px !important;
        font-size: 1rem !important;
    }

    /* Matplotlib figuren scherper */
    .stImage img { image-rendering: -webkit-optimize-contrast; }

    /* Scrollbar stijl */
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: #0a0a0f; }
    ::-webkit-scrollbar-thumb { background: #1e1e2e; border-radius: 2px; }

    /* Active sidebar knop stijl */
    .sidebar-active > button {
        background: linear-gradient(135deg, #1e3a5f, #0f2a4a) !important;
        border-color: #60a5fa !important;
        color: #60a5fa !important;
    }

    /* Presentatie slide container */
    .slide-container {
        max-width: 900px;
        margin: 0 auto;
    }

    /* Grote presentatie metric */
    .pres-metric {
        background: #0f172a;
        border: 1px solid #1e3a5f;
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    .pres-metric-value {
        font-size: 2.8rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        line-height: 1;
        margin: 0.4rem 0;
    }
    .pres-metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #475569;
    }
    .pres-metric-sub {
        font-size: 0.85rem;
        color: #475569;
        margin-top: 0.3rem;
    }

    /* Slide titel groot */
    .slide-titel {
        font-size: 2.4rem;
        font-weight: 700;
        color: #f1f5f9;
        line-height: 1.15;
        letter-spacing: -0.03em;
        margin-bottom: 0.3rem;
    }
    .slide-sub {
        font-size: 1.1rem;
        color: #60a5fa;
        margin-bottom: 0.8rem;
        font-weight: 400;
    }
    .slide-intro {
        font-size: 1.05rem;
        color: #94a3b8;
        line-height: 1.75;
        max-width: 750px;
        margin-bottom: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)


inject_css()


# ==========================
# Navigatie via session state
# ==========================

if "page" not in st.session_state:
    st.session_state.page = "home"


# ==========================
# WELKOMSTPAGINA
# ==========================

def render_home():
    # Animatie: sterrenhemel via HTML/CSS
    st.markdown("""
    <style>
    @keyframes twinkle {
        0%, 100% { opacity: 0.2; }
        50% { opacity: 1; }
    }
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-8px); }
    }
    .stars {
        position: relative;
        width: 100%;
        height: 180px;
        overflow: hidden;
        border-radius: 16px;
        background: radial-gradient(ellipse at center, #0d1b3e 0%, #0a0a0f 70%);
        margin-bottom: 2rem;
        border: 1px solid #1e1e2e;
    }
    .star {
        position: absolute;
        background: white;
        border-radius: 50%;
        animation: twinkle linear infinite;
    }
    .rocket {
        position: absolute;
        font-size: 2.5rem;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -60%) rotate(45deg);
        animation: float 3s ease-in-out infinite;
        filter: drop-shadow(0 0 12px #60a5fa88);
    }
    .formula {
        position: absolute;
        color: #60a5fa44;
        font-family: Georgia, serif;
        font-size: 0.9rem;
        pointer-events: none;
    }
    </style>
    <div class="stars">
        <div class="star" style="width:2px;height:2px;top:15%;left:10%;animation-duration:2.1s;animation-delay:0s;"></div>
        <div class="star" style="width:1px;height:1px;top:25%;left:20%;animation-duration:3.2s;animation-delay:0.5s;"></div>
        <div class="star" style="width:2px;height:2px;top:10%;left:35%;animation-duration:1.8s;animation-delay:1s;"></div>
        <div class="star" style="width:1px;height:1px;top:40%;left:50%;animation-duration:2.7s;animation-delay:0.3s;"></div>
        <div class="star" style="width:2px;height:2px;top:20%;left:65%;animation-duration:2.3s;animation-delay:0.8s;"></div>
        <div class="star" style="width:1px;height:1px;top:35%;left:75%;animation-duration:3.5s;animation-delay:0.2s;"></div>
        <div class="star" style="width:2px;height:2px;top:12%;left:85%;animation-duration:1.9s;animation-delay:1.2s;"></div>
        <div class="star" style="width:1px;height:1px;top:55%;left:15%;animation-duration:2.8s;animation-delay:0.6s;"></div>
        <div class="star" style="width:2px;height:2px;top:60%;left:40%;animation-duration:2.2s;animation-delay:0.9s;"></div>
        <div class="star" style="width:1px;height:1px;top:70%;left:60%;animation-duration:3.1s;animation-delay:0.1s;"></div>
        <div class="star" style="width:2px;height:2px;top:45%;left:80%;animation-duration:2.6s;animation-delay:0.7s;"></div>
        <div class="star" style="width:1px;height:1px;top:80%;left:90%;animation-duration:1.7s;animation-delay:1.4s;"></div>
        <div class="formula" style="top:20%;left:5%;">γ = 1/√(1−β²)</div>
        <div class="formula" style="top:60%;left:70%;">E = mc²</div>
        <div class="formula" style="top:75%;left:15%;">t = γ·τ</div>
        <div class="rocket">🛸</div>
    </div>
    """, unsafe_allow_html=True)

    # Hero tekst
    st.markdown("""
    <div class="hero-title">Spacetime <span class="highlight">Forge</span></div>
    <div style="margin-top: 0.8rem;">
    <p class="hero-sub">
        Een interactieve toolkit voor de speciale relativiteitstheorie van Einstein.
        Verken tijdsvertraging, lengtecontractie, Doppler-verschuiving en massa-energie — 
        visueel, intuïtief en wiskundig correct.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_a:
        st.markdown("""
        <div style="background:#0f172a;border:1px solid #1e3a5f;border-radius:12px;padding:1.4rem;text-align:center;">
            <div style="font-size:2rem;margin-bottom:0.6rem;">🎓</div>
            <div style="color:#60a5fa;font-size:1rem;font-weight:600;margin-bottom:0.4rem;">Presentatie-modus</div>
            <div style="color:#475569;font-size:0.82rem;line-height:1.5;">Leg relativiteit stap voor stap uit aan iemand anders. Grote visuals, geen formules.</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🎓  Start presentatie", key="start_presentatie", width='stretch'):
            st.session_state.page = "presentatie"
            st.session_state.slide = 0
            st.rerun()

    with col_b:
        st.markdown("""
        <div style="background:#0f172a;border:1px solid #1e2a1e;border-radius:12px;padding:1.4rem;text-align:center;">
            <div style="font-size:2rem;margin-bottom:0.6rem;">🛸</div>
            <div style="color:#34d399;font-size:1rem;font-weight:600;margin-bottom:0.4rem;">Toolkit</div>
            <div style="color:#475569;font-size:0.82rem;line-height:1.5;">Alle modules, berekeningen en visualisaties. Voor dieper onderzoek en studie.</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🛸  Open toolkit", key="start_btn", width='stretch'):
            st.session_state.page = "toolkit"
            st.rerun()

    with col_c:
        st.markdown("""
        <div style="background:#0f172a;border:1px solid #2d1a1a;border-radius:12px;padding:1.4rem;text-align:center;">
            <div style="font-size:2rem;margin-bottom:0.6rem;">📖</div>
            <div style="color:#f472b6;font-size:1rem;font-weight:600;margin-bottom:0.4rem;">Formulekaart</div>
            <div style="color:#475569;font-size:0.82rem;line-height:1.5;">Alle formules van de speciale relativiteit op één overzichtelijke pagina.</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📖  Open formules", key="start_formules", width='stretch'):
            st.session_state.page = "toolkit"
            st.session_state.active_module = "📖 Formulekaart"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### Wat zit erin?")

    cols = st.columns(4)
    features = [
        ("⏱", "Lorentz & tijdsvertraging", "Bereken γ, tijdsvertraging en lengtecontractie bij elke snelheid."),
        ("➕", "Snelheidsoptelling", "Vergelijk klassieke en relativistische snelheidsoptelling."),
        ("🚀", "Scenario-simulator", "Beschrijf een scenario in gewone taal en krijg een berekening."),
        ("📐", "Minkowski-diagram", "Visualiseer worldlines en referentiekaders in ruimtetijd."),
        ("👯", "Tweelingparadox", "Ontdek waarom de reizende tweeling jonger terugkomt."),
        ("🌈", "Doppler-effect", "Rood- en blauwverschuiving bij bewegende lichtbronnen."),
        ("⚡", "E = mc²", "Bereken de enorme energie die in massa verborgen zit."),
        ("📖", "Formulekaart", "Alle formules overzichtelijk op één pagina."),
    ]

    for i, (icon, title, desc) in enumerate(features):
        col = cols[i % 4]
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""
    <div style="color: #334155; font-size: 0.78rem; text-align: center;">
        Alle berekeningen draaien lokaal · Gebouwd met Python & Streamlit · Speciale relativiteitstheorie (A. Einstein, 1905)
    </div>
    """, unsafe_allow_html=True)


# ==========================
# TOOLKIT PAGINA
# ==========================

def render_toolkit():
    # Sidebar navigatie
    alle_modules = [
        ("📚 Fase 1 — Basis SR", [
            "⏱ Lorentz & tijdsvertraging",
            "➕ Snelheidsoptelling",
            "🚀 Scenario-simulator",
            "📐 Minkowski-diagram",
            "👯 Tweelingparadox",
            "🌈 Doppler-effect",
            "⚡ E=mc²",
        ]),
        ("📗 Fase 1 — Epstein & Takeuchi", [
            "🔵 Epstein-cirkel",
            "💥 Lichtkegel",
            "🕐 Kloksynchronisatie",
            "💡 Lichtklok",
            "🚂 Gelijktijdigheid",
            "🏛 Galileï vs Einstein",
            "🔷 Spacetime-volume",
            "⚽ Sport-scenarios",
        ]),
        ("📙 Fase 2 — Taylor & Wheeler", [
            "🔄 Lorentz-transformaties",
            "💫 Relativistisch impuls",
        ]),
        ("📕 Fase 3-4 — Richting GR", [
            "🕳 Zwarte gaten",
            "🛰 Gravitationele tijdvertraging",
        ]),
        ("📖 Referentie", [
            "📖 Formulekaart",
        ]),
    ]

    flat_modules = [m for _, mods in alle_modules for m in mods]

    with st.sidebar:
        st.markdown("### 🛸 Spacetime Forge")
        if st.button("← Home", key="sidebar_home"):
            st.session_state.page = "home"
            st.rerun()
        st.markdown("---")

        if "active_module" not in st.session_state:
            st.session_state.active_module = flat_modules[0]

        for fase_naam, mods in alle_modules:
            st.markdown(f"**{fase_naam}**")
            for mod in mods:
                is_active = (st.session_state.active_module == mod)
                # Actieve knop krijgt een accent kleur
                label = f"→ {mod}" if is_active else mod
                if st.button(label, key=f"nav_{mod}",
                            width='stretch',
                            type="primary" if is_active else "secondary"):
                    st.session_state.active_module = mod
                    st.rerun()
            st.markdown("")

    active = st.session_state.active_module
    st.markdown(f"### {active}")
    st.markdown("---")

    # ==============================
    # TAB 1 – Lorentz & tijdsvertraging
    # ==============================
    if active == "⏱ Lorentz & tijdsvertraging":
        st.subheader("Lorentz-factor, tijdsvertraging en lengtecontractie")
        col_left, col_right = st.columns([1.1, 1.9])

        with col_left:
            st.markdown("""
**Formules**

$$γ = \\frac{1}{\\sqrt{1-β^2}}, \\quad β = \\frac{v}{c}$$

$$t = γ·τ \\quad (\\text{tijdsvertraging})$$

$$L = \\frac{L_0}{γ} \\quad (\\text{lengtecontractie})$$
            """)
            st.markdown("---")
            st.markdown("#### Invoer")

            beta_in = st.text_input("β (v/c)", value="", help="|β| < 1", key="t1_beta")
            vel_in = st.text_input("Snelheid v (m/s)", value="", help="Laat leeg als je β invult", key="t1_v")
            tau_in = st.text_input("Eigen tijd τ", value="10", key="t1_tau")
            L0_in = st.text_input("Eigen lengte L₀ (m)", value="100", key="t1_L0")
            go = st.button("Bereken", key="t1_go")

        with col_right:
            if go:
                try:
                    if beta_in.strip():
                        beta = float(beta_in.replace(",", "."))
                        v = beta * SPEED_OF_LIGHT
                    elif vel_in.strip():
                        v = float(vel_in.replace(",", "."))
                        beta = beta_from_velocity(v)
                    else:
                        st.error("Vul β of v in.")
                        st.stop()

                    if abs(beta) >= 1:
                        st.error("|β| ≥ 1 is niet fysisch toegestaan.")
                        st.stop()

                    gamma = gamma_from_beta(beta)
                    tau = float(tau_in.replace(",", ".")) if tau_in.strip() else None
                    L0 = float(L0_in.replace(",", ".")) if L0_in.strip() else None
                    t_dil = time_dilation(tau, beta) if tau is not None else None
                    L_con = length_contraction(L0, beta) if L0 is not None else None

                    # Metric cards
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">β = v/c</div>
                            <div class="metric-value">{fmt(beta, 5)}</div>
                        </div>""", unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Lorentz-factor γ</div>
                            <div class="metric-value">{fmt(gamma, 5)}</div>
                        </div>""", unsafe_allow_html=True)
                    with c3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Snelheid v</div>
                            <div class="metric-value">{v:.3e}<span class="metric-unit">m/s</span></div>
                        </div>""", unsafe_allow_html=True)

                    if tau is not None and t_dil is not None:
                        c4, c5 = st.columns(2)
                        with c4:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Eigen tijd τ</div>
                                <div class="metric-value">{fmt(tau, 2)}</div>
                            </div>""", unsafe_allow_html=True)
                        with c5:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Tijd in ander frame t = γ·τ</div>
                                <div class="metric-value" style="color:#f472b6">{fmt(t_dil, 2)}</div>
                            </div>""", unsafe_allow_html=True)

                    if L0 is not None and L_con is not None:
                        c6, c7 = st.columns(2)
                        with c6:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Eigen lengte L₀</div>
                                <div class="metric-value">{fmt(L0, 1)}<span class="metric-unit">m</span></div>
                            </div>""", unsafe_allow_html=True)
                        with c7:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Waargenomen lengte L = L₀/γ</div>
                                <div class="metric-value" style="color:#34d399">{fmt(L_con, 1)}<span class="metric-unit">m</span></div>
                            </div>""", unsafe_allow_html=True)

                    # Grafieken
                    st.markdown("---")
                    gc1, gc2 = st.columns(2)
                    betas = np.linspace(0, 0.999, 300)
                    gammas = 1.0 / np.sqrt(1 - betas**2)

                    with gc1:
                        fig, ax = plt.subplots(figsize=(4.5, 3))
                        apply_style(ax, fig)
                        ax.plot(betas, gammas, color=ACCENT1, linewidth=1.8)
                        ax.scatter([beta], [gamma], color=ACCENT4, s=60, zorder=5)
                        ax.axvline(beta, color=ACCENT4, alpha=0.25, linewidth=0.8, linestyle="--")
                        ax.set_xlabel("β = v/c")
                        ax.set_ylabel("γ")
                        ax.set_title("Lorentz-factor γ(β)")
                        st.pyplot(fig)
                        plt.close(fig)

                    with gc2:
                        if tau is not None:
                            t_vals = gammas * tau
                            fig2, ax2 = plt.subplots(figsize=(4.5, 3))
                            apply_style(ax2, fig2)
                            ax2.plot(betas, t_vals, color=ACCENT2, linewidth=1.8, label="t = γ·τ (ander frame)")
                            ax2.axhline(tau, color=ACCENT3, linewidth=1.2, linestyle="--", label=f"τ = {fmt(tau, 1)} (eigen tijd)")
                            ax2.scatter([beta], [t_dil], color=ACCENT4, s=60, zorder=5)
                            ax2.set_xlabel("β = v/c")
                            ax2.set_ylabel("Tijd")
                            ax2.set_title("Tijdsvertraging t(β)")
                            ax2.legend(fontsize=8, facecolor=BG2, edgecolor="#2d2d3d", labelcolor=TEXT)
                            st.pyplot(fig2)
                            plt.close(fig2)

                except Exception as e:
                    st.error(f"Fout: {e}")
            else:
                st.info("Vul de waarden in links en klik op **Bereken**.")

    # ==============================
    # TAB 2 – Snelheidsoptelling
    # ==============================
    if active == "➕ Snelheidsoptelling":
        st.subheader("Relativistische snelheidsoptelling")
        col_left, col_right = st.columns([1.1, 1.9])

        with col_left:
            st.markdown("""
**Klassiek:** $u = v_1 + v_2$

**Relativistisch:**
$$u = \\frac{v_1 + v_2}{1 + \\frac{v_1 v_2}{c^2}}$$

De uitkomst blijft altijd $|u| < c$.
            """)
            st.markdown("---")
            mode = st.radio("Invoer als", ["β (v/c)", "Snelheid v (m/s)"], key="t2_mode")
            if mode == "β (v/c)":
                b1 = st.text_input("β₁", value="0,6", key="t2_b1")
                b2 = st.text_input("β₂", value="0,6", key="t2_b2")
                v1_s, v2_s = "", ""
            else:
                v1_s = st.text_input("v₁ (m/s)", value="1,8e8", key="t2_v1")
                v2_s = st.text_input("v₂ (m/s)", value="1,8e8", key="t2_v2")
                b1, b2 = "", ""
            go2 = st.button("Bereken", key="t2_go")

        with col_right:
            if go2:
                try:
                    if mode == "β (v/c)":
                        beta1 = float(b1.replace(",", "."))
                        beta2 = float(b2.replace(",", "."))
                        v1 = beta1 * SPEED_OF_LIGHT
                        v2 = beta2 * SPEED_OF_LIGHT
                    else:
                        v1 = float(v1_s.replace(",", "."))
                        v2 = float(v2_s.replace(",", "."))
                        beta1 = v1 / SPEED_OF_LIGHT
                        beta2 = v2 / SPEED_OF_LIGHT

                    if abs(beta1) >= 1 or abs(beta2) >= 1:
                        st.error("|β| ≥ 1 is niet fysisch toegestaan.")
                        st.stop()

                    v_klas = v1 + v2
                    v_rel = relativistic_velocity_add(v1, v2)
                    bk = v_klas / SPEED_OF_LIGHT
                    br = v_rel / SPEED_OF_LIGHT

                    c1, c2, c3, c4 = st.columns(4)
                    cards = [
                        ("β₁", fmt(beta1, 4), ACCENT1),
                        ("β₂", fmt(beta2, 4), ACCENT1),
                        ("β klassiek", fmt(bk, 4), ACCENT2),
                        ("β relativistisch", fmt(br, 4), ACCENT3),
                    ]
                    for col, (label, val, color) in zip([c1, c2, c3, c4], cards):
                        with col:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">{label}</div>
                                <div class="metric-value" style="color:{color}">{val}</div>
                            </div>""", unsafe_allow_html=True)

                    betas2 = np.linspace(-0.999, 0.999, 500)
                    v2g = betas2 * SPEED_OF_LIGHT
                    bkl = (v1 + v2g) / SPEED_OF_LIGHT
                    brl = (v1 + v2g) / (1 + v1 * v2g / SPEED_OF_LIGHT**2) / SPEED_OF_LIGHT

                    fig, ax = plt.subplots(figsize=(6, 3.5))
                    apply_style(ax, fig)
                    ax.axhline(1, color="#ef4444", linewidth=0.8, linestyle=":", alpha=0.5)
                    ax.axhline(-1, color="#ef4444", linewidth=0.8, linestyle=":", alpha=0.5)
                    ax.fill_between(betas2, -1, 1, alpha=0.04, color=ACCENT3)
                    ax.plot(betas2, bkl, color=ACCENT2, linewidth=1.5, linestyle="--", label="Klassiek β₁ + β₂")
                    ax.plot(betas2, brl, color=ACCENT1, linewidth=2, label="Relativistisch")
                    ax.scatter([beta2], [bk], color=ACCENT2, s=50, zorder=5)
                    ax.scatter([beta2], [br], color=ACCENT4, s=60, zorder=6, label="Jouw invoer")
                    ax.set_xlabel("β₂")
                    ax.set_ylabel("β resultaat")
                    ax.set_title(f"Gecombineerde snelheid bij β₁ = {fmt(beta1, 3)}")
                    ax.set_ylim(-2, 2)
                    ax.legend(fontsize=8, facecolor=BG2, edgecolor="#2d2d3d", labelcolor=TEXT)
                    st.pyplot(fig)
                    plt.close(fig)

                except Exception as e:
                    st.error(f"Fout: {e}")
            else:
                st.info("Vul de waarden in en klik op **Bereken**.")

    # ==============================
    # TAB 3 – Scenario-simulator
    # ==============================
    if active == "🚀 Scenario-simulator":
        st.subheader("Scenario-simulator")
        col_left, col_right = st.columns([1.2, 1.8])

        with col_left:
            st.markdown("""
Beschrijf een scenario in gewone taal.
De tool herkent automatisch de snelheid (in **c**) en tijd (in **jaar**).
            """)
            st.markdown("**Voorbeelden:**")

            examples = [
                "Een raket beweegt met 0,8c ten opzichte van de aarde en reist 10 jaar.",
                "Een ruimteschip vliegt met 0,99c en reist 50 jaar.",
                "Een sonde beweegt met 0,5c en reist 20 jaar.",
                "Muonen bewegen met 0,998c en bereiken de aarde na 2,2 jaar.",
                "Een astronaut reist met 90% van de lichtsnelheid en reist 5 jaar.",
                "Een deeltje beweegt met 3/4 lichtsnelheid en reist 8 jaar.",
                "Een raket gaat met 60 procent van de lichtsnelheid en reist 15 jaar.",
            ]
            for ex in examples:
                if st.button(f"↗ {ex[:55]}…", key=f"ex_{ex[:20]}"):
                    st.session_state["scenario_text"] = ex

            st.markdown("---")
            default = st.session_state.get("scenario_text",
                "Een raket beweegt met 0,8c ten opzichte van de aarde en reist 10 jaar volgens de klok op aarde.")
            scenario = st.text_area("Scenario", value=default, height=120, key="t3_scenario")
            go3 = st.button("Analyseer & bereken", key="t3_go")

        with col_right:
            def parse_scenario(text):
                tl = text.lower()
                bm = re.search(r"(\d+[.,]?\d*)\s*c", tl)
                tm = re.search(r"(\d+[.,]?\d*)\s*(jaar|jaren)", tl)
                if not bm or not tm:
                    return None
                try:
                    bv = float(bm.group(1).replace(",", "."))
                    tv = float(tm.group(1).replace(",", "."))
                except ValueError:
                    return None
                if not (0 < bv < 1):
                    return None
                return {
                    "beta": bv,
                    "t_earth": tv,
                    "rest": "aarde" if "aarde" in tl else "rustframe",
                    "moving": "raket" if "raket" in tl else ("ruimteschip" if "ruimteschip" in tl else "object"),
                }

            if not go3:
                st.info("Kies een voorbeeld of beschrijf zelf een scenario, dan klik je op **Analyseer & bereken**.")
            else:
                s = parse_scenario(scenario)
                if s is None:
                    st.error("Kon het scenario niet herkennen. Zorg voor een snelheid (bijv. 0,8c) en een tijd (bijv. 10 jaar).")
                else:
                    beta = s["beta"]
                    t_earth = s["t_earth"]
                    gamma = gamma_from_beta(beta)
                    tau_rocket = t_earth / gamma
                    delta = t_earth - tau_rocket

                    eenheid = s.get("eenheid", "jaar")
                    st.success(
                        f"In **{fmt(t_earth, 1)} {eenheid}** op de {s['rest']} verstrijkt er in de "
                        f"{s['moving']} slechts **{fmt(tau_rocket, 2)} {eenheid}** eigen tijd. "
                        f"De {s['moving']} veroudert **{fmt(delta, 2)} {eenheid} minder**."
                    )

                    c1, c2, c3, c4 = st.columns(4)
                    for col, (label, val, color) in zip(
                        [c1, c2, c3, c4],
                        [
                            ("β", fmt(beta, 3), ACCENT1),
                            ("γ", fmt(gamma, 4), ACCENT1),
                            (f"Tijd ({s['rest']})", f"{fmt(t_earth, 1)} {eenheid}", ACCENT2),
                            (f"Eigen tijd ({s['moving']})", f"{fmt(tau_rocket, 2)} {eenheid}", ACCENT3),
                        ]
                    ):
                        with col:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">{label}</div>
                                <div class="metric-value" style="color:{color}">{val}</div>
                            </div>""", unsafe_allow_html=True)

                    # Bar chart
                    fig, ax = plt.subplots(figsize=(5, 3))
                    apply_style(ax, fig)
                    bars = ax.bar(
                        [f"Tijd\n({s['rest']})", f"Eigen tijd\n({s['moving']})"],
                        [t_earth, tau_rocket],
                        color=[ACCENT1, ACCENT3],
                        width=0.5,
                        edgecolor="#0a0a0f",
                    )
                    for bar, val in zip(bars, [t_earth, tau_rocket]):
                        ax.text(bar.get_x() + bar.get_width()/2,
                                bar.get_height() + 0.05 * t_earth,
                                f"{fmt(val, 2)} jr",
                                ha="center", va="bottom", color=TEXT, fontsize=10, fontweight="500")
                    ax.set_ylabel("Tijd (jaar)")
                    ax.set_ylim(0, t_earth * 1.3)
                    ax.set_title("Verstreken tijd vergeleken")
                    st.pyplot(fig)
                    plt.close(fig)

                    with st.expander("Stap-voor-stap"):
                        st.write(f"1. β = {fmt(beta, 3)} gehaald uit het scenario.")
                        st.write(f"2. γ = 1/√(1−β²) = {fmt(gamma, 5)}")
                        st.write(f"3. Coördinaattijd op {s['rest']}: t = {fmt(t_earth, 1)} jaar")
                        st.write(f"4. Eigen tijd in {s['moving']}: τ = t/γ = {fmt(t_earth, 1)}/{fmt(gamma, 3)} = {fmt(tau_rocket, 2)} jaar")
                        st.write(f"5. Verschil: {fmt(delta, 2)} jaar")

    # ==============================
    # TAB 4 – Minkowski-diagram
    # ==============================
    if active == "📐 Minkowski-diagram":
        st.markdown("## Minkowski-diagram")

        # Modus selector bovenaan
        mk_mode = st.radio(
            "Modus",
            ["🌐 Vrije worldlines", "🚂 Trein & bliksem (gelijktijdigheid)"],
            horizontal=True,
            key="mk_mode"
        )
        st.markdown("---")

        # ---- MODUS 1: Vrije worldlines ----
        if mk_mode == "🌐 Vrije worldlines":
            col_left, col_right = st.columns([1.2, 1.8])

            with col_left:
                st.markdown("""
**c = 1** eenheden:
- Verticaal: tijd t
- Horizontaal: ruimte x
- Lichtkegel: t = |x| (45°)
- Stippen op worldlines = eigen-tijd ticks (τ)
                """)
                t_max = st.slider("Tijdsas bereik", 2.0, 20.0, 5.0, 0.5, key="mk_tmax")
                num_obj = st.slider("Aantal objecten", 2, 4, 3, key="mk_nobj")

                default_labels = ["Aarde", "Raket", "Licht", "Sonde"]
                default_betas  = [0.0, 0.6, 0.999, -0.4]
                default_colors = [ACCENT1, ACCENT2, "#9ca3af", ACCENT3]

                worldlines = []
                st.markdown("**Objecten:**")
                for i in range(num_obj):
                    lbl = st.text_input(
                        f"Label {i+1}",
                        default_labels[i] if i < 4 else f"Obj {i+1}",
                        key=f"mk_l{i}"
                    )
                    bs = st.text_input(
                        f"β {i+1}",
                        str(default_betas[i]).replace(".", ",") if i < 4 else "0,0",
                        key=f"mk_b{i}"
                    )
                    try:
                        bv = max(-0.999, min(0.999, float(bs.replace(",", "."))))
                    except ValueError:
                        bv = 0.0
                    worldlines.append({
                        "label": lbl or f"Obj {i+1}",
                        "beta": bv,
                        "color": default_colors[i % 4]
                    })

                tau_max  = st.slider("Max eigen tijd τ", 1.0, 10.0, 4.0, 0.5, key="mk_tau")
                tau_step = st.slider("Δτ tick", 0.5, 3.0, 1.0, 0.5, key="mk_dtau")
                toon_sim = st.checkbox("Toon simultaneïteitslijnen", value=False, key="mk_sim")

            with col_right:
                ref_label = st.selectbox(
                    "Referentiekader",
                    [w["label"] for w in worldlines],
                    key="mk_ref"
                )
                ref_beta = next(w["beta"] for w in worldlines if w["label"] == ref_label)

                beta_prime = {}
                gamma_prime = {}
                for w in worldlines:
                    bp = transform_beta_to_frame(w["beta"], ref_beta)
                    beta_prime[w["label"]] = bp
                    gamma_prime[w["label"]] = gamma_from_beta(bp)

                fig, ax = plt.subplots(figsize=(6, 6))
                apply_style(ax, fig)

                t_vals = np.linspace(0, t_max, 300)
                # Lichtkegel
                ax.plot( t_vals, t_vals, "--", color="#4b5563", linewidth=1, alpha=0.5)
                ax.plot(-t_vals, t_vals, "--", color="#4b5563", linewidth=1, alpha=0.5)
                ax.fill_between( t_vals, t_vals, t_max, alpha=0.03, color=ACCENT1)
                ax.fill_between(-t_vals, t_vals, t_max, alpha=0.03, color=ACCENT1)
                ax.text( t_max * 0.55, t_max * 0.6, "licht", color="#4b5563", fontsize=7, rotation=45)
                ax.text(-t_max * 0.7, t_max * 0.6, "licht", color="#4b5563", fontsize=7, rotation=-45)

                for w in worldlines:
                    bp  = beta_prime[w["label"]]
                    gp  = gamma_prime[w["label"]]
                    is_ref = (w["label"] == ref_label)
                    ax.plot(bp * t_vals, t_vals,
                            color=w["color"],
                            linewidth=2.5 if is_ref else 1.5,
                            alpha=1.0 if is_ref else 0.8,
                            label=w["label"])

                    # Eigen-tijd ticks
                    ticks   = np.arange(tau_step, tau_max + 1e-9, tau_step)
                    t_ticks = ticks * gp
                    t_ticks = t_ticks[t_ticks <= t_max]
                    if len(t_ticks):
                        ax.scatter(bp * t_ticks, t_ticks,
                                   color=w["color"], s=30, zorder=5, alpha=0.8)

                    # Simultaneïteitslijnen (assen van het bewegende frame)
                    if toon_sim and not math.isnan(bp):
                        for tt in t_ticks:
                            x_sim = bp * tt
                            # Simultaneiteitslijn heeft helling β (in (x,t) plot helling = β)
                            x_range = np.linspace(x_sim - t_max * 0.6, x_sim + t_max * 0.6, 100)
                            t_range = tt + bp * (x_range - x_sim)
                            mask = (t_range >= 0) & (t_range <= t_max)
                            ax.plot(x_range[mask], t_range[mask],
                                    color=w["color"], linewidth=0.6,
                                    linestyle=":", alpha=0.35)

                ax.set_xlim(-t_max, t_max)
                ax.set_ylim(0, t_max)
                ax.set_aspect("equal")
                ax.set_xlabel("Ruimte x′")
                ax.set_ylabel("Tijd t′")
                ax.set_title(f"Frame van: {ref_label}")
                ax.legend(fontsize=8, facecolor=BG2, edgecolor="#2d2d3d",
                          labelcolor=TEXT, loc="upper left")
                st.pyplot(fig, width='stretch')
                plt.close(fig)

                st.markdown("**Snelheden in dit frame:**")
                rows = []
                for w in worldlines:
                    bp = beta_prime[w["label"]]
                    gp = gamma_prime[w["label"]]
                    rows.append({
                        "Object": w["label"],
                        "β (lab)": fmt(w["beta"], 3),
                        f"β′ (frame {ref_label})": fmt(bp, 3),
                        "γ′": fmt(gp, 3),
                    })
                st.table(rows)

        # ---- MODUS 2: Trein & bliksem ----
        else:
            st.markdown("""
Het **trein-en-bliksem experiment** in een Minkowski-diagram.
Versleep de sliders en zie direct hoe de simultaneïteitslijnen van beide frames
de bliksemevents anders verbinden.
            """)

            col_left, col_right = st.columns([1.1, 1.9])

            with col_left:
                beta_trein = st.slider(
                    "Treinsnelheid β", -0.95, 0.95, 0.6, 0.01,
                    key="tb_beta",
                    help="Positief = trein beweegt naar rechts"
                )
                L_half = st.slider(
                    "Halve treinlengte L (lichtjaar)", 1.0, 8.0, 3.0, 0.5,
                    key="tb_L"
                )
                toon_lichtsignalen = st.checkbox(
                    "Toon lichtsignalen vanaf bliksems", value=True,
                    key="tb_licht"
                )
                toon_trein_sim = st.checkbox(
                    "Toon simultaneïteitslijn treinframe", value=True,
                    key="tb_sim_trein"
                )
                toon_perron_sim = st.checkbox(
                    "Toon simultaneïteitslijn perronframe", value=True,
                    key="tb_sim_perron"
                )
                t_window = st.slider(
                    "Tijdsvenster", 2.0, 20.0, 10.0, 0.5,
                    key="tb_tmax"
                )

                # Uitleg berekening
                gamma_tb = gamma_from_beta(abs(beta_trein)) if abs(beta_trein) > 0 else 1.0

                # Event A (achter) en B (voor) in perronframe op t=0
                # trein beweegt met beta_trein, midden passeert x=0 op t=0
                # Achter: x_A = -L_half, voor: x_B = +L_half, t_A = t_B = 0
                xA_lab, tA_lab = -L_half, 0.0
                xB_lab, tB_lab =  L_half, 0.0

                # Lorentz naar treinframe (beta_ref = beta_trein)
                def lor(t, x, b):
                    if abs(b) >= 1:
                        return t, x
                    g = 1.0 / math.sqrt(1 - b*b)
                    return g * (t - b * x), g * (x - b * t)

                tA_tr, xA_tr = lor(tA_lab, xA_lab, beta_trein)
                tB_tr, xB_tr = lor(tB_lab, xB_lab, beta_trein)
                delta_t_trein_tb = tB_tr - tA_tr

                st.markdown("---")
                st.markdown("**Events in beide frames:**")
                st.markdown(f"""
<div class="metric-card">
    <div class="metric-label">Bliksem Achter (A) — perronframe</div>
    <div style="color:{ACCENT2};font-size:0.9rem;">t = {fmt(tA_lab,2)}, x = {fmt(xA_lab,2)}</div>
</div>
<div class="metric-card" style="margin-top:0.3rem;">
    <div class="metric-label">Bliksem Voor (B) — perronframe</div>
    <div style="color:{ACCENT3};font-size:0.9rem;">t = {fmt(tB_lab,2)}, x = {fmt(xB_lab,2)}</div>
</div>
<div class="metric-card" style="margin-top:0.3rem;">
    <div class="metric-label">Bliksem A — treinframe</div>
    <div style="color:{ACCENT2};font-size:0.9rem;">t′ = {fmt(tA_tr,3)}, x′ = {fmt(xA_tr,2)}</div>
</div>
<div class="metric-card" style="margin-top:0.3rem;">
    <div class="metric-label">Bliksem B — treinframe</div>
    <div style="color:{ACCENT3};font-size:0.9rem;">t′ = {fmt(tB_tr,3)}, x′ = {fmt(xB_tr,2)}</div>
</div>
<div class="metric-card" style="margin-top:0.3rem;border-color:#f472b644;">
    <div class="metric-label">Tijdsverschil in treinframe Δt′</div>
    <div style="color:#f472b6;font-size:1.1rem;font-weight:600;">{fmt(delta_t_trein_tb,4)} jaar</div>
    <div style="color:#475569;font-size:0.78rem;">
        {"B sloeg eerder in" if delta_t_trein_tb < 0 else "A sloeg eerder in"} vanuit de trein
    </div>
</div>
                """, unsafe_allow_html=True)

            with col_right:
                # Plot in twee naast elkaar: perronframe | treinframe
                fig, axes = plt.subplots(1, 2, figsize=(10, 7))
                fig.patch.set_facecolor(BG)

                frame_data = [
                    ("Perronframe", 0.0,        [(xA_lab, tA_lab), (xB_lab, tB_lab)]),
                    ("Treinframe",  beta_trein,  [(xA_tr,  tA_tr),  (xB_tr,  tB_tr)]),
                ]

                for ax_tb, (frame_title, beta_view, events_tb) in zip(axes, frame_data):
                    apply_style(ax_tb)
                    xlim = t_window * 1.1
                    ax_tb.set_xlim(-xlim, xlim)
                    ax_tb.set_ylim(-t_window * 0.3, t_window)
                    ax_tb.set_xlabel("Ruimte x (lichtjaar)")
                    ax_tb.set_ylabel("Tijd t (jaar)")
                    ax_tb.set_title(frame_title, color=TEXT, fontsize=10, fontweight="500")

                    # Lichtkegel
                    t_lk = np.linspace(-t_window * 0.3, t_window, 200)
                    ax_tb.plot( t_lk, t_lk, "--", color="#2d3748", linewidth=0.8, alpha=0.5)
                    ax_tb.plot(-t_lk, t_lk, "--", color="#2d3748", linewidth=0.8, alpha=0.5)

                    # ---- Worldlines ----
                    t_wl = np.linspace(-t_window * 0.3, t_window, 300)

                    # Perron waarnemer (stilstaand in perronframe)
                    beta_perron_in_view = transform_beta_to_frame(0.0, beta_view)
                    ax_tb.plot(
                        beta_perron_in_view * t_wl, t_wl,
                        color=ACCENT1, linewidth=2,
                        label="Perron (waarnemer)", alpha=0.9
                    )

                    # Trein midden
                    beta_trein_in_view = transform_beta_to_frame(beta_trein, beta_view)
                    ax_tb.plot(
                        beta_trein_in_view * t_wl, t_wl,
                        color=ACCENT4, linewidth=1.5,
                        linestyle="-", label="Trein (midden)", alpha=0.8
                    )

                    # Trein voor- en achterkant
                    # In lab: voor = x_B + beta_trein * t, achter = x_A + beta_trein * t
                    for x_start, kleur, naam in [
                        (xA_lab, ACCENT2, "Achterkant trein"),
                        (xB_lab, ACCENT3, "Voorkant trein"),
                    ]:
                        # In lab: x(t) = x_start + beta_trein * t
                        # In view frame: Lorentz
                        t_arr = np.linspace(-t_window * 0.3, t_window, 300)
                        x_arr = x_start + beta_trein * t_arr
                        t_view_arr = np.array([lor(tt, xx, beta_view)[0]
                                               for tt, xx in zip(t_arr, x_arr)])
                        x_view_arr = np.array([lor(tt, xx, beta_view)[1]
                                               for tt, xx in zip(t_arr, x_arr)])
                        mask = (t_view_arr >= -t_window * 0.3) & (t_view_arr <= t_window)
                        ax_tb.plot(x_view_arr[mask], t_view_arr[mask],
                                   color=kleur, linewidth=1.2,
                                   linestyle="--", alpha=0.6, label=naam)

                    # ---- Bliksemevents ----
                    ev_kleuren = [ACCENT2, ACCENT3]
                    ev_labels  = ["⚡ Bliksem Achter (A)", "⚡ Bliksem Voor (B)"]
                    evs_view   = [
                        lor(tA_lab, xA_lab, beta_view),
                        lor(tB_lab, xB_lab, beta_view),
                    ]

                    for (t_ev, x_ev), kleur, lbl_ev in zip(evs_view, ev_kleuren, ev_labels):
                        ax_tb.scatter([x_ev], [t_ev], color=kleur, s=120,
                                      zorder=10, marker="*")
                        ax_tb.text(x_ev + xlim * 0.03, t_ev + t_window * 0.02,
                                   lbl_ev, color=kleur, fontsize=7.5, zorder=11)

                        # Lichtsignalen vanaf event
                        if toon_lichtsignalen:
                            t_licht = np.linspace(t_ev, t_window, 100)
                            ax_tb.plot( (x_ev + (t_licht - t_ev)), t_licht,
                                       color=kleur, linewidth=0.8, alpha=0.3, linestyle="-")
                            ax_tb.plot( (x_ev - (t_licht - t_ev)), t_licht,
                                       color=kleur, linewidth=0.8, alpha=0.3, linestyle="-")

                    # ---- Simultaneïteitslijnen ----
                    x_sim_range = np.linspace(-xlim, xlim, 200)

                    # Perronframe: t = const (horizontaal)
                    if toon_perron_sim:
                        t_perron_sim = lor(tA_lab, xA_lab, beta_view)[0]
                        t_sim_perron_view = np.full_like(x_sim_range, t_perron_sim)
                        ax_tb.plot(x_sim_range, t_sim_perron_view,
                                   color=ACCENT1, linewidth=1.5,
                                   linestyle="-.", alpha=0.7,
                                   label=f"Simultaan (perron) t={fmt(tA_lab,2)}")

                    # Treinframe: simultaneiteitslijn heeft helling beta_trein in (x,t) plot
                    if toon_trein_sim:
                        # In treinframe: t′ = const voor event A
                        # Terug naar view: lijn door (xA_tr, tA_tr) met helling beta_trein
                        # in het lab: dt/dx = beta_trein  =>  t = tA_lab + beta_trein*(x - xA_lab)
                        # dan Lorentz naar view
                        x_lab_arr = np.linspace(-xlim * 2, xlim * 2, 300)
                        t_lab_arr = tA_lab + beta_trein * (x_lab_arr - xA_lab)
                        t_view_sim = np.array([lor(tt, xx, beta_view)[0]
                                               for tt, xx in zip(t_lab_arr, x_lab_arr)])
                        x_view_sim = np.array([lor(tt, xx, beta_view)[1]
                                               for tt, xx in zip(t_lab_arr, x_lab_arr)])
                        mask_s = (
                            (t_view_sim >= -t_window * 0.3) &
                            (t_view_sim <= t_window) &
                            (np.abs(x_view_sim) <= xlim)
                        )
                        ax_tb.plot(x_view_sim[mask_s], t_view_sim[mask_s],
                                   color=ACCENT4, linewidth=1.5,
                                   linestyle="-.", alpha=0.8,
                                   label=f"Simultaan (trein) t′={fmt(tA_tr,2)}")

                    ax_tb.axhline(0, color="#374151", linewidth=0.5, alpha=0.4)
                    ax_tb.axvline(0, color="#374151", linewidth=0.5, alpha=0.4)
                    ax_tb.legend(fontsize=7, facecolor=BG2, edgecolor="#2d2d3d",
                                 labelcolor=TEXT, loc="upper left",
                                 framealpha=0.9)

                plt.tight_layout(pad=1.5)
                st.pyplot(fig, width='stretch')
                plt.close(fig)

                with st.expander("Hoe lees je dit diagram?"):
                    st.markdown(f"""
**Sterretjes (★)** = de twee blikseminslagen (events A en B).

**Stippellijnen (-·-)** = simultaneïteitslijnen:
- **Blauwe lijn** = alle events die in het *perronframe* gelijktijdig zijn met de bliksems.
  In het perronframe is dit een horizontale lijn (t = 0).
- **Gele lijn** = alle events die in het *treinframe* gelijktijdig zijn met bliksem A.
  In het treinframe heeft deze lijn een helling omdat de trein beweegt.

**Wat zie je?**
- In het **perronframe**: de twee sterretjes liggen op dezelfde horizontale lijn → gelijktijdig.
- In het **treinframe**: de twee sterretjes liggen op *verschillende* t′-waarden →
  niet gelijktijdig. Het tijdsverschil is **{fmt(abs(delta_t_trein_tb), 3)} jaar**.

Sleep de β-slider en zie hoe de gele simultaneïteitslijn kantelt:
hoe sneller de trein, hoe meer de lijn kantelt en hoe groter het tijdsverschil.
                    """)

                with st.expander("Formule voor het tijdsverschil"):
                    st.markdown(r"""
Twee events gelijktijdig in frame S ($\Delta t = 0$, $\Delta x = 2L$):

$$\Delta t' = \gamma \left(\Delta t - \frac{v \cdot \Delta x}{c^2}\right)
= \gamma \cdot \frac{-v \cdot 2L}{c^2}$$

Hoe groter $v$ of $L$, hoe groter het tijdsverschil in het treinframe.
                    """)

    # ==============================
    # TAB 5 – Tweelingparadox
    # ==============================
    if active == "👯 Tweelingparadox":
        st.subheader("De tweelingparadox")
        col_left, col_right = st.columns([1.2, 1.8])

        with col_left:
            st.markdown("""
**Het scenario:**

Tweeling A blijft op aarde.
Tweeling B vertrekt met snelheid β, reist een afstand, keert om en komt terug.

Door tijdsvertraging is B **jonger** dan A bij terugkomst.

Het is *geen echte paradox*: B versnelt bij het omdraaien, waardoor de situatie niet symmetrisch is.
            """)
            st.markdown("---")
            beta_tw = st.slider("Reissnelheid β", 0.1, 0.999, 0.8, 0.001, format="%.3f", key="tw_beta")
            t_total = st.slider("Totale reistijd A (jaar)", 2.0, 100.0, 20.0, 1.0, key="tw_t")

        with col_right:
            gamma_tw = gamma_from_beta(beta_tw)
            tau_B = t_total / gamma_tw
            age_diff = t_total - tau_B
            dist = beta_tw * (t_total / 2)

            c1, c2, c3, c4 = st.columns(4)
            for col, (lbl, val, color) in zip([c1, c2, c3, c4], [
                ("β", fmt(beta_tw, 3), ACCENT1),
                ("γ", fmt(gamma_tw, 4), ACCENT1),
                ("Leeftijd A bij terugkomst", f"{fmt(t_total, 1)} jr", ACCENT2),
                ("Leeftijd B bij terugkomst", f"{fmt(tau_B, 2)} jr", ACCENT3),
            ]):
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{lbl}</div>
                        <div class="metric-value" style="color:{color}">{val}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-card" style="border-color:#f472b688;margin-top:0.5rem;">
                <div class="metric-label">Leeftijdsverschil bij terugkomst</div>
                <div class="metric-value" style="color:#f472b6">{fmt(age_diff, 2)} jaar</div>
                <div style="color:#475569;font-size:0.8rem;margin-top:0.3rem;">
                    Tweeling B is {fmt(age_diff, 2)} jaar jonger dan A.
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Worldline diagram
            st.markdown("---")
            fig, ax = plt.subplots(figsize=(5, 5))
            apply_style(ax, fig)

            # A: stilstaand
            ax.plot([0, 0], [0, t_total], color=ACCENT1, linewidth=2.5, label=f"A (thuis)")

            # B: heen en terug
            t_half = t_total / 2
            x_max = beta_tw * t_half
            ax.plot([0, x_max], [0, t_half], color=ACCENT2, linewidth=2, label=f"B (reizend)")
            ax.plot([x_max, 0], [t_half, t_total], color=ACCENT2, linewidth=2)

            # Eigen-tijd ticks op B
            tau_half = t_half / gamma_tw
            tick_interval = max(0.5, round(tau_half / 5, 1))
            for k in np.arange(tick_interval, tau_B + 1e-9, tick_interval):
                if k <= tau_B / 2:
                    t_tick = k * gamma_tw
                    x_tick = beta_tw * t_tick
                else:
                    k2 = k - tau_B / 2
                    t_tick = t_half + k2 * gamma_tw
                    x_tick = x_max - beta_tw * k2 * gamma_tw
                if 0 <= t_tick <= t_total:
                    ax.scatter([x_tick], [t_tick], color=ACCENT2, s=20, zorder=5, alpha=0.6)

            # Ontmoetingspunten
            ax.scatter([0], [0], color=ACCENT4, s=80, zorder=10)
            ax.scatter([0], [t_total], color=ACCENT4, s=80, zorder=10)
            ax.scatter([x_max], [t_half], color=ACCENT3, s=60, zorder=8)

            ax.annotate("Vertrek", (0, 0), (-0.8, 0.3), color="#94a3b8", fontsize=8)
            ax.annotate("Omdraaipunt", (x_max, t_half), (x_max + 0.1, t_half - 0.5), color=ACCENT3, fontsize=8)
            ax.annotate(f"Terugkomst\nA: {fmt(t_total,1)}jr\nB: {fmt(tau_B,2)}jr",
                        (0, t_total), (0.2, t_total - 1.5), color=ACCENT4, fontsize=8)

            ax.set_xlabel("Ruimte x (lichtjaar)")
            ax.set_ylabel("Tijd t (jaar)")
            ax.set_title("Tweelingparadox — worldlines")
            ax.legend(fontsize=8, facecolor=BG2, edgecolor="#2d2d3d", labelcolor=TEXT)
            st.pyplot(fig)
            plt.close(fig)

            with st.expander("Waarom is het geen paradox?"):
                st.markdown("""
De **schijnbare paradox** is: als beweging relatief is, kun je ook zeggen dat A beweegt
t.o.v. B. Dan zou A toch ook jonger moeten zijn?

Het verschil is dat **B versnelt** bij het omdraaien. B bevindt zich daarna in een
*ander inertialstelsel* dan bij de heenreis. A blijft de hele tijd in hetzelfde inertiaalstelsel.

De situatie is dus **niet symmetrisch**. Alleen B ondervindt echte versnelling (kracht).
Dat is meetbaar — B voelt een kracht bij het omdraaien, A niet.

De tijdsvertraging van B is reëel en experimenteel bevestigd, o.a. met atoomklokken
in vliegtuigen (Hafele-Keating experiment, 1971).
                """)

    # ==============================
    # TAB 6 – Doppler-effect
    # ==============================
    if active == "🌈 Doppler-effect":
        st.subheader("Relativistisch Doppler-effect")
        col_left, col_right = st.columns([1.2, 1.8])

        with col_left:
            st.markdown("""
**Formule:**

Naderend (blauwverschuiving):
$$f_{obs} = f_0 \\sqrt{\\frac{1+β}{1-β}}$$

Verwijderend (roodverschuiving):
$$f_{obs} = f_0 \\sqrt{\\frac{1-β}{1+β}}$$
            """)
            st.markdown("---")
            beta_dp = st.slider("β (v/c)", 0.001, 0.999, 0.3, 0.001, format="%.3f", key="dp_beta")
            lambda0 = st.slider("Golflengte lichtbron λ₀ (nm)", 380, 700, 550, 5, key="dp_lam")
            direction = st.radio("Richting", ["Naderend ↗", "Verwijderend ↙"], key="dp_dir")
            approaching = direction == "Naderend ↗"

        with col_right:
            df = doppler_factor(beta_dp, approaching)
            f_obs_ratio = df
            lambda_obs = wavelength_shift(lambda0, beta_dp, approaching)
            delta_lambda = lambda_obs - lambda0
            z = (lambda_obs - lambda0) / lambda0  # roodverschuiving parameter

            rgb0 = wavelength_to_rgb(lambda0)
            rgb_obs = wavelength_to_rgb(lambda_obs)

            c1, c2, c3, c4 = st.columns(4)
            for col, (lbl, val, color) in zip([c1, c2, c3, c4], [
                ("Doppler-factor", fmt(df, 4), ACCENT1),
                ("f_obs / f₀", fmt(f_obs_ratio, 4), ACCENT1),
                ("λ₀ (bron)", f"{lambda0} nm", ACCENT2),
                ("λ_obs (waarnemer)", f"{fmt(lambda_obs, 1)} nm", ACCENT3),
            ]):
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{lbl}</div>
                        <div class="metric-value" style="color:{color}">{val}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-card" style="margin-top:0.5rem;">
                <div class="metric-label">Roodverschuiving parameter z = Δλ/λ₀</div>
                <div class="metric-value" style="color:{'#f472b6' if not approaching else '#34d399'}">{fmt(z, 4)}</div>
                <div style="color:#475569;font-size:0.8rem;margin-top:0.2rem;">
                    {'Blauwverschuiving (z < 0): bron nadert' if approaching else 'Roodverschuiving (z > 0): bron verwijdert zich'}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Kleurvisualisatie
            st.markdown("---")
            st.markdown("**Kleuren van de lichtbundel:**")

            fig, axes = plt.subplots(2, 1, figsize=(6, 2))
            fig.patch.set_facecolor(BG)

            for ax_c, (rgb, wl, lbl) in zip(axes, [
                (rgb0, lambda0, f"Bron λ₀ = {lambda0} nm"),
                (rgb_obs, lambda_obs, f"Waargenomen λ = {fmt(lambda_obs, 1)} nm"),
            ]):
                ax_c.set_facecolor(BG2)
                ax_c.barh(0, 1, color=rgb, height=0.8)
                ax_c.set_xlim(0, 1)
                ax_c.set_ylim(-0.5, 0.5)
                ax_c.set_yticks([])
                ax_c.set_xticks([])
                ax_c.set_title(lbl, color=TEXT, fontsize=9, pad=3)
                for spine in ax_c.spines.values():
                    spine.set_color("#2d2d3d")

            plt.tight_layout(pad=0.5)
            st.pyplot(fig)
            plt.close(fig)

            # Grafiek: doppler factor als functie van beta
            fig2, ax2 = plt.subplots(figsize=(6, 3))
            apply_style(ax2, fig2)
            betas_dp = np.linspace(0.001, 0.999, 300)
            df_approach = [doppler_factor(b, True) for b in betas_dp]
            df_recede = [doppler_factor(b, False) for b in betas_dp]
            ax2.plot(betas_dp, df_approach, color=ACCENT3, linewidth=1.8, label="Naderend (blauwtrekking)")
            ax2.plot(betas_dp, df_recede, color=ACCENT2, linewidth=1.8, label="Verwijderend (roodverschuiving)")
            ax2.axhline(1, color="#4b5563", linewidth=0.8, linestyle=":")
            ax2.scatter([beta_dp], [df], color=ACCENT4, s=60, zorder=5)
            ax2.set_xlabel("β = v/c")
            ax2.set_ylabel("Doppler-factor")
            ax2.set_title("Relativistisch Doppler-effect")
            ax2.legend(fontsize=8, facecolor=BG2, edgecolor="#2d2d3d", labelcolor=TEXT)
            st.pyplot(fig2)
            plt.close(fig2)

            with st.expander("Meer uitleg"):
                st.markdown("""
Het **klassieke Doppler-effect** (geluid) hangt af van of de bron of waarnemer beweegt.

Bij **licht** is er geen medium, en de relativiteitstheorie zegt dat alleen de
*relatieve snelheid* telt. De formule verschilt daardoor van het klassieke geval.

**Roodverschuiving** (z > 0): lichtbron verwijdert zich → golflengtes worden langer.
**Blauwverschuiving** (z < 0): lichtbron nadert → golflengtes worden korter.

Dit effect is cruciaal in de kosmologie: de roodverschuiving van verre sterrenstelsels
bewijst dat het universum uitdijt (Hubble, 1929).
                """)

    # ==============================
    # TAB 7 – E = mc²
    # ==============================
    if active == "⚡ E=mc²":
        st.subheader("E = mc² — massa-energie equivalentie")
        col_left, col_right = st.columns([1.2, 1.8])

        with col_left:
            st.markdown("""
**Rustenergie:**
$$E_0 = mc^2$$

**Totale energie:**
$$E = γmc^2$$

**Kinetische energie (relativistisch):**
$$E_k = (γ-1)mc^2$$

**Klassieke kinetische energie (ter vergelijking):**
$$E_{k,klas} = \\frac{1}{2}mv^2$$
            """)
            st.markdown("---")
            mass_input = st.text_input("Massa m (kg)", value="1,0", key="emc_m")
            beta_emc = st.slider("Snelheid β", 0.0, 0.999, 0.5, 0.001, format="%.3f", key="emc_beta")

            st.markdown("**Bekende massa's:**")
            presets = {
                "Proton (1,67×10⁻²⁷ kg)": "1.67e-27",
                "Elektron (9,11×10⁻³¹ kg)": "9.11e-31",
                "1 gram": "0.001",
                "1 kg": "1.0",
                "Auto (~1500 kg)": "1500",
            }
            for name, val in presets.items():
                if st.button(name, key=f"preset_{name}"):
                    st.session_state["emc_mass_val"] = val

        with col_right:
            try:
                mass_val = st.session_state.get("emc_mass_val", None)
                if mass_val:
                    mass_kg = float(mass_val)
                else:
                    mass_kg = float(mass_input.replace(",", "."))

                gamma_emc = gamma_from_beta(beta_emc)
                E0 = mass_energy(mass_kg)
                Ek_rel = relativistic_kinetic_energy(mass_kg, beta_emc)
                Ek_klas = 0.5 * mass_kg * (beta_emc * SPEED_OF_LIGHT) ** 2
                E_total = gamma_emc * mass_kg * SPEED_OF_LIGHT ** 2
                E0_mev = rest_energy_mev(mass_kg)

                def fmt_energy(e):
                    if e >= 1e18:
                        return f"{e/1e18:.3g} EJ"
                    elif e >= 1e15:
                        return f"{e/1e15:.3g} PJ"
                    elif e >= 1e12:
                        return f"{e/1e12:.3g} TJ"
                    elif e >= 1e9:
                        return f"{e/1e9:.3g} GJ"
                    elif e >= 1e6:
                        return f"{e/1e6:.3g} MJ"
                    elif e >= 1e3:
                        return f"{e/1e3:.3g} kJ"
                    elif e >= 1:
                        return f"{e:.3g} J"
                    elif e >= 1e-10:
                        return f"{e*1e13:.3g} ×10⁻¹³ J"
                    else:
                        return f"{e:.3e} J"

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Massa m</div>
                        <div class="metric-value">{mass_kg:.3e}<span class="metric-unit">kg</span></div>
                    </div>
                    <div class="metric-card" style="margin-top:0.4rem;">
                        <div class="metric-label">Rustenergie E₀ = mc²</div>
                        <div class="metric-value" style="color:{ACCENT1}">{fmt_energy(E0)}</div>
                    </div>
                    <div class="metric-card" style="margin-top:0.4rem;">
                        <div class="metric-label">Rustenergie (deeltjesfysica)</div>
                        <div class="metric-value" style="color:{ACCENT1}">{E0_mev:.3e}<span class="metric-unit">MeV</span></div>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">γ bij β = {fmt(beta_emc,3)}</div>
                        <div class="metric-value">{fmt(gamma_emc, 4)}</div>
                    </div>
                    <div class="metric-card" style="margin-top:0.4rem;">
                        <div class="metric-label">Kinetische energie (relativistisch)</div>
                        <div class="metric-value" style="color:{ACCENT2}">{fmt_energy(Ek_rel)}</div>
                    </div>
                    <div class="metric-card" style="margin-top:0.4rem;">
                        <div class="metric-label">Kinetische energie (klassiek)</div>
                        <div class="metric-value" style="color:{ACCENT3}">{fmt_energy(Ek_klas)}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Grafiek: Ek relativistisch vs klassiek
                st.markdown("---")
                betas_e = np.linspace(0.001, 0.999, 300)
                gamma_e = 1.0 / np.sqrt(1 - betas_e**2)
                Ek_r = (gamma_e - 1) * mass_kg * SPEED_OF_LIGHT**2
                Ek_c = 0.5 * mass_kg * (betas_e * SPEED_OF_LIGHT)**2

                fig, ax = plt.subplots(figsize=(6, 3.5))
                apply_style(ax, fig)
                ax.plot(betas_e, Ek_r, color=ACCENT2, linewidth=2, label="Relativistisch Eₖ = (γ−1)mc²")
                ax.plot(betas_e, Ek_c, color=ACCENT3, linewidth=1.5, linestyle="--", label="Klassiek Eₖ = ½mv²")
                ax.scatter([beta_emc], [Ek_rel], color=ACCENT4, s=60, zorder=5)
                ax.set_xlabel("β = v/c")
                ax.set_ylabel("Kinetische energie (J)")
                ax.set_title(f"Kinetische energie voor m = {mass_kg:.2e} kg")
                ax.legend(fontsize=8, facecolor=BG2, edgecolor="#2d2d3d", labelcolor=TEXT)
                st.pyplot(fig)
                plt.close(fig)

                with st.expander("Context & vergelijkingen"):
                    hiroshima_j = 6.3e13
                    st.write(f"- Rustenergie van {mass_kg:.2e} kg = {fmt_energy(E0)}")
                    st.write(f"- Dat is equivalent aan {E0/hiroshima_j:.2g}× de atoombom op Hiroshima (~63 TJ)")
                    st.write(f"- Bij β = {fmt(beta_emc,3)} is de relativistische Eₖ een factor {Ek_rel/Ek_klas:.2f}× de klassieke Eₖ")

            except Exception as e:
                st.error(f"Fout: {e}")

    # ==============================
    # TAB 8 – Lichtklok
    # ==============================
    if active == "💡 Lichtklok":
        st.subheader("💡 Lichtklok — waarom tijd vertraagt")
        col_left, col_right = st.columns([1.2, 1.8])

        with col_left:
            st.markdown("""
**De lichtklok** is het eenvoudigste gedachte-experiment om tijdsvertraging te begrijpen.

Een lichtpuls stuitert op en neer tussen twee spiegels.

- In het **rustframe**: de puls gaat recht omhoog en omlaag.
- In een **bewegend frame**: de puls legt een langere diagonale weg af.

Omdat licht altijd met $c$ reist, duurt elke tik **langer** in het bewegende frame.

$$t = \\frac{t_0}{\\sqrt{1-β^2}} = γ \\cdot t_0$$
            """)
            st.markdown("---")
            beta_lk = st.slider("Snelheid klok β", 0.0, 0.95, 0.6, 0.01, key="lk_beta")
            L_lk = st.slider("Afstand tussen spiegels L (m)", 1.0, 10.0, 3.0, 0.5, key="lk_L")
            n_ticks = st.slider("Aantal tikken", 1, 5, 3, key="lk_ticks")

        with col_right:
            gamma_lk = gamma_from_beta(beta_lk) if beta_lk > 0 else 1.0
            t0 = 2 * L_lk / SPEED_OF_LIGHT
            t_dil_lk = gamma_lk * t0

            c1, c2, c3 = st.columns(3)
            for col, (lbl, val, color) in zip([c1, c2, c3], [
                ("β", fmt(beta_lk, 3), ACCENT1),
                ("γ", fmt(gamma_lk, 4), ACCENT1),
                ("Tijdsfactor", f"×{fmt(gamma_lk, 3)}", ACCENT2),
            ]):
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{lbl}</div>
                        <div class="metric-value" style="color:{color}">{val}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("---")
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            fig.patch.set_facecolor(BG)

            for ax_lk, (title, beta_plot, color) in zip(axes, [
                ("Rustframe (β = 0)", 0.0, ACCENT1),
                (f"Bewegend frame (β = {fmt(beta_lk,2)})", beta_lk, ACCENT2),
            ]):
                apply_style(ax_lk)
                ax_lk.set_xlim(-1, n_ticks * 3 + 1)
                ax_lk.set_ylim(-0.5, L_lk + 1)
                ax_lk.set_title(title, color=TEXT, fontsize=9)
                ax_lk.set_xlabel("x (richting beweging)")
                ax_lk.set_ylabel("y (hoogte)")

                # Spiegels en lichtpad
                x_pos = 0.0
                going_up = True
                y_cur = 0.0
                for tick in range(n_ticks * 2):
                    dx = beta_plot * L_lk
                    x_next = x_pos + dx
                    y_next = L_lk if going_up else 0.0

                    # Spiegel
                    ax_lk.plot([x_pos - 0.15, x_pos + 0.15], [y_cur, y_cur],
                               color="#4b5563", linewidth=3)
                    # Lichtpad
                    ax_lk.annotate("", xy=(x_next, y_next), xytext=(x_pos, y_cur),
                                   arrowprops=dict(arrowstyle="->", color=color,
                                                   lw=1.5, connectionstyle="arc3,rad=0"))
                    x_pos = x_next
                    y_cur = y_next
                    going_up = not going_up

                ax_lk.plot([x_pos - 0.15, x_pos + 0.15], [y_cur, y_cur],
                           color="#4b5563", linewidth=3)

            plt.tight_layout(pad=1.0)
            st.pyplot(fig)
            plt.close(fig)

            st.markdown(f"""
            <div class="metric-card" style="margin-top:0.5rem;">
                <div class="metric-label">Conclusie</div>
                <div style="color:#94a3b8;font-size:0.88rem;line-height:1.6;">
                    In het rustframe duurt één tik <strong style="color:{ACCENT1}">{t0:.2e} s</strong>.
                    In het bewegende frame duurt dezelfde tik
                    <strong style="color:{ACCENT2}">{t_dil_lk:.2e} s</strong> —
                    een factor <strong style="color:{ACCENT4}">{fmt(gamma_lk,3)}×</strong> langer.
                    De bewegende klok loopt trager.
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("Wiskundige afleiding"):
                st.markdown(r"""
In het rustframe legt de lichtpuls een afstand $2L$ af per tik:
$$t_0 = \frac{2L}{c}$$

In het bewegende frame (snelheid $v$) legt de puls een diagonale weg af:
$$t = \frac{2\sqrt{L^2 + (vt/2)^2}}{c}$$

Dit oplossen naar $t$ geeft:
$$t = \frac{t_0}{\sqrt{1 - v^2/c^2}} = \gamma \cdot t_0$$

Dit is de tijdsvertraging — puur geometrie in ruimtetijd.
                """)

    # ==============================
    # TAB 9 – Gelijktijdigheid
    # ==============================
    if active == "🚂 Gelijktijdigheid":
        st.subheader("🚂 Relativiteit van gelijktijdigheid")
        col_left, col_right = st.columns([1.2, 1.8])

        with col_left:
            st.markdown("""
**Het trein-en-bliksem experiment** (Einstein, 1905):

Een trein rijdt met snelheid β. Op het moment dat het midden van de trein langs een waarnemer op het perron komt, slaan twee bliksems in — aan de voor- en achterkant van de trein.

- **Perronwaarnemer**: beide bliksems zijn gelijktijdig.
- **Treinpassagier**: de voorste bliksem sloeg *eerder* in.

Twee events die gelijktijdig zijn in het ene frame, zijn **niet** gelijktijdig in een ander frame.
            """)
            st.markdown("---")
            beta_gk = st.slider("Treinsnelheid β", 0.1, 0.99, 0.6, 0.01, key="gk_beta")
            L_trein = st.slider("Halve treinlengte L (lichtjaar)", 1.0, 10.0, 5.0, 0.5, key="gk_L")

        with col_right:
            gamma_gk = gamma_from_beta(beta_gk)
            # Tijdsverschil in het treinframe: Δt' = 2·γ·β·L (Lorentz-transformatie van
            # twee events die in het perronframe gelijktijdig zijn op x=±L)
            delta_t_trein = 2 * gamma_gk * beta_gk * L_trein

            c1, c2, c3 = st.columns(3)
            for col, (lbl, val, color) in zip([c1, c2, c3], [
                ("β trein", fmt(beta_gk, 3), ACCENT1),
                ("γ", fmt(gamma_gk, 4), ACCENT1),
                ("Δt treinframe", f"{fmt(delta_t_trein, 3)} jr", ACCENT2),
            ]):
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{lbl}</div>
                        <div class="metric-value" style="color:{color}">{val}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("---")
            fig, axes = plt.subplots(2, 1, figsize=(8, 5))
            fig.patch.set_facecolor(BG)

            for idx, (ax_g, (title, beta_view, color_train)) in enumerate(zip(axes, [
                ("Perronframe — beide bliksems gelijktijdig (t = 0)", 0.0, ACCENT2),
                (f"Treinframe — voorste bliksem sloeg eerder in", beta_gk, ACCENT3),
            ])):
                apply_style(ax_g)
                ax_g.set_xlim(-L_trein * 2.5, L_trein * 2.5)
                ax_g.set_ylim(-1, 2)
                ax_g.set_title(title, color=TEXT, fontsize=9)
                ax_g.set_yticks([])

                if idx == 0:
                    # Perronframe: trein in het midden, bliksems gelijktijdig
                    train_left = -L_trein
                    train_right = L_trein
                    t_offset = 0
                else:
                    # Treinframe: trein stilstaand, perron beweegt
                    train_left = -L_trein
                    train_right = L_trein
                    t_offset = delta_t_trein

                # Trein
                train_rect = plt.Rectangle((train_left, 0.2), 2 * L_trein, 0.6,
                                           facecolor="#1e3a5f", edgecolor=color_train,
                                           linewidth=1.5, zorder=3)
                ax_g.add_patch(train_rect)
                ax_g.text(0, 0.5, "🚂 Trein", ha="center", va="center",
                          color=TEXT, fontsize=9, zorder=4)

                # Rails
                ax_g.axhline(0.2, color="#374151", linewidth=1, zorder=1)

                # Bliksems
                bliksem_kleur_voor = ACCENT4 if idx == 1 else "#ef4444"
                bliksem_kleur_achter = "#ef4444"

                # Achterste bliksem
                ax_g.annotate("", xy=(train_left, 0.8), xytext=(train_left, 1.6),
                              arrowprops=dict(arrowstyle="-|>", color=bliksem_kleur_achter,
                                             lw=2))
                ax_g.text(train_left, 1.7, f"⚡ t=0", ha="center", color=bliksem_kleur_achter,
                          fontsize=8)

                # Voorste bliksem
                t_voor_label = f"t={fmt(-delta_t_trein,2)} jr" if idx == 1 else "t=0"
                ax_g.annotate("", xy=(train_right, 0.8), xytext=(train_right, 1.6),
                              arrowprops=dict(arrowstyle="-|>", color=bliksem_kleur_voor,
                                             lw=2))
                ax_g.text(train_right, 1.7, f"⚡ {t_voor_label}", ha="center",
                          color=bliksem_kleur_voor, fontsize=8)

                # Waarnemer op perron
                if idx == 0:
                    ax_g.scatter([0], [-0.4], color=ACCENT1, s=80, zorder=5)
                    ax_g.text(0, -0.7, "👤 Perron", ha="center", color=ACCENT1, fontsize=8)

            plt.tight_layout(pad=1.0)
            st.pyplot(fig)
            plt.close(fig)

            st.markdown(f"""
            <div class="metric-card" style="margin-top:0.5rem;">
                <div class="metric-label">Tijdsverschil in het treinframe</div>
                <div style="color:#94a3b8;font-size:0.88rem;line-height:1.6;">
                    Twee events die in het perronframe <strong style="color:{ACCENT1}">gelijktijdig</strong> zijn,
                    liggen in het treinframe <strong style="color:{ACCENT2}">{fmt(delta_t_trein, 3)} jaar</strong> uit elkaar.
                    De voorste bliksem sloeg eerder in vanuit de trein gezien.
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("Formule voor het tijdsverschil"):
                st.markdown(r"""
Twee events hebben in het perronframe coördinaten:
- Event A (achter): $(t=0, x=-L)$
- Event B (voor): $(t=0, x=+L)$

Na Lorentz-transformatie naar het treinframe:

$$\Delta t' = \gamma \left(\Delta t - \frac{v \Delta x}{c^2}\right) = \gamma \cdot \frac{-v \cdot 2L}{c^2}$$

Omdat $\Delta t = 0$ (gelijktijdig in perronframe) maar $\Delta x = 2L \neq 0$,
geldt $\Delta t' \neq 0$ in het treinframe. In grootte: $|\Delta t'| = 2\gamma\beta L$.

Dit is de **relativiteit van gelijktijdigheid**: events die ruimtelijk gescheiden zijn
en gelijktijdig in één frame, zijn niet gelijktijdig in een ander frame.
                """)

    # ==============================
    # TAB 10 – Lorentz-transformaties
    # ==============================
    if active == "🔄 Lorentz-transformaties":
        st.subheader("🔄 Lorentz-transformaties")
        col_left, col_right = st.columns([1.2, 1.8])

        with col_left:
            st.markdown("""
**Lorentz-transformatie** converteert coördinaten van het ene inertiaalstelsel naar het andere.

$$t' = γ(t - βx/c)$$
$$x' = γ(x - βt \\cdot c)$$

*(in eenheden c = 1: $t' = γ(t - βx)$, $x' = γ(x - βt)$)*

Voer een **event** in (een punt in ruimtetijd) en zie hoe de coördinaten veranderen in een bewegend frame.
            """)
            st.markdown("---")
            st.markdown("**Event coördinaten (c = 1):**")
            t_ev = st.number_input("t (tijd)", value=5.0, step=0.5, key="lt_t")
            x_ev = st.number_input("x (ruimte)", value=3.0, step=0.5, key="lt_x")
            beta_lt = st.slider("Framesnelheid β", -0.99, 0.99, 0.6, 0.01, key="lt_beta")

            st.markdown("**Voeg meerdere events toe:**")
            events_raw = st.text_area(
                "Extra events (t, x per regel)",
                value="0, 0\n2, 1\n4, 2\n6, 4",
                height=100,
                key="lt_events"
            )

        with col_right:
            gamma_lt = gamma_from_beta(abs(beta_lt))
            t_prime = gamma_lt * (t_ev - beta_lt * x_ev)
            x_prime = gamma_lt * (x_ev - beta_lt * t_ev)

            c1, c2, c3, c4 = st.columns(4)
            for col, (lbl, val, color) in zip([c1, c2, c3, c4], [
                ("t (lab)", fmt(t_ev, 3), ACCENT1),
                ("x (lab)", fmt(x_ev, 3), ACCENT1),
                ("t′ (bewegend frame)", fmt(t_prime, 3), ACCENT2),
                ("x′ (bewegend frame)", fmt(x_prime, 3), ACCENT2),
            ]):
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{lbl}</div>
                        <div class="metric-value" style="color:{color}">{val}</div>
                    </div>""", unsafe_allow_html=True)

            # Invariant interval
            ds2 = t_ev**2 - x_ev**2
            ds2_prime = t_prime**2 - x_prime**2
            st.markdown(f"""
            <div class="metric-card" style="margin-top:0.5rem;border-color:#34d39944;">
                <div class="metric-label">Ruimtetijdinterval ds² = t² − x² (invariant)</div>
                <div style="display:flex;gap:2rem;margin-top:0.3rem;">
                    <span style="color:{ACCENT3}">Lab: <strong>{fmt(ds2,4)}</strong></span>
                    <span style="color:{ACCENT3}">Bewegend frame: <strong>{fmt(ds2_prime,4)}</strong></span>
                    <span style="color:#64748b;font-size:0.8rem;">✓ gelijk (invariant)</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")

            # Events parsen en visualiseren
            events = [(t_ev, x_ev)]
            try:
                for line in events_raw.strip().split("\n"):
                    parts = line.split(",")
                    if len(parts) == 2:
                        et, ex = float(parts[0].strip()), float(parts[1].strip())
                        events.append((et, ex))
            except Exception:
                pass

            events_prime = [(gamma_lt * (e[0] - beta_lt * e[1]),
                             gamma_lt * (e[1] - beta_lt * e[0])) for e in events]

            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            fig.patch.set_facecolor(BG)

            t_all = [e[0] for e in events] + [e[0] for e in events_prime]
            x_all = [e[1] for e in events] + [e[1] for e in events_prime]
            margin = max(abs(max(t_all)), abs(max(x_all)), 1) * 1.3

            for ax_lt, (title, evs, color) in zip(axes, [
                ("Lab-frame", events, ACCENT1),
                (f"Bewegend frame (β={fmt(beta_lt,2)})", events_prime, ACCENT2),
            ]):
                apply_style(ax_lt)
                ax_lt.set_xlim(-margin, margin)
                ax_lt.set_ylim(-0.5, margin)
                ax_lt.set_xlabel("x")
                ax_lt.set_ylabel("t")
                ax_lt.set_title(title, color=TEXT, fontsize=9)

                # Lichtkegel
                tl = np.linspace(0, margin, 100)
                ax_lt.plot(tl, tl, "--", color="#374151", linewidth=0.8, alpha=0.5)
                ax_lt.plot(-tl, tl, "--", color="#374151", linewidth=0.8, alpha=0.5)

                # Events
                for i, (et, ex) in enumerate(evs):
                    is_main = (i == 0)
                    ax_lt.scatter([ex], [et], color=ACCENT4 if is_main else color,
                                  s=80 if is_main else 40, zorder=5)
                    ax_lt.text(ex + margin * 0.04, et, f"E{i}", color=color,
                               fontsize=7, va="center")

            plt.tight_layout(pad=1.0)
            st.pyplot(fig)
            plt.close(fig)

            with st.expander("Meer over het ruimtetijdinterval"):
                st.markdown(r"""
De grootheid $ds^2 = c^2 t^2 - x^2$ is **invariant** onder Lorentz-transformaties.
Alle waarnemers meten dezelfde waarde, ongeacht hun snelheid.

- $ds^2 > 0$: **tijdachtig** interval — er bestaat een frame waarin de events op dezelfde plaats plaatsvinden.
- $ds^2 = 0$: **lichtachtig** interval — de events zijn verbonden door een lichtsignaal.
- $ds^2 < 0$: **ruimteachtig** interval — er bestaat een frame waarin de events gelijktijdig zijn.
                """)

    # ==============================
    # TAB 11 – Relativistisch impuls
    # ==============================
    if active == "💫 Relativistisch impuls":
        st.subheader("💫 Relativistisch impuls en energie")
        col_left, col_right = st.columns([1.2, 1.8])

        with col_left:
            st.markdown("""
**Relativistisch impuls:**
$$p = γmv = γmβc$$

**Totale energie:**
$$E = γmc^2$$

**Energie-impuls relatie:**
$$E^2 = (pc)^2 + (mc^2)^2$$

Voor $m = 0$ (fotonen): $E = pc$
            """)
            st.markdown("---")
            mass_imp = st.text_input("Massa m (kg)", value="1,0", key="imp_m")
            beta_imp = st.slider("Snelheid β", 0.01, 0.999, 0.5, 0.001, key="imp_beta")

        with col_right:
            try:
                m_imp = float(mass_imp.replace(",", "."))
                gamma_imp = gamma_from_beta(beta_imp)
                v_imp = beta_imp * SPEED_OF_LIGHT
                p_rel = gamma_imp * m_imp * v_imp
                p_klas = m_imp * v_imp
                E_tot = gamma_imp * m_imp * SPEED_OF_LIGHT**2
                E0_imp = m_imp * SPEED_OF_LIGHT**2

                c1, c2, c3, c4 = st.columns(4)
                for col, (lbl, val, color) in zip([c1, c2, c3, c4], [
                    ("γ", fmt(gamma_imp, 4), ACCENT1),
                    ("p relativistisch (kg·m/s)", f"{p_rel:.3e}", ACCENT2),
                    ("p klassiek (kg·m/s)", f"{p_klas:.3e}", ACCENT3),
                    ("Factor p_rel/p_klas", fmt(p_rel/p_klas, 4), ACCENT4),
                ]):
                    with col:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">{lbl}</div>
                            <div class="metric-value" style="color:{color}">{val}</div>
                        </div>""", unsafe_allow_html=True)

                st.markdown("---")
                gc1, gc2 = st.columns(2)

                betas_imp = np.linspace(0.01, 0.999, 300)
                gammas_imp = 1.0 / np.sqrt(1 - betas_imp**2)
                v_imp_arr = betas_imp * SPEED_OF_LIGHT
                p_rel_arr = gammas_imp * m_imp * v_imp_arr
                p_klas_arr = m_imp * v_imp_arr

                with gc1:
                    fig, ax = plt.subplots(figsize=(4.5, 3.5))
                    apply_style(ax, fig)
                    ax.plot(betas_imp, p_rel_arr, color=ACCENT2, linewidth=2, label="p = γmv (relativistisch)")
                    ax.plot(betas_imp, p_klas_arr, color=ACCENT3, linewidth=1.5,
                            linestyle="--", label="p = mv (klassiek)")
                    ax.scatter([beta_imp], [p_rel], color=ACCENT4, s=60, zorder=5)
                    ax.set_xlabel("β = v/c")
                    ax.set_ylabel("Impuls p (kg·m/s)")
                    ax.set_title("Impuls vs snelheid")
                    ax.legend(fontsize=8, facecolor=BG2, edgecolor="#2d2d3d", labelcolor=TEXT)
                    st.pyplot(fig)
                    plt.close(fig)

                with gc2:
                    # Energie-impuls diagram (hyperbool)
                    p_range = np.linspace(0, p_rel * 3 + 1e-10, 300)
                    E_range = np.sqrt((p_range * SPEED_OF_LIGHT)**2 + (m_imp * SPEED_OF_LIGHT**2)**2)

                    fig2, ax2 = plt.subplots(figsize=(4.5, 3.5))
                    apply_style(ax2, fig2)
                    ax2.plot(p_range, E_range, color=ACCENT1, linewidth=2, label="E² = (pc)² + (mc²)²")
                    ax2.scatter([p_rel], [E_tot], color=ACCENT4, s=60, zorder=5, label="Huidig punt")
                    ax2.axhline(E0_imp, color=ACCENT3, linewidth=1, linestyle=":",
                                label=f"E₀ = mc² = {E0_imp:.2e} J")
                    ax2.set_xlabel("Impuls p (kg·m/s)")
                    ax2.set_ylabel("Energie E (J)")
                    ax2.set_title("Energie-impuls relatie")
                    ax2.legend(fontsize=7, facecolor=BG2, edgecolor="#2d2d3d", labelcolor=TEXT)
                    st.pyplot(fig2)
                    plt.close(fig2)

                with st.expander("Energie-impuls voor fotonen (m=0)"):
                    st.markdown(r"""
Voor **massaloze deeltjes** (fotonen) geldt $m = 0$, dus:
$$E^2 = (pc)^2 \implies E = pc$$

Een foton met energie $E$ heeft impuls $p = E/c$.

Voorbeeld: een zichtbaar lichtfoton ($\lambda = 500$ nm):
$$E = hf = \frac{hc}{\lambda} \approx 3{,}97 \times 10^{-19} \text{ J}$$
$$p = \frac{E}{c} \approx 1{,}32 \times 10^{-27} \text{ kg·m/s}$$

Dit **stralingsdruk** is meetbaar en wordt gebruikt in zonne-zeilen.
                    """)

            except Exception as e:
                st.error(f"Fout: {e}")

    # ==============================
    # TAB 12 – Zwarte gaten
    # ==============================
    if active == "🕳 Zwarte gaten":
        st.subheader("🕳 Zwarte gaten — Schwarzschild-straal")
        col_left, col_right = st.columns([1.2, 1.8])

        with col_left:
            st.markdown("""
**Schwarzschild-straal** — de grens waarbuiten niets kan ontsnappen:

$$r_s = \\frac{2GM}{c^2}$$

Als een object kleiner is dan $r_s$, wordt het een zwart gat.

**Tijdsvertraging door zwaartekracht:**
$$t_{\\infty} = t_{\\text{lokaal}} \\cdot \\frac{1}{\\sqrt{1 - r_s/r}}$$

Vlak bij de Schwarzschild-straal loopt de tijd oneindig langzaam.
            """)
            st.markdown("---")
            st.markdown("**Massa van het object:**")
            presets_bh = {
                "Zon (2×10³⁰ kg)": 1.989e30,
                "Aarde (6×10²⁴ kg)": 5.972e24,
                "Mens (70 kg)": 70.0,
                "Sagittarius A* (4M☉×10⁶)": 1.989e30 * 4e6,
                "M87* (6,5M☉×10⁹)": 1.989e30 * 6.5e9,
            }
            bh_preset = st.selectbox("Kies een object", list(presets_bh.keys()), key="bh_preset")
            mass_bh = presets_bh[bh_preset]
            r_max_factor = st.slider("Afstand tot zwart gat (× rs)", 1.01, 20.0, 5.0, 0.1, key="bh_r")

        with col_right:
            G = 6.674e-11
            rs = 2 * G * mass_bh / SPEED_OF_LIGHT**2
            r_observer = r_max_factor * rs
            time_factor = 1.0 / math.sqrt(1 - rs / r_observer)

            def fmt_distance(d):
                if d >= 9.461e15:
                    return f"{d/9.461e15:.3g} lichtjaar"
                elif d >= 1.496e11:
                    return f"{d/1.496e11:.3g} AU"
                elif d >= 1000:
                    return f"{d/1000:.3g} km"
                else:
                    return f"{d:.3g} m"

            c1, c2, c3 = st.columns(3)
            for col, (lbl, val, color) in zip([c1, c2, c3], [
                ("Massa", f"{mass_bh:.2e} kg", ACCENT1),
                ("Schwarzschild-straal rs", fmt_distance(rs), ACCENT2),
                ("Tijdsfactor op afstand", f"×{fmt(time_factor, 4)}", ACCENT4),
            ]):
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{lbl}</div>
                        <div class="metric-value" style="color:{color}">{val}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("---")
            gc1, gc2 = st.columns(2)

            with gc1:
                # Tijdsfactor vs afstand
                r_factors = np.linspace(1.001, 20, 500)
                time_factors = 1.0 / np.sqrt(1 - 1.0 / r_factors)

                fig, ax = plt.subplots(figsize=(4.5, 3.5))
                apply_style(ax, fig)
                ax.plot(r_factors, time_factors, color=ACCENT2, linewidth=2)
                ax.scatter([r_max_factor], [time_factor], color=ACCENT4, s=60, zorder=5)
                ax.axvline(1, color="#ef4444", linewidth=0.8, linestyle=":", alpha=0.5)
                ax.fill_betweenx([0, 30], 0, 1, alpha=0.1, color="#ef4444")
                ax.text(0.5, 15, "Zwart\ngat", ha="center", color="#ef4444", fontsize=8)
                ax.set_xlim(0, 20)
                ax.set_ylim(0.9, min(30, time_factors[0] * 0.5))
                ax.set_xlabel("Afstand r / rs")
                ax.set_ylabel("Tijdsfactor (t∞ / t_lokaal)")
                ax.set_title("Gravitationele tijdsvertraging")
                st.pyplot(fig)
                plt.close(fig)

            with gc2:
                # Visuele weergave zwart gat
                fig2, ax2 = plt.subplots(figsize=(4.5, 3.5))
                apply_style(ax2, fig2)
                ax2.set_xlim(-12, 12)
                ax2.set_ylim(-12, 12)
                ax2.set_aspect("equal")
                ax2.set_title("Zwart gat — bovenaanzicht", color=TEXT, fontsize=9)
                ax2.set_xticks([])
                ax2.set_yticks([])

                # Achtergrond sterren
                np.random.seed(42)
                stars_x = np.random.uniform(-12, 12, 80)
                stars_y = np.random.uniform(-12, 12, 80)
                ax2.scatter(stars_x, stars_y, s=1, color="white", alpha=0.4, zorder=1)

                # Accresieschijf
                for i, (r_in, r_out, alpha, color) in enumerate([
                    (1.0, 2.0, 0.6, "#f97316"),
                    (2.0, 3.5, 0.4, "#fbbf24"),
                    (3.5, 5.0, 0.2, "#ef4444"),
                ]):
                    theta = np.linspace(0, 2 * np.pi, 200)
                    for r_ring in np.linspace(r_in, r_out, 5):
                        ax2.plot(r_ring * np.cos(theta), r_ring * np.sin(theta) * 0.3,
                                 color=color, alpha=alpha, linewidth=0.8, zorder=2)

                # Event horizon
                theta = np.linspace(0, 2 * np.pi, 200)
                ax2.fill(np.cos(theta), np.sin(theta), color="black", zorder=5)
                ax2.plot(np.cos(theta), np.sin(theta), color="#1e40af", linewidth=1.5,
                         zorder=6, label="Event horizon (rs)")

                # Fotonring
                ax2.plot(1.5 * np.cos(theta), 1.5 * np.sin(theta),
                         color=ACCENT4, linewidth=0.8, linestyle="--", alpha=0.6,
                         zorder=4, label="Fotonring (1,5 rs)")

                # Observer
                r_obs_vis = min(r_max_factor, 10)
                ax2.scatter([r_obs_vis], [0], color=ACCENT1, s=80, zorder=7)
                ax2.text(r_obs_vis + 0.3, 0.3, "👤", fontsize=10, zorder=8)

                ax2.legend(fontsize=7, facecolor=BG2, edgecolor="#2d2d3d",
                           labelcolor=TEXT, loc="upper right")
                st.pyplot(fig2)
                plt.close(fig2)

            st.markdown(f"""
            <div class="metric-card" style="margin-top:0.5rem;">
                <div class="metric-label">Bekende Schwarzschild-stralen</div>
                <div style="color:#94a3b8;font-size:0.85rem;line-height:1.8;">
                    🌟 Zon → rs ≈ 3 km &nbsp;|&nbsp;
                    🌍 Aarde → rs ≈ 9 mm &nbsp;|&nbsp;
                    🕳 Sagittarius A* → rs ≈ 12 miljoen km &nbsp;|&nbsp;
                    🕳 M87* → rs ≈ 19 miljard km
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("Meer over zwarte gaten"):
                st.markdown(r"""
**Schwarzschild-metriek** (buiten het zwarte gat):
$$ds^2 = -\left(1 - \frac{r_s}{r}\right)c^2 dt^2 + \left(1 - \frac{r_s}{r}\right)^{-1} dr^2 + r^2 d\Omega^2$$

Op $r = r_s$ (event horizon) wordt $g_{tt} = 0$ — de tijd staat stil voor een verre waarnemer.

**Fotonring**: op $r = 1{,}5\, r_s$ kunnen fotonen cirkelbanen maken (onstabiel).

**Hawking-straling**: kwantummechanisch straalt een zwart gat toch energie uit.
Temperatuur: $T_H = \frac{\hbar c^3}{8\pi G M k_B}$
                """)

    # ==============================
    # TAB 13 – Gravitationele tijdsvertraging
    # ==============================
    if active == "🛰 Gravitationele tijdvertraging":
        st.subheader("🛰 Gravitationele tijdsvertraging — GPS als voorbeeld")
        col_left, col_right = st.columns([1.2, 1.8])

        with col_left:
            st.markdown("""
**GPS-satellieten** zijn het perfecte voorbeeld van relativiteit in de praktijk.

Twee effecten spelen een rol:

**1. Speciale relativiteit** (snelheid satelliet):
Klok loopt *trager* door beweging.

**2. Algemene relativiteit** (hoogte/zwaartekracht):
Klok loopt *sneller* verder van de aarde.

Het GR-effect wint: per dag lopen GPS-klokken **+38 microseconde** voor.
Zonder correctie: **11 km** positiefout per dag!
            """)
            st.markdown("---")
            h_km = st.slider("Hoogte satelliet (km)", 200, 42000, 20200, 100, key="gps_h")
            v_sat = st.slider("Snelheid satelliet (m/s)", 1000, 8000, 3874, 10, key="gps_v")

        with col_right:
            # Constanten
            G = 6.674e-11
            M_earth = 5.972e24
            R_earth = 6.371e6
            c = SPEED_OF_LIGHT

            r_sat = R_earth + h_km * 1000
            beta_sat = v_sat / c

            # SR effect: tijdsfactor door snelheid
            gamma_sr = gamma_from_beta(beta_sat)
            sr_rate = 1.0 / gamma_sr  # klok loopt trager

            # GR effect: gravitationeel potentiaal verschil
            phi_surface = -G * M_earth / R_earth
            phi_sat = -G * M_earth / r_sat
            gr_rate = math.sqrt(1 + 2 * (phi_sat - phi_surface) / c**2)

            # Netto effect
            net_rate = sr_rate * gr_rate
            delta_per_day_us = (net_rate - 1) * 86400 * 1e6  # microseconden per dag
            pos_error_m = abs(delta_per_day_us * 1e-6) * c  # meter per dag

            c1, c2, c3 = st.columns(3)
            for col, (lbl, val, color) in zip([c1, c2, c3], [
                ("SR effect (snelheid)", f"{(sr_rate-1)*1e6:.2f} μs/dag", ACCENT2),
                ("GR effect (hoogte)", f"+{(gr_rate-1)*86400*1e6:.2f} μs/dag", ACCENT3),
                ("Netto effect", f"{delta_per_day_us:+.2f} μs/dag", ACCENT4),
            ]):
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{lbl}</div>
                        <div class="metric-value" style="color:{color}">{val}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-card" style="margin-top:0.5rem;border-color:#f472b644;">
                <div class="metric-label">Positiefout zonder relativiteitscorrectie</div>
                <div class="metric-value" style="color:#f472b6">{pos_error_m/1000:.1f} km per dag</div>
                <div style="color:#475569;font-size:0.8rem;margin-top:0.2rem;">
                    GPS-nauwkeurigheid zonder correctie: onbruikbaar.
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")

            # Grafiek: beide effecten vs hoogte
            heights = np.linspace(200, 42000, 300)
            r_sats = R_earth + heights * 1000

            # Typische orbitaalsnelheden (circular orbit: v = sqrt(GM/r))
            v_orbs = np.sqrt(G * M_earth / r_sats)
            beta_orbs = v_orbs / c

            sr_effects = (1.0 / np.sqrt(1 - beta_orbs**2) - 1) * 86400 * 1e6 * (-1)
            gr_effects = (np.sqrt(1 + 2 * (-G * M_earth / r_sats + G * M_earth / R_earth) / c**2) - 1) * 86400 * 1e6
            net_effects = gr_effects - sr_effects

            fig, ax = plt.subplots(figsize=(7, 3.5))
            apply_style(ax, fig)
            ax.plot(heights, sr_effects, color=ACCENT2, linewidth=1.8,
                    linestyle="--", label="SR effect (trager door snelheid)")
            ax.plot(heights, gr_effects, color=ACCENT3, linewidth=1.8,
                    label="GR effect (sneller door hoogte)")
            ax.plot(heights, net_effects, color=ACCENT4, linewidth=2.5,
                    label="Netto effect")
            ax.axhline(0, color="#4b5563", linewidth=0.8, linestyle=":")
            ax.axvline(h_km, color=ACCENT1, linewidth=0.8, linestyle=":", alpha=0.5)
            ax.scatter([h_km], [delta_per_day_us], color=ACCENT4, s=60, zorder=5)
            ax.set_xlabel("Hoogte (km)")
            ax.set_ylabel("Tijdsverschil (μs/dag)")
            ax.set_title("Relativistische tijdscorrecties voor satellieten")
            ax.legend(fontsize=8, facecolor=BG2, edgecolor="#2d2d3d", labelcolor=TEXT)
            st.pyplot(fig)
            plt.close(fig)

            with st.expander("Meer over GPS en relativiteit"):
                st.markdown(r"""
**GPS-satellieten** (hoogte ≈ 20.200 km, snelheid ≈ 3.874 m/s):

- **SR-effect**: $\Delta t_{SR} \approx -7$ μs/dag (klok loopt trager door snelheid)
- **GR-effect**: $\Delta t_{GR} \approx +45$ μs/dag (klok loopt sneller door lagere zwaartekracht)
- **Netto**: $\approx +38$ μs/dag

Dit wordt gecorrigeerd door de klokken in de satellieten *iets trager* in te stellen
voordat ze gelanceerd worden: $f_{sat} = f_0 \times (1 - 4{,}46 \times 10^{-10})$.

Zonder deze correctie: elke dag ~11 km fout in positiebepaling.

Dit is **experimentele bevestiging** van zowel SR als GR in dagelijks gebruik.
                """)

    # ==============================
    # TAB 15 – Galilei vs Einstein
    # ==============================
    if active == "🏛 Galileï vs Einstein":
        st.subheader("🏛 Galileï vs Einstein — twee soorten ruimtetijd")
        st.markdown("""
Beide theorieën gebruiken ruimtetijddiagrammen, maar de regels zijn anders.
Sleep de snelheidsslider en zie live hoe de twee transformaties van elkaar verschillen.
        """)

        beta_ge = st.slider("Snelheid β (v/c)", 0.0, 0.95, 0.6, 0.01, key="ge_beta")
        gamma_ge = gamma_from_beta(beta_ge) if beta_ge > 0 else 1.0

        fig, axes = plt.subplots(1, 2, figsize=(11, 6))
        fig.patch.set_facecolor(BG)
        fig.suptitle(f"β = {fmt(beta_ge, 2)},  γ = {fmt(gamma_ge, 3)}", 
                     color=TEXT, fontsize=11, y=1.01)

        t_max_ge = 6.0
        x_max_ge = 6.0

        # ---- Gemeenschappelijke worldlines in lab-frame ----
        objects = [
            ("Boom (rust)", 0.0,       ACCENT1),
            ("Auto (β)",   beta_ge,    ACCENT2),
            ("Bal (β/2)",  beta_ge/2,  ACCENT3),
        ]

        t_arr = np.linspace(0, t_max_ge, 200)

        titles = ["Galileïsche transformatie (klassiek, Newton)", "Lorentz-transformatie (relativistisch, Einstein)"]

        for ax_ge, title in zip(axes, titles):
            apply_style(ax_ge)
            ax_ge.set_xlim(-x_max_ge, x_max_ge)
            ax_ge.set_ylim(0, t_max_ge)
            ax_ge.set_xlabel("Ruimte x (m of lichtjaar)")
            ax_ge.set_ylabel("Tijd t")
            ax_ge.set_title(title, color=TEXT, fontsize=10)
            ax_ge.axhline(0, color="#374151", linewidth=0.5, alpha=0.4)
            ax_ge.axvline(0, color="#374151", linewidth=0.5, alpha=0.4)

        # Lichtsnelheid lijn alleen in Lorentz
        axes[1].plot( t_arr, t_arr, "--", color="#9ca3af", linewidth=1, alpha=0.4, label="licht (c)")
        axes[1].plot(-t_arr, t_arr, "--", color="#9ca3af", linewidth=1, alpha=0.4)

        for obj_label, beta_obj, color in objects:
            # --- Galileï: x' = x - v*t, t' = t (tijd blijft horizontaal) ---
            # Worldline in auto-frame (Galilei): x_gal = (beta_obj - beta_ge) * t
            beta_gal = beta_obj - beta_ge  # simpele aftrekking
            axes[0].plot(beta_gal * t_arr, t_arr, color=color, linewidth=2, label=obj_label)
            axes[0].text(beta_gal * t_max_ge * 0.8 + 0.1, t_max_ge * 0.85, 
                        obj_label, color=color, fontsize=8)

            # --- Lorentz: x' = gamma*(x - beta*t), t' = gamma*(t - beta*x) ---
            beta_lor = transform_beta_to_frame(beta_obj, beta_ge)
            axes[1].plot(beta_lor * t_arr, t_arr, color=color, linewidth=2, label=obj_label)
            axes[1].text(beta_lor * t_max_ge * 0.8 + 0.1, t_max_ge * 0.85,
                        obj_label, color=color, fontsize=8)

        # Simultaneïteitslijnen
        for t_sim in [1.0, 2.0, 3.0, 4.0, 5.0]:
            x_range = np.linspace(-x_max_ge, x_max_ge, 100)
            # Galileï: simultaneiteit is altijd horizontaal
            axes[0].axhline(t_sim, color="#1e3a5f", linewidth=0.7, alpha=0.5)
            # Lorentz: simultaneiteitslijn heeft helling beta_ge
            t_lorentz_sim = t_sim + beta_ge * x_range
            mask = (t_lorentz_sim >= 0) & (t_lorentz_sim <= t_max_ge)
            axes[1].plot(x_range[mask], t_lorentz_sim[mask], 
                        color="#1e3a5f", linewidth=0.7, alpha=0.6)

        # Tijd-as van het bewegende frame (auto-worldline in lab)
        # Galileï: schuin (maar simultaneïteitslijnen horizontaal)
        # Lorentz: schuin EN simultaneïteitslijnen ook schuin
        axes[0].annotate("Tijd-as auto-frame", 
                        xy=(beta_ge * t_max_ge * 0.6, t_max_ge * 0.6),
                        xytext=(beta_ge * t_max_ge * 0.6 + 0.5, t_max_ge * 0.6),
                        color=ACCENT2, fontsize=7,
                        arrowprops=dict(arrowstyle="->", color=ACCENT2, lw=0.8))
        axes[1].annotate("Tijd-as auto-frame", 
                        xy=(beta_ge * t_max_ge * 0.6, t_max_ge * 0.6),
                        xytext=(beta_ge * t_max_ge * 0.6 + 0.5, t_max_ge * 0.6),
                        color=ACCENT2, fontsize=7,
                        arrowprops=dict(arrowstyle="->", color=ACCENT2, lw=0.8))

        axes[0].legend(fontsize=7, facecolor=BG2, edgecolor="#2d2d3d", labelcolor=TEXT, loc="upper left")
        axes[1].legend(fontsize=7, facecolor=BG2, edgecolor="#2d2d3d", labelcolor=TEXT, loc="upper left")

        plt.tight_layout(pad=1.5)
        st.pyplot(fig, width='stretch')
        plt.close(fig)

        # Uitleg verschil
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown(f"""
<div class="metric-card" style="border-color:#60a5fa44;">
    <div class="metric-label">🏛 Galileïsche transformatie (Newton)</div>
    <div style="color:#94a3b8;font-size:0.85rem;line-height:1.7;margin-top:0.4rem;">
        <strong style="color:{ACCENT1}">Tijd is absoluut</strong> — horizontale simultaneïteitslijnen in alle frames.<br>
        Snelheden tellen gewoon op: u = v₁ + v₂<br>
        Werkt perfect voor v ≪ c.<br><br>
        Transformatie: x′ = x − vt, &nbsp; t′ = t
    </div>
</div>
            """, unsafe_allow_html=True)
        with col_g2:
            st.markdown(f"""
<div class="metric-card" style="border-color:#f472b644;">
    <div class="metric-label">⚡ Lorentz-transformatie (Einstein)</div>
    <div style="color:#94a3b8;font-size:0.85rem;line-height:1.7;margin-top:0.4rem;">
        <strong style="color:{ACCENT2}">Tijd is relatief</strong> — simultaneïteitslijnen kantelen mee met het frame.<br>
        Snelheden tellen relativistisch op.<br>
        Lichtsnelheid c is gelijk in alle frames.<br><br>
        Transformatie: x′ = γ(x − βt), &nbsp; t′ = γ(t − βx)
    </div>
</div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
<div class="metric-card" style="margin-top:0.5rem;border-color:#34d39944;">
    <div class="metric-label">Het cruciale verschil bij β = {fmt(beta_ge, 2)}</div>
    <div style="color:#94a3b8;font-size:0.85rem;line-height:1.7;margin-top:0.3rem;">
        Galileï: snelheid auto t.o.v. boom = <strong style="color:{ACCENT1}">{fmt(beta_ge - 0.0, 3)}c</strong> 
        (simpele aftrekking)<br>
        Lorentz: snelheid auto t.o.v. boom = <strong style="color:{ACCENT2}">{fmt(transform_beta_to_frame(beta_ge, 0.0), 3)}c</strong>
        (relativistische transformatie)<br>
        <span style="color:#475569">Bij lage snelheden zijn ze nagenoeg gelijk. 
        Pas bij β > 0,3 wordt het verschil zichtbaar.</span>
    </div>
</div>
        """, unsafe_allow_html=True)

        with st.expander("Waarom faalt Galileï bij hoge snelheden?"):
            st.markdown(r"""
De Galileïsche transformatie gaat ervan uit dat **tijd absoluut** is:
iedereen meet dezelfde tijdsduur voor hetzelfde event, ongeacht zijn snelheid.

Einstein ontdekte dat dit niet klopt zodra objecten snel bewegen.
Het probleem: als licht altijd met snelheid $c$ reist (gemeten door *iedereen*),
dan kan tijd niet absoluut zijn.

In het Minkowski-diagram zie je dit terug:
- **Galileï**: de tijd-as kantelt bij een boost, maar de simultaneïteitslijnen blijven horizontaal.
- **Lorentz**: zowel de tijd-as *als* de simultaneïteitslijnen kantelen — 
  maar zo dat de lichtkegelhoek (45°) bewaard blijft.

Dit behoud van de lichtkegelhoek is precies wat garandeert dat iedereen
dezelfde lichtsnelheid $c$ meet.
            """)

    # ==============================
    # TAB 16 – Spacetime-volume
    # ==============================
    if active == "🔷 Spacetime-volume":
        st.subheader("🔷 Spacetime-volume conservatie")
        st.markdown("""
Een kernidee uit het boek van Takeuchi: de Lorentz-transformatie **behoudt het spacetime-oppervlak**.
Een vierkant in het ene frame wordt een diamant in het andere — maar met *gelijk oppervlak*.
Dat is geen toeval: het is de reden waarom de transformatie precies zo werkt als ze werkt.
        """)

        beta_sv = st.slider("Snelheid β", 0.0, 0.95, 0.5, 0.01, key="sv_beta")
        L_sv = st.slider("Zijde vierkant L", 0.5, 4.0, 2.0, 0.5, key="sv_L")

        gamma_sv = gamma_from_beta(beta_sv) if beta_sv > 0 else 1.0

        # Vierkant in frame S: hoekpunten (t,x) = (0,0),(L,0),(L,L),(0,L)
        # Na Lorentz: t' = gamma*(t - beta*x), x' = gamma*(x - beta*t)
        def lor_point(t, x, b):
            if abs(b) < 1e-9:
                return t, x
            g = 1.0 / math.sqrt(1 - b*b)
            return g*(t - b*x), g*(x - b*t)

        square_pts = [(0, 0), (L_sv, 0), (L_sv, L_sv), (0, L_sv), (0, 0)]
        diamond_pts = [lor_point(t, x, beta_sv) for t, x in square_pts]

        area_square = L_sv * L_sv
        # Oppervlak diamant via shoelace formule
        dx = [p[1] for p in diamond_pts[:-1]]
        dy = [p[0] for p in diamond_pts[:-1]]
        n = len(dx)
        area_diamond = abs(sum(dx[i]*dy[(i+1)%n] - dx[(i+1)%n]*dy[i] for i in range(n))) / 2

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.patch.set_facecolor(BG)

        lim = L_sv * gamma_sv * 1.6 + 0.5

        for ax_sv, (pts, title, color, fill_color) in zip(axes, [
            (square_pts,  f"Frame S (lab) — oppervlak = {fmt(area_square, 2)}", ACCENT1, "#1e3a5f"),
            (diamond_pts, f"Frame S′ (β = {fmt(beta_sv,2)}) — oppervlak = {fmt(area_diamond, 2)}", ACCENT2, "#3d1a2e"),
        ]):
            apply_style(ax_sv)
            ax_sv.set_xlim(-lim, lim)
            ax_sv.set_ylim(-lim * 0.3, lim)
            ax_sv.set_xlabel("Ruimte x")
            ax_sv.set_ylabel("Tijd t")
            ax_sv.set_title(title, color=TEXT, fontsize=9)
            ax_sv.set_aspect("equal")
            ax_sv.axhline(0, color="#374151", linewidth=0.5, alpha=0.4)
            ax_sv.axvline(0, color="#374151", linewidth=0.5, alpha=0.4)

            # Lichtkegel
            t_lk = np.linspace(0, lim, 100)
            ax_sv.plot( t_lk, t_lk, "--", color="#2d3748", linewidth=0.8, alpha=0.4)
            ax_sv.plot(-t_lk, t_lk, "--", color="#2d3748", linewidth=0.8, alpha=0.4)

            # Roosterlijnen
            for k in range(-int(lim)+1, int(lim)+1):
                ax_sv.axhline(k, color="#1a1a2e", linewidth=0.4, alpha=0.5)
                ax_sv.axvline(k, color="#1a1a2e", linewidth=0.4, alpha=0.5)

            # Vul de vorm
            xs = [p[1] for p in pts]
            ts = [p[0] for p in pts]
            ax_sv.fill(xs, ts, color=fill_color, alpha=0.4, zorder=3)
            ax_sv.plot(xs, ts, color=color, linewidth=2, zorder=4)

            # Hoekpunten
            for i, (t_p, x_p) in enumerate(pts[:-1]):
                ax_sv.scatter([x_p], [t_p], color=ACCENT4, s=50, zorder=5)
                ax_sv.text(x_p + lim*0.03, t_p + lim*0.03, 
                          f"P{i+1}", color=ACCENT4, fontsize=8, zorder=6)

        plt.tight_layout(pad=1.5)
        st.pyplot(fig, width='stretch')
        plt.close(fig)

        # Verificatie oppervlak
        st.markdown(f"""
<div class="metric-card" style="border-color:#34d39944;margin-top:0.5rem;">
    <div class="metric-label">Oppervlakbehoud — verificatie</div>
    <div style="display:flex;gap:2rem;margin-top:0.5rem;flex-wrap:wrap;">
        <div>
            <span style="color:#64748b;font-size:0.78rem;">Vierkant (frame S)</span><br>
            <span style="color:{ACCENT1};font-size:1.2rem;font-weight:600;">{fmt(area_square, 4)}</span>
        </div>
        <div>
            <span style="color:#64748b;font-size:0.78rem;">Diamant (frame S′)</span><br>
            <span style="color:{ACCENT2};font-size:1.2rem;font-weight:600;">{fmt(area_diamond, 4)}</span>
        </div>
        <div>
            <span style="color:#64748b;font-size:0.78rem;">Verschil</span><br>
            <span style="color:{ACCENT3};font-size:1.2rem;font-weight:600;">{fmt(abs(area_square - area_diamond), 6)}</span>
            <span style="color:#475569;font-size:0.78rem;"> ≈ 0 ✓</span>
        </div>
    </div>
</div>
        """, unsafe_allow_html=True)

        with st.expander("Waarom is dit belangrijk? (Takeuchi, sectie 4.7)"):
            st.markdown(r"""
In zijn boek legt Takeuchi dit uit als de **sleutel tot de Lorentz-transformatie**.

Het idee: het "aantal events" in een ruimtetijdgebied mag niet afhangen van
het referentiekader — events zijn objectief. Dat betekent dat het *oppervlak*
van een ruimtetijdgebied invariant moet zijn onder de transformatie.

In de Galileïsche transformatie is dit ook al zo (parallelogram met zelfde basis en hoogte).
In de Lorentz-transformatie ook — maar nu is het effect *zichtbaarder*:
het vierkant wordt een diamant die scheef staat in het nieuwe frame.

De exacte vorm van de transformatie volgt uit twee eisen:
1. **Oppervlak blijft gelijk** (behoud van events)
2. **Lichtsnelheid is gelijk in alle frames** (c = 1 in het diagram)

Samen leiden deze twee eisen *uniek* tot de Lorentz-transformatie.
Dit is de elegantste manier om te zien *waarom* SR precies zo is als het is.
            """)

    # ==============================
    # TAB 17 – Sport-scenarios
    # ==============================
    if active == "⚽ Sport-scenarios":
        st.subheader("⚽ Sport-scenario\'s — relativiteit in beweging")
        st.markdown("""
Gebaseerd op de problemen uit hoofdstuk 8 van Takeuchi\'s boek.
Kies een scenario en zie de situatie in een Minkowski-diagram.
        """)

        sport = st.radio(
            "Kies een scenario",
            [
                "🚂 Trein en tunnel (lengtecontractie)",
                "⚽ Buitenspelregel in voetbal (gelijktijdigheid)",
                "🐢 Haas en schildpad (tijdsvertraging)",
                "💫 Sterrenschip en supernova (causaliteit)",
            ],
            key="sport_keuze"
        )

        st.markdown("---")

        if sport == "🚂 Trein en tunnel (lengtecontractie)":
            st.markdown("""
**Scenario:** Een trein heeft in zijn rustframe lengte $L_{trein}$.
Een tunnel heeft in zijn rustframe lengte $L_{tunnel}$.

De trein rijdt met snelheid β door de tunnel.
Vraag: past de trein tegelijk volledig in de tunnel?

In het **tunnelframe**: de trein is gecontraheerd → misschien past hij erin.
In het **treinframe**: de tunnel is gecontraheerd → de trein past er *niet* in.

Toch is er geen tegenstrijdigheid — het gaat over *gelijktijdigheid* van events!
            """)

            beta_tt = st.slider("Treinsnelheid β", 0.1, 0.99, 0.8, 0.01, key="tt_beta")
            L_trein_tt = st.slider("Treinlengte (rustframe, lichtjaar)", 1.0, 8.0, 5.0, 0.5, key="tt_Ltrein")
            L_tunnel_tt = st.slider("Tunnellengte (rustframe, lichtjaar)", 1.0, 8.0, 4.0, 0.5, key="tt_Ltunnel")

            gamma_tt = gamma_from_beta(beta_tt)
            L_trein_in_tunnel = L_trein_tt / gamma_tt   # trein gezien vanuit tunnel
            L_tunnel_in_trein = L_tunnel_tt / gamma_tt  # tunnel gezien vanuit trein

            c1, c2, c3, c4 = st.columns(4)
            for col, (lbl, val, color) in zip([c1, c2, c3, c4], [
                ("Trein (eigen lengte)", f"{fmt(L_trein_tt,1)} lj", ACCENT2),
                ("Trein in tunnelframe", f"{fmt(L_trein_in_tunnel,2)} lj", ACCENT3),
                ("Tunnel (eigen lengte)", f"{fmt(L_tunnel_tt,1)} lj", ACCENT1),
                ("Tunnel in treinframe", f"{fmt(L_tunnel_in_trein,2)} lj", ACCENT4),
            ]):
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{lbl}</div>
                        <div class="metric-value" style="color:{color}">{val}</div>
                    </div>""", unsafe_allow_html=True)

            past_in_tunnel = L_trein_in_tunnel <= L_tunnel_tt
            st.markdown(f"""
            <div class="metric-card" style="margin-top:0.5rem;
                border-color:{'#34d39944' if past_in_tunnel else '#f472b644'};">
                <div class="metric-label">Past de trein in de tunnel? (tunnelframe)</div>
                <div class="metric-value" style="color:{'#34d399' if past_in_tunnel else '#f472b6'}">
                    {"✓ Ja" if past_in_tunnel else "✗ Nee"} — 
                    trein is {fmt(L_trein_in_tunnel,2)} lj, tunnel is {fmt(L_tunnel_tt,1)} lj
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Minkowski diagram: tunnelframe
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            fig.patch.set_facecolor(BG)
            t_range = np.linspace(0, 8, 200)

            for ax_tt, (frame_title, beta_view, L_obj, L_cont, obj_color) in zip(axes, [
                ("Tunnelframe", 0.0, L_tunnel_tt, L_trein_in_tunnel, ACCENT2),
                ("Treinframe", beta_tt, L_trein_tt, L_tunnel_in_trein, ACCENT1),
            ]):
                apply_style(ax_tt)
                ax_tt.set_xlim(-1, 9)
                ax_tt.set_ylim(-0.5, 8)
                ax_tt.set_xlabel("Ruimte x (lj)")
                ax_tt.set_ylabel("Tijd t (jaar)")
                ax_tt.set_title(frame_title, color=TEXT, fontsize=10)

                # Tunnel worldlines (stilstaand in tunnelframe)
                beta_tunnel_view = transform_beta_to_frame(0.0, beta_view)
                beta_tunnel_exit_view = transform_beta_to_frame(0.0, beta_view)

                # Ingang tunnel: x=0 in tunnelframe
                t_in, x_in_0 = zip(*[
                    (lor_point(t, 0.0, beta_view)[0], lor_point(t, 0.0, beta_view)[1])
                    for t in t_range
                ])
                # Uitgang tunnel: x=L_tunnel in tunnelframe
                t_out, x_out = zip(*[
                    (lor_point(t, L_tunnel_tt, beta_view)[0], lor_point(t, L_tunnel_tt, beta_view)[1])
                    for t in t_range
                ])

                mask1 = np.array(t_in) >= 0
                mask2 = np.array(t_out) >= 0

                ax_tt.plot(np.array(x_in_0)[mask1], np.array(t_in)[mask1],
                          color=ACCENT1, linewidth=2, label="Tunnelingang (x=0)")
                ax_tt.plot(np.array(x_out)[mask2], np.array(t_out)[mask2],
                          color=ACCENT1, linewidth=2, linestyle="--", label=f"Tunneluitgang (x={L_tunnel_tt})")

                # Trein worldlines (beweegt met beta_tt in tunnelframe)
                # Achterkant trein: vertrekt op x=0, t=0 (of eerder)
                t_trein, x_front_arr = zip(*[
                    (lor_point(t, beta_tt*t, beta_view)[0], lor_point(t, beta_tt*t, beta_view)[1])
                    for t in t_range
                ])
                t_back, x_back_arr = zip(*[
                    (lor_point(t, beta_tt*t - L_trein_tt, beta_view)[0], lor_point(t, beta_tt*t - L_trein_tt, beta_view)[1])
                    for t in t_range
                ])

                mask_f = np.array(t_trein) >= 0
                mask_b = np.array(t_back) >= 0

                ax_tt.plot(np.array(x_front_arr)[mask_f], np.array(t_trein)[mask_f],
                          color=ACCENT2, linewidth=2, label="Trein (voorkant)")
                ax_tt.plot(np.array(x_back_arr)[mask_b], np.array(t_back)[mask_b],
                          color=ACCENT2, linewidth=1.5, linestyle="--", label="Trein (achterkant)")

                ax_tt.axhline(0, color="#374151", linewidth=0.5, alpha=0.4)
                ax_tt.legend(fontsize=7, facecolor=BG2, edgecolor="#2d2d3d",
                            labelcolor=TEXT, loc="upper right")

            plt.tight_layout(pad=1.5)
            st.pyplot(fig, width='stretch')
            plt.close(fig)

        elif sport == "⚽ Buitenspelregel in voetbal (gelijktijdigheid)":
            st.markdown("""
**Scenario:** Een speler schiet op goal. Op hetzelfde moment (in het veldframe)
staat een andere aanvaller op de grens van buitenspel.

Vraag: is de aanvaller buitenspel?

In het **veldframe**: beide events (schot en positie aanvaller) zijn gelijktijdig.
In het **frame van een snel bewegende bal**: de events zijn *niet* gelijktijdig.

Gelukkig bewegen voetballers niet met 0,8c — maar het principe is hetzelfde!
            """)

            beta_vs = st.slider("Snelheid bal β (stel je voor!)", 0.1, 0.95, 0.6, 0.01, key="vs_beta")
            x_schot = st.slider("Positie schot (m)", 0.0, 50.0, 20.0, 1.0, key="vs_xschot")
            x_speler = st.slider("Positie aanvaller (m)", 0.0, 50.0, 30.0, 1.0, key="vs_xspeler")
            x_verdediger = st.slider("Positie laatste verdediger (m)", 0.0, 50.0, 28.0, 1.0, key="vs_xverd")

            gamma_vs = gamma_from_beta(beta_vs)
            # Events: A = schot (t=0, x=x_schot), B = positie speler (t=0, x=x_speler)
            # In balframe:
            tA_bal, xA_bal = lor_point(0.0, x_schot, beta_vs)
            tB_bal, xB_bal = lor_point(0.0, x_speler, beta_vs)
            delta_t_bal = tB_bal - tA_bal

            buitenspel_veld = x_speler > x_verdediger
            buitenspel_bal_frame = (xB_bal + beta_vs * abs(delta_t_bal)) > (x_verdediger)

            c1, c2, c3 = st.columns(3)
            for col, (lbl, val, color) in zip([c1, c2, c3], [
                ("Buitenspel in veldframe?", "Ja ✗" if buitenspel_veld else "Nee ✓",
                 "#f472b6" if buitenspel_veld else ACCENT3),
                ("Δt in balframe", f"{fmt(delta_t_bal, 4)} jaar", ACCENT4),
                ("γ bal", fmt(gamma_vs, 3), ACCENT1),
            ]):
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{lbl}</div>
                        <div class="metric-value" style="color:{color}">{val}</div>
                    </div>""", unsafe_allow_html=True)

            # Minkowski diagram
            fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
            fig.patch.set_facecolor(BG)

            for ax_vs, (title, beta_view) in zip(axes, [
                ("Veldframe (rust)", 0.0),
                (f"Balframe (β = {fmt(beta_vs,2)})", beta_vs),
            ]):
                apply_style(ax_vs)
                ax_vs.set_xlim(-5, 55)
                ax_vs.set_ylim(-3, 6)
                ax_vs.set_xlabel("Positie (m)")
                ax_vs.set_ylabel("Tijd t")
                ax_vs.set_title(title, color=TEXT, fontsize=10)
                ax_vs.axhline(0, color="#374151", linewidth=0.5, alpha=0.5)

                # Events
                evA = lor_point(0.0, x_schot, beta_view)
                evB = lor_point(0.0, x_speler, beta_view)
                evV = lor_point(0.0, x_verdediger, beta_view)

                ax_vs.scatter([evA[1]], [evA[0]], color=ACCENT2, s=100, zorder=6, marker="*")
                ax_vs.text(evA[1]+0.5, evA[0]+0.2, "Schot", color=ACCENT2, fontsize=8)

                ax_vs.scatter([evB[1]], [evB[0]], color=ACCENT3, s=100, zorder=6, marker="o")
                ax_vs.text(evB[1]+0.5, evB[0]+0.2, "Aanvaller", color=ACCENT3, fontsize=8)

                ax_vs.axvline(evV[1], color=ACCENT4, linewidth=1.5, linestyle=":",
                             alpha=0.7, label=f"Laatste verdediger")

                # Simultaneiteitslijn in dit frame
                x_sim = np.linspace(-5, 55, 100)
                t_sim_line = evA[0] + beta_vs * (x_sim - evA[1]) if abs(beta_view) > 0.01 else np.zeros_like(x_sim)
                if abs(beta_view) < 0.01:
                    ax_vs.axhline(0, color=ACCENT1, linewidth=1, linestyle="-.",
                                 alpha=0.7, label="Simultaan (veldframe)")
                else:
                    ax_vs.plot(x_sim, t_sim_line, color=ACCENT1, linewidth=1,
                              linestyle="-.", alpha=0.7, label="Simultaan (veldframe)")

                ax_vs.legend(fontsize=7, facecolor=BG2, edgecolor="#2d2d3d",
                            labelcolor=TEXT, loc="upper left")

            plt.tight_layout(pad=1.5)
            st.pyplot(fig, width='stretch')
            plt.close(fig)

        elif sport == "🐢 Haas en schildpad (tijdsvertraging)":
            st.markdown("""
**Scenario (uit Takeuchi, sectie 8.2):**

De haas rent snel naar een verre bestemming en terug.
De schildpad blijft op de startplek.

Wie is ouder als de haas terugkomt?

Dit is eigenlijk de tweelingparadox in een ander jasje — maar nu met een race!
            """)

            beta_hs = st.slider("Snelheid haas β", 0.1, 0.999, 0.8, 0.01, key="hs_beta")
            afstand_hs = st.slider("Afstand bestemming (lichtjaar)", 1.0, 20.0, 5.0, 0.5, key="hs_d")

            gamma_hs = gamma_from_beta(beta_hs)
            t_schildpad = 2 * afstand_hs / beta_hs   # tijd op de klok van de schildpad
            tau_haas = t_schildpad / gamma_hs          # eigen tijd van de haas

            c1, c2, c3, c4 = st.columns(4)
            for col, (lbl, val, color) in zip([c1, c2, c3, c4], [
                ("β haas", fmt(beta_hs, 3), ACCENT2),
                ("γ", fmt(gamma_hs, 4), ACCENT1),
                ("Tijd schildpad (eigen tijd)", f"{fmt(t_schildpad, 2)} jr", ACCENT1),
                ("Tijd haas (eigen tijd)", f"{fmt(tau_haas, 2)} jr", ACCENT3),
            ]):
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{lbl}</div>
                        <div class="metric-value" style="color:{color}">{val}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-card" style="margin-top:0.5rem;border-color:#f472b644;">
                <div class="metric-label">Leeftijdsverschil bij terugkomst</div>
                <div class="metric-value" style="color:#f472b6">{fmt(t_schildpad - tau_haas, 2)} jaar</div>
                <div style="color:#475569;font-size:0.8rem;margin-top:0.2rem;">
                    De schildpad is ouder — de snel bewegende haas veroudert langzamer.
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Worldline diagram
            fig, ax_hs = plt.subplots(figsize=(5, 6))
            apply_style(ax_hs, fig)
            t_omdraaipunt = afstand_hs / beta_hs

            t_wl = np.linspace(0, t_schildpad, 300)
            # Schildpad: stilstaand
            ax_hs.plot([0]*len(t_wl), t_wl, color=ACCENT1, linewidth=2.5, label="🐢 Schildpad")
            # Haas: heen en terug
            t_heen = np.linspace(0, t_omdraaipunt, 150)
            t_terug = np.linspace(t_omdraaipunt, t_schildpad, 150)
            ax_hs.plot(beta_hs * t_heen, t_heen, color=ACCENT2, linewidth=2, label="🐇 Haas (heen)")
            ax_hs.plot(beta_hs * t_omdraaipunt - beta_hs*(t_terug - t_omdraaipunt),
                      t_terug, color=ACCENT2, linewidth=2, linestyle="--", label="🐇 Haas (terug)")

            # Markeerpunten
            ax_hs.scatter([0], [0], color=ACCENT4, s=100, zorder=8)
            ax_hs.scatter([0], [t_schildpad], color=ACCENT4, s=100, zorder=8)
            ax_hs.scatter([beta_hs * t_omdraaipunt], [t_omdraaipunt],
                         color=ACCENT3, s=80, zorder=7)

            ax_hs.text(0.2, t_schildpad, f"Schildpad: {fmt(t_schildpad,1)} jr",
                      color=ACCENT1, fontsize=8)
            ax_hs.text(0.2, tau_haas, f"Haas: {fmt(tau_haas,1)} jr",
                      color=ACCENT2, fontsize=8)
            ax_hs.text(beta_hs * t_omdraaipunt + 0.1, t_omdraaipunt,
                      "Omdraaipunt", color=ACCENT3, fontsize=8)

            ax_hs.set_xlabel("Ruimte x (lichtjaar)")
            ax_hs.set_ylabel("Tijd t (jaar)")
            ax_hs.set_title("Haas en schildpad — worldlines")
            ax_hs.legend(fontsize=9, facecolor=BG2, edgecolor="#2d2d3d", labelcolor=TEXT)
            st.pyplot(fig)
            plt.close(fig)

        else:  # Sterrenschip en supernova
            st.markdown("""
**Scenario (uit Takeuchi, sectie 8.2):**

Een sterrenschip vliegt met snelheid β langs een ster.
Op een bepaald moment explodeert de ster als supernova.

Vraag: kan het sterrenschip de supernova *veroorzaken* hebben?
Of: kan de supernova het sterrenschip *beïnvloeden*?

Dit gaat over **causaliteit** — het verschil tussen tijdachtige en ruimteachtige intervallen.
            """)

            beta_sn = st.slider("Snelheid sterrenschip β", 0.1, 0.999, 0.7, 0.01, key="sn_beta")
            x_sn = st.slider("Positie supernova (lichtjaar)", -5.0, 10.0, 4.0, 0.5, key="sn_x")
            t_sn = st.slider("Tijd supernova (jaar)", -3.0, 8.0, 2.0, 0.5, key="sn_t")

            # Ruimtetijdinterval van (0,0) naar (t_sn, x_sn)
            ds2 = t_sn**2 - x_sn**2

            if ds2 > 0:
                interval_type = "tijdachtig"
                interval_color = ACCENT3
                causal = True
                uitleg = "Er bestaat een frame waarin beide events op dezelfde plek plaatsvinden. Causaliteit is mogelijk."
            elif ds2 < 0:
                interval_type = "ruimteachtig"
                interval_color = ACCENT2
                causal = False
                uitleg = "Er bestaat een frame waarin beide events gelijktijdig zijn. Geen causaliteit mogelijk — de events kunnen elkaar niet beïnvloeden."
            else:
                interval_type = "lichtachtig"
                interval_color = ACCENT4
                causal = True
                uitleg = "De events zijn verbonden door een lichtsignaal. Causaliteit precies op de grens."

            c1, c2, c3 = st.columns(3)
            for col, (lbl, val, color) in zip([c1, c2, c3], [
                ("ds² = t² − x²", fmt(ds2, 3), interval_color),
                ("Type interval", interval_type, interval_color),
                ("Causaliteit mogelijk?", "Ja ✓" if causal else "Nee ✗", 
                 ACCENT3 if causal else "#f472b6"),
            ]):
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{lbl}</div>
                        <div class="metric-value" style="color:{color}">{val}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-card" style="margin-top:0.5rem;border-color:{interval_color}44;">
                <div style="color:#94a3b8;font-size:0.88rem;line-height:1.6;">{uitleg}</div>
            </div>
            """, unsafe_allow_html=True)

            # Minkowski diagram met lichtkegel
            fig, ax_sn = plt.subplots(figsize=(6, 6))
            apply_style(ax_sn, fig)
            lim_sn = max(abs(x_sn), abs(t_sn), 4) * 1.4

            ax_sn.set_xlim(-lim_sn, lim_sn)
            ax_sn.set_ylim(-lim_sn * 0.5, lim_sn)
            ax_sn.set_aspect("equal")
            ax_sn.set_xlabel("Ruimte x (lichtjaar)")
            ax_sn.set_ylabel("Tijd t (jaar)")
            ax_sn.set_title("Sterrenschip en supernova — causaliteit")

            # Lichtkegel vanuit oorsprong
            t_lk = np.linspace(0, lim_sn, 100)
            ax_sn.fill_between(t_lk, t_lk, lim_sn, alpha=0.06, color=ACCENT3)
            ax_sn.fill_between(-t_lk, t_lk, lim_sn, alpha=0.06, color=ACCENT3)
            ax_sn.plot( t_lk, t_lk, "--", color="#4b5563", linewidth=1)
            ax_sn.plot(-t_lk, t_lk, "--", color="#4b5563", linewidth=1)
            ax_sn.fill_between(t_lk, -t_lk, 0, alpha=0.06, color="#f472b6")
            ax_sn.fill_between(-t_lk, -t_lk, 0, alpha=0.06, color="#f472b6")
            ax_sn.plot( t_lk, -t_lk, "--", color="#4b5563", linewidth=1)
            ax_sn.plot(-t_lk, -t_lk, "--", color="#4b5563", linewidth=1)

            ax_sn.text(lim_sn*0.4, lim_sn*0.7, "Toekomst (tijdachtig)", 
                      color=ACCENT3, fontsize=8, ha="center")
            ax_sn.text(lim_sn*0.7, lim_sn*0.2, "Ruimteachtig", 
                      color=ACCENT2, fontsize=8, ha="center")

            # Sterrenschip worldline
            t_ship = np.linspace(-lim_sn * 0.5, lim_sn, 200)
            ax_sn.plot(beta_sn * t_ship, t_ship, color=ACCENT1, linewidth=2,
                      label=f"Sterrenschip (β={fmt(beta_sn,2)})")

            # Events
            ax_sn.scatter([0], [0], color=ACCENT4, s=120, zorder=8, marker="*")
            ax_sn.text(0.1, 0.2, "Event O (schip x=0, t=0)", color=ACCENT4, fontsize=8)

            ax_sn.scatter([x_sn], [t_sn], color=interval_color, s=120, zorder=8, marker="*")
            ax_sn.text(x_sn+0.1, t_sn+0.2, f"Supernova (x={fmt(x_sn,1)}, t={fmt(t_sn,1)})",
                      color=interval_color, fontsize=8)

            # Verbindingslijn
            ax_sn.plot([0, x_sn], [0, t_sn], color=interval_color, linewidth=1.5,
                      linestyle=":", alpha=0.7, label=f"Interval (ds²={fmt(ds2,2)})")

            ax_sn.axhline(0, color="#374151", linewidth=0.5, alpha=0.4)
            ax_sn.axvline(0, color="#374151", linewidth=0.5, alpha=0.4)
            ax_sn.legend(fontsize=8, facecolor=BG2, edgecolor="#2d2d3d",
                        labelcolor=TEXT, loc="upper left")
            st.pyplot(fig)
            plt.close(fig)

    # ==============================
    # TAB 18 – Epstein-cirkel
    # ==============================
    if active == "🔵 Epstein-cirkel":
        st.subheader("🔵 Epstein-cirkel — ruimtetijdsnelheid")
        st.markdown("""
Het kernidee van Epstein: **elk object beweegt altijd met snelheid c door ruimtetijd**.
De vraag is alleen: hoeveel van die snelheid gaat door de *ruimte*, en hoeveel door de *tijd*?

Sleep de slider en zie hoe de verdeling verschuift.
        """)

        col_left, col_right = st.columns([1.2, 1.8])

        with col_left:
            beta_ep = st.slider("Snelheid β (v/c)", 0.0, 0.999, 0.0, 0.001,
                                format="%.3f", key="ep_beta")
            gamma_ep = gamma_from_beta(beta_ep) if beta_ep > 0 else 1.0

            v_space = beta_ep          # snelheid door ruimte (in eenheden c)
            v_time  = 1.0 / gamma_ep   # snelheid door tijd = 1/gamma

            st.markdown("---")
            st.markdown("**Verdeling van snelheid c:**")

            for lbl, val, color, uitleg in [
                ("Snelheid door ruimte (v/c)", v_space, ACCENT2,
                 "Hoe snel het object door de ruimte beweegt"),
                ("Snelheid door tijd (1/γ)", v_time,  ACCENT3,
                 "Hoe snel eigen tijd tikt t.o.v. coördinaattijd"),
            ]:
                st.markdown(f"""
<div class="metric-card">
    <div class="metric-label">{lbl}</div>
    <div class="metric-value" style="color:{color}">{fmt(val, 4)} c</div>
    <div style="color:#475569;font-size:0.78rem;margin-top:0.2rem;">{uitleg}</div>
</div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
<div class="metric-card" style="margin-top:0.4rem;border-color:#34d39944;">
    <div class="metric-label">Verificatie: v² + (1/γ)² = c²</div>
    <div class="metric-value" style="color:{ACCENT3}">{fmt(v_space**2 + v_time**2, 6)} c²</div>
    <div style="color:#475569;font-size:0.78rem;margin-top:0.2rem;">
        Altijd exact 1 — de Pythagoras-stelling in ruimtetijd ✓
    </div>
</div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("""
**Kernboodschap van Epstein:**

- Stilstaand (β=0): alle snelheid gaat door de *tijd* → klok loopt maximaal snel
- Snel bewegen: meer snelheid door *ruimte* → minder overblijft voor de tijd → klok loopt trager
- Bij β→c: bijna alle snelheid door ruimte → klok staat bijna stil
- Licht (β=1): alles door ruimte → klok staat stil (fotonen verouderen niet)
            """)

        with col_right:
            fig, axes = plt.subplots(1, 2, figsize=(9, 5))
            fig.patch.set_facecolor(BG)

            # ---- Links: de Epstein-cirkel ----
            ax_ep = axes[0]
            apply_style(ax_ep)
            ax_ep.set_xlim(-0.15, 1.2)
            ax_ep.set_ylim(-0.15, 1.2)
            ax_ep.set_aspect("equal")
            ax_ep.set_xlabel("Snelheid door ruimte (v/c)", color=ACCENT2)
            ax_ep.set_ylabel("Snelheid door tijd (1/γ)", color=ACCENT3)
            ax_ep.set_title("Epstein-cirkel", color=TEXT, fontsize=10)

            # De cirkel (straal = 1 = c)
            theta = np.linspace(0, np.pi / 2, 200)
            ax_ep.plot(np.cos(theta), np.sin(theta),
                      color=ACCENT1, linewidth=2.5, label="Ruimtetijdsnelheid = c")

            # Hulplijnen
            ax_ep.axhline(0, color="#2d3748", linewidth=0.5)
            ax_ep.axvline(0, color="#2d3748", linewidth=0.5)

            # Het punt op de cirkel
            angle = math.asin(v_time) if v_time <= 1 else math.pi / 2
            px, py = v_space, v_time
            ax_ep.scatter([px], [py], color=ACCENT4, s=120, zorder=8)

            # Pijlen vanuit oorsprong
            ax_ep.annotate("", xy=(px, 0), xytext=(0, 0),
                          arrowprops=dict(arrowstyle="->", color=ACCENT2, lw=2))
            ax_ep.annotate("", xy=(0, py), xytext=(0, 0),
                          arrowprops=dict(arrowstyle="->", color=ACCENT3, lw=2))
            ax_ep.annotate("", xy=(px, py), xytext=(0, 0),
                          arrowprops=dict(arrowstyle="->", color=ACCENT4, lw=2.5))

            # Stippellijnen naar assen
            ax_ep.plot([px, px], [0, py], color=ACCENT2, linewidth=0.8,
                      linestyle=":", alpha=0.5)
            ax_ep.plot([0, px], [py, py], color=ACCENT3, linewidth=0.8,
                      linestyle=":", alpha=0.5)

            # Labels
            if px > 0.05:
                ax_ep.text(px/2, -0.08, f"v/c = {fmt(px,3)}",
                          color=ACCENT2, fontsize=8, ha="center")
            ax_ep.text(-0.12, py/2, f"1/γ = {fmt(py,3)}",
                      color=ACCENT3, fontsize=8, ha="center", rotation=90)

            # Speciale punten
            ax_ep.scatter([0], [1], color=ACCENT3, s=60, zorder=6)
            ax_ep.text(0.03, 1.02, "Stilstand\n(β=0)", color=ACCENT3, fontsize=7)
            ax_ep.scatter([1], [0], color=ACCENT2, s=60, zorder=6)
            ax_ep.text(0.88, 0.04, "Licht\n(β=c)", color=ACCENT2, fontsize=7)

            ax_ep.legend(fontsize=8, facecolor=BG2, edgecolor="#2d2d3d",
                        labelcolor=TEXT, loc="lower left")

            # ---- Rechts: meerdere objecten op de cirkel ----
            ax_ep2 = axes[1]
            apply_style(ax_ep2)
            ax_ep2.set_xlim(-0.15, 1.2)
            ax_ep2.set_ylim(-0.15, 1.2)
            ax_ep2.set_aspect("equal")
            ax_ep2.set_xlabel("Snelheid door ruimte (v/c)")
            ax_ep2.set_ylabel("Snelheid door tijd (1/γ)")
            ax_ep2.set_title("Vergelijking van objecten", color=TEXT, fontsize=10)

            ax_ep2.plot(np.cos(theta), np.sin(theta),
                       color=ACCENT1, linewidth=2, alpha=0.5)
            ax_ep2.axhline(0, color="#2d3748", linewidth=0.5)
            ax_ep2.axvline(0, color="#2d3748", linewidth=0.5)

            voorbeelden = [
                ("Stilstaand", 0.0,   ACCENT3),
                ("Auto (100 km/u)", 100/3.6/SPEED_OF_LIGHT, "#94a3b8"),
                ("Vliegtuig (900 km/u)", 900/3.6/SPEED_OF_LIGHT, "#64748b"),
                ("ISS (7,66 km/s)", 7660/SPEED_OF_LIGHT, "#475569"),
                ("GPS-satelliet", 3874/SPEED_OF_LIGHT, "#334155"),
                (f"Jouw object (β={fmt(beta_ep,2)})", beta_ep, ACCENT4),
                ("Licht (β=1)", 0.9999, ACCENT2),
            ]

            for naam, b, kleur in voorbeelden:
                if b >= 1:
                    b = 0.9999
                gam = 1.0 / math.sqrt(1 - b*b) if b > 0 else 1.0
                vr = b
                vt = 1.0 / gam
                size = 100 if naam.startswith("Jouw") else 40
                ax_ep2.scatter([vr], [vt], color=kleur, s=size, zorder=6)
                if naam not in ["Auto (100 km/u)", "Vliegtuig (900 km/u)",
                                "ISS (7,66 km/s)", "GPS-satelliet"]:
                    offset_x = 0.02
                    ax_ep2.text(vr + offset_x, vt + 0.02, naam,
                               color=kleur, fontsize=7)

            plt.tight_layout(pad=1.5)
            st.pyplot(fig, width='stretch')
            plt.close(fig)

            # Tijdlijn animatie: hoe snel tikt de klok bij elke β
            st.markdown("---")
            st.markdown("**Kloksnelheid als functie van β:**")

            betas_ep = np.linspace(0, 0.999, 300)
            clock_rate = np.sqrt(1 - betas_ep**2)  # = 1/gamma

            fig2, ax3 = plt.subplots(figsize=(7, 2.8))
            apply_style(ax3, fig2)
            ax3.fill_between(betas_ep, 0, clock_rate, alpha=0.15, color=ACCENT3)
            ax3.plot(betas_ep, clock_rate, color=ACCENT3, linewidth=2,
                    label="Kloksnelheid = 1/γ = √(1−β²)")
            ax3.axhline(1, color="#2d3748", linewidth=0.5, linestyle=":")
            ax3.scatter([beta_ep], [v_time], color=ACCENT4, s=80, zorder=5)
            ax3.axvline(beta_ep, color=ACCENT4, linewidth=0.8, linestyle="--", alpha=0.4)
            ax3.set_xlabel("β = v/c")
            ax3.set_ylabel("Kloksnelheid (relatief)")
            ax3.set_title("Hoe snel tikt de eigen klok?")
            ax3.set_ylim(0, 1.1)
            ax3.legend(fontsize=8, facecolor=BG2, edgecolor="#2d2d3d", labelcolor=TEXT)
            st.pyplot(fig2)
            plt.close(fig2)

        with st.expander("Epstein vs. standaard relativiteit — hoe verhouden ze zich?"):
            st.markdown(r"""
Epstein gebruikt een ander diagram dan het standaard Minkowski-diagram,
maar de *fysica* is identiek.

**Epstein-cirkel:** de pijl heeft altijd lengte $c$ (in ruimtetijdeenheden).
De hoek $\theta$ met de tijdas bepaalt de snelheid:
$$v_{ruimte} = c \sin\theta = \beta c, \quad v_{tijd} = c \cos\theta = c/\gamma$$

De Pythagoras-stelling geeft dan:
$$v_{ruimte}^2 + v_{tijd}^2 = c^2\sin^2\theta + c^2\cos^2\theta = c^2 \checkmark$$

**Tijdsvertraging via Epstein:**
Als een object sneller door de ruimte gaat ($\sin\theta$ groter),
gaat het langzamer door de tijd ($\cos\theta$ kleiner).
De eigen klok tikt met factor $\cos\theta = 1/\gamma$ t.o.v. de coördinaatstijd.

Dit is precies $\tau = t/\gamma$ — maar nu zichtbaar als geometrie van een cirkel.
            """)

    # ==============================
    # TAB 19 – Interactieve lichtkegel
    # ==============================
    if active == "💥 Lichtkegel":
        st.subheader("💥 Interactieve lichtkegel — causaliteit in ruimtetijd")
        st.markdown("""
Klik een event op de tijdlijn en zie direct welke andere events causaal bereikbaar zijn.
De lichtkegel scheidt **verleden**, **toekomst** en **ruimteachtig** (niet bereikbaar met causale signalen).
        """)

        col_left, col_right = st.columns([1.2, 1.8])

        with col_left:
            st.markdown("""
**Drie soorten intervallen:**

🟢 **Tijdachtig** (ds² > 0): event ligt binnen de lichtkegel.
Causaliteit mogelijk — er bestaat een frame waarin beide events
op dezelfde plek plaatsvinden.

🟡 **Lichtachtig** (ds² = 0): event ligt precies op de lichtkegel.
Verbonden door een lichtsignaal.

🔴 **Ruimteachtig** (ds² < 0): event ligt buiten de lichtkegel.
Geen causaliteit mogelijk — er bestaat een frame waarin
de events gelijktijdig zijn.
            """)
            st.markdown("---")
            st.markdown("**Positie van het centrale event (oorsprong = 0,0):**")
            n_events = st.slider("Aantal willekeurige events", 3, 15, 8, key="lk_n")
            seed_lk = st.slider("Willekeurig zaad", 0, 99, 42, key="lk_seed")
            t_window_lk = st.slider("Tijdsvenster", 3.0, 12.0, 6.0, 0.5, key="lk_twindow")

            st.markdown("**Specifiek event bekijken:**")
            t_ev_lk = st.slider("Tijd t van event", -t_window_lk, t_window_lk, 3.0, 0.5, key="lk_t")
            x_ev_lk = st.slider("Ruimte x van event", -t_window_lk, t_window_lk, 2.0, 0.5, key="lk_x")

        with col_right:
            # Bereken interval type voor het gekozen event
            ds2_lk = t_ev_lk**2 - x_ev_lk**2

            if ds2_lk > 1e-6:
                ev_type = "tijdachtig"
                ev_color = ACCENT3
                ev_uitleg = f"Dit event ligt BINNEN de lichtkegel van de oorsprong. ds² = {fmt(ds2_lk,3)} > 0. Causaliteit mogelijk."
                in_future = t_ev_lk > 0
                zone = "toekomst" if in_future else "verleden"
            elif ds2_lk < -1e-6:
                ev_type = "ruimteachtig"
                ev_color = ACCENT2
                ev_uitleg = f"Dit event ligt BUITEN de lichtkegel. ds² = {fmt(ds2_lk,3)} < 0. Geen causaliteit mogelijk."
                zone = "ruimteachtig gebied"
            else:
                ev_type = "lichtachtig"
                ev_color = ACCENT4
                ev_uitleg = f"Dit event ligt OP de lichtkegel. ds² ≈ 0. Verbonden door lichtsignaal."
                zone = "lichtkegel"

            c1, c2, c3 = st.columns(3)
            for col, (lbl, val, color) in zip([c1, c2, c3], [
                ("ds² = t² − x²", fmt(ds2_lk, 3), ev_color),
                ("Type", ev_type, ev_color),
                ("Zone", zone, ev_color),
            ]):
                with col:
                    st.markdown(f"""
<div class="metric-card">
    <div class="metric-label">{lbl}</div>
    <div class="metric-value" style="color:{color};font-size:1rem;">{val}</div>
</div>""", unsafe_allow_html=True)

            st.markdown(f"""
<div class="metric-card" style="margin-top:0.4rem;border-color:{ev_color}44;">
    <div style="color:#94a3b8;font-size:0.85rem;line-height:1.6;">{ev_uitleg}</div>
</div>
            """, unsafe_allow_html=True)

            # Genereer willekeurige events
            rng = np.random.RandomState(seed_lk)
            rand_t = rng.uniform(-t_window_lk * 0.9, t_window_lk * 0.9, n_events)
            rand_x = rng.uniform(-t_window_lk * 0.9, t_window_lk * 0.9, n_events)
            rand_ds2 = rand_t**2 - rand_x**2

            fig, ax_lk2 = plt.subplots(figsize=(7, 6.5))
            apply_style(ax_lk2, fig)
            ax_lk2.set_xlim(-t_window_lk, t_window_lk)
            ax_lk2.set_ylim(-t_window_lk, t_window_lk)
            ax_lk2.set_aspect("equal")
            ax_lk2.set_xlabel("Ruimte x (lichtjaar)")
            ax_lk2.set_ylabel("Tijd t (jaar)")
            ax_lk2.set_title("Lichtkegel vanuit oorsprong (0,0)", color=TEXT)

            # Zones inkleuren
            t_fill = np.linspace(0, t_window_lk, 200)
            # Toekomstige lichtkegel (groen)
            ax_lk2.fill_between(t_fill, t_fill, t_window_lk, alpha=0.07, color=ACCENT3)
            ax_lk2.fill_between(-t_fill, t_fill, t_window_lk, alpha=0.07, color=ACCENT3)
            # Verleden lichtkegel (paars)
            ax_lk2.fill_between(t_fill, -t_window_lk, -t_fill, alpha=0.07, color=ACCENT5)
            ax_lk2.fill_between(-t_fill, -t_window_lk, -t_fill, alpha=0.07, color=ACCENT5)
            # Ruimteachtige zones (rood)
            ax_lk2.fill_betweenx(
                np.linspace(-t_window_lk, t_window_lk, 200),
                np.linspace(-t_window_lk, t_window_lk, 200),
                t_window_lk,
                alpha=0.04, color=ACCENT2
            )
            ax_lk2.fill_betweenx(
                np.linspace(-t_window_lk, t_window_lk, 200),
                -t_window_lk,
                -np.linspace(-t_window_lk, t_window_lk, 200),
                alpha=0.04, color=ACCENT2
            )

            # Lichtkegels
            ax_lk2.plot( t_fill, t_fill, "--", color="#6b7280", linewidth=1.5, alpha=0.8)
            ax_lk2.plot(-t_fill, t_fill, "--", color="#6b7280", linewidth=1.5, alpha=0.8)
            ax_lk2.plot( t_fill, -t_fill, "--", color="#6b7280", linewidth=1.5, alpha=0.8)
            ax_lk2.plot(-t_fill, -t_fill, "--", color="#6b7280", linewidth=1.5, alpha=0.8)

            # Zone labels
            ax_lk2.text(0, t_window_lk * 0.75, "TOEKOMST\n(tijdachtig)",
                       ha="center", color=ACCENT3, fontsize=8, alpha=0.8)
            ax_lk2.text(0, -t_window_lk * 0.75, "VERLEDEN\n(tijdachtig)",
                       ha="center", color=ACCENT5, fontsize=8, alpha=0.8)
            ax_lk2.text(t_window_lk * 0.7, 0, "RUIMTE-\nACHTIG",
                       ha="center", color=ACCENT2, fontsize=7, alpha=0.7)
            ax_lk2.text(-t_window_lk * 0.7, 0, "RUIMTE-\nACHTIG",
                       ha="center", color=ACCENT2, fontsize=7, alpha=0.7)

            # Willekeurige events
            for i, (rt, rx, rds2) in enumerate(zip(rand_t, rand_x, rand_ds2)):
                if rds2 > 1e-6:
                    kleur = ACCENT3 if rt > 0 else ACCENT5
                    marker = "^" if rt > 0 else "v"
                elif rds2 < -1e-6:
                    kleur = ACCENT2
                    marker = "s"
                else:
                    kleur = ACCENT4
                    marker = "D"
                ax_lk2.scatter([rx], [rt], color=kleur, s=50,
                              zorder=5, marker=marker, alpha=0.7)
                ax_lk2.text(rx + 0.1, rt + 0.1, str(i+1),
                           color=kleur, fontsize=7, alpha=0.8)

            # Gekozen event
            ax_lk2.scatter([x_ev_lk], [t_ev_lk], color=ev_color, s=200,
                          zorder=8, marker="*")
            ax_lk2.text(x_ev_lk + 0.15, t_ev_lk + 0.15, "★ Jouw event",
                       color=ev_color, fontsize=9, fontweight="500", zorder=9)

            # Lichtsignalen vanuit het gekozen event
            t_sig = np.linspace(t_ev_lk, t_window_lk, 100)
            ax_lk2.plot(x_ev_lk + (t_sig - t_ev_lk), t_sig,
                       color=ev_color, linewidth=1, alpha=0.4, linestyle="-")
            ax_lk2.plot(x_ev_lk - (t_sig - t_ev_lk), t_sig,
                       color=ev_color, linewidth=1, alpha=0.4, linestyle="-")
            t_sig_past = np.linspace(-t_window_lk, t_ev_lk, 100)
            ax_lk2.plot(x_ev_lk + (t_sig_past - t_ev_lk), t_sig_past,
                       color=ev_color, linewidth=1, alpha=0.2, linestyle="-")
            ax_lk2.plot(x_ev_lk - (t_sig_past - t_ev_lk), t_sig_past,
                       color=ev_color, linewidth=1, alpha=0.2, linestyle="-")

            # Oorsprong
            ax_lk2.scatter([0], [0], color=ACCENT4, s=100, zorder=7, marker="o")
            ax_lk2.text(0.1, 0.15, "Oorsprong", color=ACCENT4, fontsize=8)

            ax_lk2.axhline(0, color="#374151", linewidth=0.5, alpha=0.4)
            ax_lk2.axvline(0, color="#374151", linewidth=0.5, alpha=0.4)

            # Legenda
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker="^", color="w", markerfacecolor=ACCENT3,
                       markersize=8, label="Toekomst (tijdachtig)"),
                Line2D([0], [0], marker="v", color="w", markerfacecolor=ACCENT5,
                       markersize=8, label="Verleden (tijdachtig)"),
                Line2D([0], [0], marker="s", color="w", markerfacecolor=ACCENT2,
                       markersize=8, label="Ruimteachtig"),
                Line2D([0], [0], marker="*", color="w", markerfacecolor=ev_color,
                       markersize=12, label="Jouw event"),
            ]
            ax_lk2.legend(handles=legend_elements, fontsize=7, facecolor=BG2,
                         edgecolor="#2d2d3d", labelcolor=TEXT, loc="lower right")

            st.pyplot(fig, width='stretch')
            plt.close(fig)

        with st.expander("Wat betekent het ruimtetijdinterval precies?"):
            st.markdown(r"""
Het **ruimtetijdinterval** $ds^2 = c^2\Delta t^2 - \Delta x^2$ is invariant:
alle waarnemers meten dezelfde waarde, ongeacht hun snelheid.

**Tijdachtig** ($ds^2 > 0$):
Er bestaat een frame waarin beide events op dezelfde plek plaatsvinden.
De tijdsvolgorde is vast: event A is altijd vóór of na event B in alle frames.
Causaliteit is mogelijk.

**Lichtachtig** ($ds^2 = 0$):
Een lichtsignaal kan precies van het ene event naar het andere reizen.
Dit is de grens van de lichtkegel.

**Ruimteachtig** ($ds^2 < 0$):
Er bestaat een frame waarin de events gelijktijdig zijn ($\Delta t' = 0$).
Er bestaat ook een frame waarin de volgorde omgedraaid is.
Geen enkel signaal kan de events verbinden — causaliteit is onmogelijk.

De lichtkegel beschermt causaliteit: oorzaak komt altijd vóór gevolg.
            """)

    # ==============================
    # TAB 20 – Kloksynchronisatie
    # ==============================
    if active == "🕐 Kloksynchronisatie":
        st.subheader("🕐 Kloksynchronisatie — Einstein's methode")
        st.markdown("""
Hoe synchroniseer je twee klokken die ver van elkaar staan?
Einstein bedacht een methode met lichtsignalen — maar die werkt alleen binnen één frame.
        """)

        col_left, col_right = st.columns([1.2, 1.8])

        with col_left:
            st.markdown("""
**Einstein-synchronisatie:**

1. Klok A stuurt een lichtsignaal naar klok B op tijd $t_1$
2. Klok B reflecteert het signaal terug
3. Klok A ontvangt het terug op tijd $t_3$

Klok B wordt ingesteld op:
$$t_B = t_1 + \\frac{t_3 - t_1}{2} = \\frac{t_1 + t_3}{2}$$

Dit werkt perfect — maar alleen in het frame van A en B.
In een bewegend frame zijn de klokken *niet* gesynchroniseerd!
            """)
            st.markdown("---")
            afstand_ks = st.slider("Afstand A→B (lichtjaar)", 0.5, 8.0, 3.0, 0.5, key="ks_d")
            beta_ks    = st.slider("Snelheid waarnemer β", -0.9, 0.9, 0.5, 0.01, key="ks_beta")
            t1_ks      = st.slider("Vertrektijd signaal t₁", 0.0, 5.0, 1.0, 0.5, key="ks_t1")

        with col_right:
            # Tijden in het rustframe van A en B
            t1 = t1_ks
            t_arrive_B = t1 + afstand_ks      # aankomst bij B
            t_B_set    = t1 + afstand_ks / 2  # B instellen op dit moment
            t3         = t1 + 2 * afstand_ks  # terug bij A

            # In bewegend frame: hoe ziet dit eruit?
            gamma_ks = gamma_from_beta(abs(beta_ks)) if abs(beta_ks) > 0 else 1.0

            def lor_t(t, x, b):
                if abs(b) < 1e-9:
                    return t
                g = 1.0 / math.sqrt(1 - b*b)
                return g * (t - b * x)

            # Events: vertrek (t1, 0), aankomst B (t_arrive_B, L),
            #         reflectie (t_arrive_B, L), terug A (t3, 0)
            t1_prime       = lor_t(t1, 0.0, beta_ks)
            tB_prime       = lor_t(t_arrive_B, afstand_ks, beta_ks)
            t3_prime       = lor_t(t3, 0.0, beta_ks)
            tB_set_prime   = lor_t(t_B_set, afstand_ks, beta_ks)
            t_mid_prime    = (t1_prime + t3_prime) / 2
            sync_error     = tB_prime - t_mid_prime

            c1, c2, c3, c4 = st.columns(4)
            for col, (lbl, val, color) in zip([c1, c2, c3, c4], [
                ("t₁ (vertrek signaal)", fmt(t1, 2), ACCENT1),
                ("t_B (instelling klok B)", fmt(t_B_set, 2), ACCENT3),
                ("t₃ (terug bij A)", fmt(t3, 2), ACCENT1),
                ("Synchronisatiefout (bewegend frame)", fmt(sync_error, 4), ACCENT2),
            ]):
                with col:
                    st.markdown(f"""
<div class="metric-card">
    <div class="metric-label">{lbl}</div>
    <div class="metric-value" style="color:{color};font-size:0.95rem;">{val}</div>
</div>""", unsafe_allow_html=True)

            kleur_sync = ACCENT3 if abs(sync_error) < 0.01 else ACCENT2
            st.markdown(f"""
<div class="metric-card" style="margin-top:0.4rem;border-color:{kleur_sync}44;">
    <div class="metric-label">Conclusie synchronisatie</div>
    <div style="color:#94a3b8;font-size:0.85rem;line-height:1.6;margin-top:0.3rem;">
        In het rustframe van A en B zijn de klokken
        <strong style="color:{ACCENT3}">perfect gesynchroniseerd</strong>.<br>
        In het bewegende frame (β = {fmt(beta_ks,2)}) ziet de waarnemer een
        <strong style="color:{ACCENT2}">fout van {fmt(abs(sync_error),4)} jaar</strong> —
        de klokken lopen <em>niet</em> gelijk vanuit zijn perspectief.<br>
        Dit is de relativiteit van gelijktijdigheid in actie.
    </div>
</div>
            """, unsafe_allow_html=True)

            # Minkowski diagram
            st.markdown("---")
            fig, axes = plt.subplots(1, 2, figsize=(10, 5.5))
            fig.patch.set_facecolor(BG)

            for ax_ks, (title, bv) in zip(axes, [
                ("Rustframe A & B", 0.0),
                (f"Bewegend frame (β = {fmt(beta_ks,2)})", beta_ks),
            ]):
                apply_style(ax_ks)
                t_wl_ks = np.linspace(0, t3 + 2, 200)
                xlim_ks = afstand_ks * 1.5

                ax_ks.set_xlim(-xlim_ks * 0.3, xlim_ks * 1.3)
                ax_ks.set_ylim(-0.5, t3 + 1.5)
                ax_ks.set_xlabel("Ruimte x (lichtjaar)")
                ax_ks.set_ylabel("Tijd t (jaar)")
                ax_ks.set_title(title, color=TEXT, fontsize=9)

                # Worldlines A en B
                def lv(t, x, b):
                    if abs(b) < 1e-9:
                        return t, x
                    g = 1.0 / math.sqrt(1 - b*b)
                    return g*(t - b*x), g*(x - b*t)

                t_A = np.array([lv(t, 0.0, bv)[0] for t in t_wl_ks])
                x_A = np.array([lv(t, 0.0, bv)[1] for t in t_wl_ks])
                t_B_wl = np.array([lv(t, afstand_ks, bv)[0] for t in t_wl_ks])
                x_B_wl = np.array([lv(t, afstand_ks, bv)[1] for t in t_wl_ks])

                ax_ks.plot(x_A, t_A, color=ACCENT1, linewidth=2.5, label="Klok A")
                ax_ks.plot(x_B_wl, t_B_wl, color=ACCENT2, linewidth=2.5, label="Klok B")

                # Lichtsignaal heen
                ev1 = lv(t1, 0.0, bv)
                ev_B = lv(t_arrive_B, afstand_ks, bv)
                ev3 = lv(t3, 0.0, bv)

                ax_ks.annotate("", xy=ev_B, xytext=ev1,
                              arrowprops=dict(arrowstyle="->", color=ACCENT4, lw=1.8))
                ax_ks.annotate("", xy=ev3, xytext=ev_B,
                              arrowprops=dict(arrowstyle="->", color=ACCENT4, lw=1.8))

                # Events markeren
                for ev, lbl_ev, kleur_ev in [
                    (ev1, f"t₁={fmt(t1,1)}", ACCENT1),
                    (ev_B, f"B={fmt(t_B_set,2)}", ACCENT3),
                    (ev3, f"t₃={fmt(t3,1)}", ACCENT1),
                ]:
                    ax_ks.scatter([ev[1]], [ev[0]], color=kleur_ev, s=80, zorder=7)
                    ax_ks.text(ev[1] + 0.1, ev[0] + 0.1, lbl_ev,
                              color=kleur_ev, fontsize=8)

                # Synchronisatiemoment op B
                ev_sync = lv(t_B_set, afstand_ks, bv)
                ax_ks.scatter([ev_sync[1]], [ev_sync[0]],
                             color=ACCENT3, s=120, zorder=8, marker="*")

                # Simultaneiteitslijn in dit frame
                ev_mid = lv((t1 + t3)/2, 0.0, bv)
                x_sim_line = np.linspace(-xlim_ks * 0.2, xlim_ks * 1.2, 100)
                t_sim_line = ev_mid[0] + bv * (x_sim_line - ev_mid[1])
                mask_sim = (t_sim_line >= -0.5) & (t_sim_line <= t3 + 1.5)
                ax_ks.plot(x_sim_line[mask_sim], t_sim_line[mask_sim],
                          color=ACCENT1, linewidth=0.8, linestyle="-.",
                          alpha=0.5, label="Simultaan (dit frame)")

                ax_ks.legend(fontsize=7, facecolor=BG2, edgecolor="#2d2d3d",
                            labelcolor=TEXT, loc="upper left")

            plt.tight_layout(pad=1.5)
            st.pyplot(fig, width='stretch')
            plt.close(fig)

            with st.expander("Waarom mislukken gesynchroniseerde klokken in andere frames?"):
                st.markdown(r"""
Einstein-synchronisatie werkt als volgt: stuur een lichtsignaal heen en terug,
en stel de tussenliggende klok in op het *gemiddelde* van vertrektijd en aankomsttijd.

In het restframe van A en B is dit perfect — licht heeft dezelfde snelheid in beide richtingen.

Maar in een *bewegend* frame ziet een waarnemer het licht in de ene richting
*sneller* en in de andere richting *langzamer* bewegen (hoewel hij zelf ook $c$ meet —
het is de *reistijd* die anders is door de beweging van A en B).

Het gevolg: de waarnemer in het bewegende frame ziet dat klok B is ingesteld op
een *ander* moment dan het midden — de klokken zijn uit sync.

Dit is geen praktisch probleem maar een fundamenteel gevolg van de relativiteit van gelijktijdigheid:
$$\Delta t_{sync} = \gamma \cdot \frac{v \cdot L}{c^2}$$
waarbij $L$ de afstand tussen de klokken is in het rustframe.
                """)

    # ==============================
    # TAB 20 – Formulekaart
    # ==============================
    if active == "📖 Formulekaart":
        st.subheader("📖 Formulekaart — speciale relativiteit")
        st.markdown("<br>", unsafe_allow_html=True)

        fc1, fc2 = st.columns(2)

        with fc1:
            st.markdown("""
**Lorentz-factor**
$$γ = \\frac{1}{\\sqrt{1 - β^2}}, \\quad β = \\frac{v}{c}$$

---

**Tijdsvertraging**
$$t = γ \\cdot τ$$
$t$ = coördinaattijd, $τ$ = eigen tijd

---

**Lengtecontractie**
$$L = \\frac{L_0}{γ}$$
$L_0$ = eigen lengte (rustlengte)

---

**Relativistische snelheidsoptelling**
$$u = \\frac{v_1 + v_2}{1 + \\dfrac{v_1 v_2}{c^2}}$$

---

**Lichtklok**
$$t = γ \\cdot t_0, \\quad t_0 = \\frac{2L}{c}$$

---

**Gelijktijdigheid**
$$\\Delta t' = γ\\left(\\Delta t - \\frac{v \\Delta x}{c^2}\\right)$$
            """)

        with fc2:
            st.markdown("""
**Massa-energie equivalentie**
$$E_0 = mc^2, \\quad E = γmc^2, \\quad E_k = (γ-1)mc^2$$

---

**Relativistisch Doppler-effect**
$$f_{obs} = f_0 \\sqrt{\\frac{1 \\pm β}{1 \\mp β}}$$

---

**Relativistisch impuls**
$$p = γmv, \\quad E^2 = (pc)^2 + (mc^2)^2$$

---

**Lorentz-transformatie** *(c = 1)*
$$t' = γ(t - βx), \\quad x' = γ(x - βt)$$

---

**Schwarzschild-straal**
$$r_s = \\frac{2GM}{c^2}$$

---

**Gravitationele tijdsvertraging**
$$\\frac{t_\\infty}{t_{lokaal}} = \\frac{1}{\\sqrt{1 - r_s/r}}$$

---

**Ruimtetijdinterval** *(invariant)*
$$ds^2 = c^2 dt^2 - dx^2 - dy^2 - dz^2$$
            """)

        st.markdown("---")
        st.markdown("""
**Constanten**

| Constante | Symbool | Waarde |
|---|---|---|
| Lichtsnelheid | c | 299 792 458 m/s |
| Gravitatieconstante | G | 6,674 × 10⁻¹¹ N·m²/kg² |
| Massa proton | mₚ | 1,672 × 10⁻²⁷ kg |
| Massa elektron | mₑ | 9,109 × 10⁻³¹ kg |
| Planck-constante | h | 6,626 × 10⁻³⁴ J·s |
| Massa zon | M☉ | 1,989 × 10³⁰ kg |
| Straal aarde | R⊕ | 6,371 × 10⁶ m |
        """)


# ==========================
# PRESENTATIE-MODUS
# ==========================

SLIDES = [
    {
        "titel": "Alles beweegt altijd met de snelheid van het licht",
        "ondertitel": "Maar door de ruimte — of door de tijd?",
        "intro": "Dit is het meest verrassende inzicht van Einstein. Jij beweegt op dit moment met precies de snelheid van het licht — maar bijna volledig door de **tijd**, niet door de ruimte.",
        "type": "epstein",
        "vraag": "Hoe snel beweeg jij door de ruimte?",
    },
    {
        "titel": "Sneller door de ruimte = langzamer door de tijd",
        "ondertitel": "De ruilhandel van ruimtetijd",
        "intro": "Hoe meer snelheid je door de ruimte gebruikt, hoe minder er overblijft voor de tijd. Je klok loopt dan langzamer — niet als een illusie, maar echt.",
        "type": "tijdsvertraging",
        "vraag": "Wat als je met 80% van de lichtsnelheid reist?",
    },
    {
        "titel": "De tweelingparadox",
        "ondertitel": "De astronaut komt jonger terug",
        "intro": "Stel je hebt een tweelingbroer die in een raket stapt en op grote snelheid naar een ster vliegt. Als hij terugkomt, is hij **jonger** dan jij. Dit is geen sciencefiction — het is bewezen.",
        "type": "tweeling",
        "vraag": "Hoeveel jonger is de astronaut?",
    },
    {
        "titel": "GPS werkt dankzij relativiteit",
        "ondertitel": "Zonder Einstein: 11 km fout per dag",
        "intro": "GPS-satellieten bewegen snel (tijdvertraging door snelheid) én bevinden zich hoog (tijdversnelling door minder zwaartekracht). Per dag telt dit op tot 38 microseconde. Zonder correctie: jouw navigatie is na één dag 11 km de fout in.",
        "type": "gps",
        "vraag": "Welk effect is groter?",
    },
    {
        "titel": "E = mc²",
        "ondertitel": "Massa is bevroren energie",
        "intro": "In een paperclip van 1 gram zit genoeg energie om een stad van stroom te voorzien gedurende een jaar. Dat is wat E = mc² betekent. De c² maakt de hoeveelheid gigantisch.",
        "type": "emc2",
        "vraag": "Hoeveel energie zit er in alledaagse objecten?",
    },
    {
        "titel": "Licht heeft altijd dezelfde snelheid",
        "ondertitel": "Voor iedereen, in elk frame",
        "intro": "Of je nu stilstaat of met 99% van de lichtsnelheid vliegt — licht nadert je altijd met exact 299.792.458 m/s. Dit lijkt onmogelijk, maar het is de fundering van de hele relativiteitstheorie.",
        "type": "lichtsnelheid",
        "vraag": "Hoe kan dat kloppen?",
    },
]


def render_slide(slide_idx):
    """Render één presentatieslide met grote visual en minimale tekst."""
    slide = SLIDES[slide_idx]
    n = len(SLIDES)

    # Voortgangsdots
    dot_html = "".join([
        f'<div style="width:{"28" if i == slide_idx else "8"}px;height:8px;'
        f'border-radius:4px;background:{"#60a5fa" if i == slide_idx else "#1e1e2e"};'
        f'transition:all 0.3s;display:inline-block;margin-right:5px;"></div>'
        for i in range(n)
    ])

    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;
                padding:0.6rem 0;border-bottom:1px solid #1e1e2e;margin-bottom:1.2rem;">
        <div>{dot_html}</div>
        <div style="color:#334155;font-size:0.8rem;">{slide_idx+1}/{n}</div>
    </div>
    """, unsafe_allow_html=True)

    # Navigatie knoppen — groot voor touch
    col_home, col_jump, col_prev, col_next = st.columns([1.5, 4, 1.2, 1.2])
    with col_home:
        if st.button("← Home", key="pres_home", width='stretch'):
            st.session_state.page = "home"
            st.rerun()
    with col_jump:
        slide_opties = [f"{i+1}. {SLIDES[i]['titel'][:30]}" for i in range(n)]
        gekozen = st.selectbox("Ga naar hoofdstuk", slide_opties, index=slide_idx,
                              key="pres_jump", label_visibility="collapsed")
        nieuw_idx = slide_opties.index(gekozen)
        if nieuw_idx != slide_idx:
            st.session_state.slide = nieuw_idx
            st.rerun()
    with col_prev:
        prev_disabled = (slide_idx == 0)
        if st.button("◀ Vorige", key="pres_prev",
                    width='stretch',
                    disabled=prev_disabled):
            st.session_state.slide = slide_idx - 1
            st.rerun()
    with col_next:
        if slide_idx < n - 1:
            if st.button("Volgende ▶", key="pres_next",
                        width='stretch', type="primary"):
                st.session_state.slide = slide_idx + 1
                st.rerun()
        else:
            if st.button("🛸 Toolkit", key="pres_toolkit",
                        width='stretch', type="primary"):
                st.session_state.page = "toolkit"
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Titel + intro
    st.markdown(f"""
    <div style="margin-bottom:0.6rem;">
        <span style="background:#1e3a5f;color:#60a5fa;font-size:0.75rem;
                     text-transform:uppercase;letter-spacing:0.1em;
                     padding:0.2rem 0.7rem;border-radius:20px;">
            {slide_idx+1} van {n}
        </span>
    </div>
    <div class="slide-titel">{slide['titel']}</div>
    <div class="slide-sub">{slide['ondertitel']}</div>
    <div class="slide-intro">{slide['intro']}</div>
    <div style="color:#60a5fa;font-size:1rem;font-weight:500;
                border-left:3px solid #60a5fa;padding-left:0.8rem;
                margin-bottom:1.5rem;">{slide['vraag']}</div>
    """, unsafe_allow_html=True)

    # Visual per slide type
    if slide["type"] == "epstein":
        _pres_epstein()
    elif slide["type"] == "tijdsvertraging":
        _pres_tijdsvertraging()
    elif slide["type"] == "tweeling":
        _pres_tweeling()
    elif slide["type"] == "gps":
        _pres_gps()
    elif slide["type"] == "emc2":
        _pres_emc2()
    elif slide["type"] == "lichtsnelheid":
        _pres_lichtsnelheid()

    # Toolkit link onderaan
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="text-align:center;color:#334155;font-size:0.8rem;">
        Wil je dieper duiken? Open de
        <span style="color:#60a5fa;cursor:pointer;">🛸 Toolkit</span>
        voor alle berekeningen en interactieve modules.
    </div>
    """, unsafe_allow_html=True)


def _pres_epstein():
    """Slide 1: Epstein-cirkel — groot en interactief."""
    # Slider direct aan het begin, waarde wordt direct gebruikt
    beta = st.slider("Jouw snelheid (als fractie van lichtsnelheid c)",
                     0.0, 0.999, 0.0, 0.001,
                     format="%.3f β", key="pres_ep_beta")

    # Bereken direct na slider — Streamlit voert dit synchroon uit
    if beta >= 1.0:
        beta = 0.999
    gamma = gamma_from_beta(beta) if beta > 1e-6 else 1.0
    v_ruimte = beta
    v_tijd = 1.0 / gamma
    klok_pct = v_tijd * 100

    # Metric cards groot
    c1, c2, c3 = st.columns(3)
    voorbeelden_snelheden = {
        0.0: "Je staat stil",
        100/3.6/SPEED_OF_LIGHT: "Auto op snelweg",
        900/3.6/SPEED_OF_LIGHT: "Vliegtuig",
        7660/SPEED_OF_LIGHT: "ISS ruimtestation",
        0.8: "80% lichtsnelheid",
        0.99: "99% lichtsnelheid",
    }
    dichtstbij = min(voorbeelden_snelheden.keys(), key=lambda x: abs(x - beta))
    context = voorbeelden_snelheden[dichtstbij] if abs(dichtstbij - beta) < 0.05 else ""

    with c1:
        st.markdown(f"""
        <div class="metric-card" style="text-align:center;padding:1.5rem;">
            <div class="metric-label">Jouw snelheid</div>
            <div style="font-size:2rem;font-weight:600;color:{ACCENT1};">{fmt(beta*100, 2)}%</div>
            <div style="color:#475569;font-size:0.8rem;">van de lichtsnelheid</div>
            <div style="color:#60a5fa;font-size:0.75rem;margin-top:0.3rem;">{context}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card" style="text-align:center;padding:1.5rem;">
            <div class="metric-label">Jouw klok loopt</div>
            <div style="font-size:2rem;font-weight:600;color:{ACCENT2};">{fmt(klok_pct, 4)}%</div>
            <div style="color:#475569;font-size:0.8rem;">zo snel als een stilstaande klok</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        if beta < 1e-6:
            boodschap = "Je beweegt volledig door de tijd ←"
            kleur = ACCENT3
        elif beta > 0.95:
            boodschap = "Bijna alle snelheid door de ruimte →"
            kleur = ACCENT2
        else:
            boodschap = f"γ = {fmt(gamma, 3)} — klok loopt {fmt((1-v_tijd)*100, 1)}% trager"
            kleur = ACCENT4
        st.markdown(f"""
        <div class="metric-card" style="text-align:center;padding:1.5rem;">
            <div class="metric-label">Wat betekent dit?</div>
            <div style="font-size:1rem;font-weight:500;color:{kleur};margin-top:0.5rem;line-height:1.4;">{boodschap}</div>
        </div>""", unsafe_allow_html=True)

    # Grote Epstein-cirkel
    fig, ax = plt.subplots(figsize=(7, 5))
    apply_style(ax, fig)
    ax.set_xlim(-0.1, 1.15)
    ax.set_ylim(-0.1, 1.15)
    ax.set_aspect("equal")
    ax.set_xlabel("← Door de ruimte →", fontsize=11, color=ACCENT2)
    ax.set_ylabel("← Door de tijd →", fontsize=11, color=ACCENT3)
    ax.set_title("Jouw snelheid door ruimtetijd = altijd c", color=TEXT, fontsize=12)

    # Cirkel
    theta = np.linspace(0, np.pi/2, 300)
    ax.plot(np.cos(theta), np.sin(theta), color=ACCENT1, linewidth=3,
            label="Ruimtetijdsnelheid = c (altijd!)")

    # Vulling onder de boog
    ax.fill_between(np.cos(theta), 0, np.sin(theta), alpha=0.04, color=ACCENT1)

    # Huidig punt
    ax.scatter([v_ruimte], [v_tijd], color=ACCENT4, s=250, zorder=10,
               edgecolors="white", linewidths=1.5)

    # Pijlen
    ax.annotate("", xy=(v_ruimte, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=ACCENT2, lw=2.5))
    ax.annotate("", xy=(0, v_tijd), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=ACCENT3, lw=2.5))
    ax.annotate("", xy=(v_ruimte, v_tijd), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=ACCENT4, lw=3,
                               connectionstyle="arc3,rad=0"))

    # Labels
    if v_ruimte > 0.05:
        ax.text(v_ruimte/2, -0.07, f"{fmt(v_ruimte*100,1)}% c",
               color=ACCENT2, fontsize=10, ha="center", fontweight="500")
    ax.text(-0.09, v_tijd/2, f"{fmt(v_tijd*100,1)}% c",
           color=ACCENT3, fontsize=10, ha="center", fontweight="500", rotation=90)

    # Vaste punten
    ax.scatter([0], [1], color=ACCENT3, s=100, zorder=8)
    ax.text(0.03, 1.03, "Stilstand\n(volledig in tijd)", color=ACCENT3, fontsize=8)
    ax.scatter([1], [0], color=ACCENT2, s=100, zorder=8)
    ax.text(0.82, 0.04, "Lichtsnelheid\n(geen tijd meer)", color=ACCENT2, fontsize=8)

    # Stippellijnen
    if v_ruimte > 0.01:
        ax.plot([v_ruimte, v_ruimte], [0, v_tijd], color=ACCENT2,
               linewidth=1, linestyle=":", alpha=0.5)
        ax.plot([0, v_ruimte], [v_tijd, v_tijd], color=ACCENT3,
               linewidth=1, linestyle=":", alpha=0.5)

    ax.legend(fontsize=9, facecolor=BG2, edgecolor="#2d2d3d", labelcolor=TEXT,
             loc="lower left")
    ax.axhline(0, color="#2d3748", linewidth=0.5)
    ax.axvline(0, color="#2d3748", linewidth=0.5)
    st.pyplot(fig, width='stretch')
    plt.close(fig)


def _pres_tijdsvertraging():
    """Slide 2: tijdsvertraging visueel."""
    beta = st.slider("Reissnelheid van de astronaut",
                     0.0, 0.999, 0.8, 0.001,
                     format="%.3f β", key="pres_td_beta")
    reis_jaren = st.slider("Reisduur op aarde (jaren)",
                          1.0, 50.0, 10.0, 1.0, key="pres_td_t")

    gamma = gamma_from_beta(beta) if beta > 0 else 1.0
    tau = reis_jaren / gamma
    verschil = reis_jaren - tau

    c1, c2, c3 = st.columns(3)
    for col, (lbl, val, sub, color) in zip([c1, c2, c3], [
        ("Op aarde verstreken", f"{fmt(reis_jaren, 1)} jaar", "coördinaattijd", ACCENT1),
        ("Voor astronaut verstreken", f"{fmt(tau, 2)} jaar", "eigen tijd", ACCENT2),
        ("Astronaut is jonger met", f"{fmt(verschil, 2)} jaar", f"factor γ = {fmt(gamma, 3)}", ACCENT4),
    ]):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="text-align:center;padding:1.3rem;">
                <div class="metric-label">{lbl}</div>
                <div style="font-size:1.8rem;font-weight:600;color:{color};">{val}</div>
                <div style="color:#475569;font-size:0.8rem;">{sub}</div>
            </div>""", unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    fig.patch.set_facecolor(BG)

    # Links: vergelijkende klokken
    ax1 = axes[0]
    apply_style(ax1)
    ax1.set_xlim(-0.5, 2.5)
    ax1.set_ylim(0, max(reis_jaren, tau) * 1.3)
    ax1.set_title("Verstreken tijd vergeleken", color=TEXT, fontsize=10)
    ax1.set_xticks([0.5, 1.5])
    ax1.set_xticklabels(["Op aarde", "Astronaut"], color=TEXT, fontsize=10)

    bars = ax1.bar([0.5, 1.5], [reis_jaren, tau], width=0.7,
                  color=[ACCENT1, ACCENT2], edgecolor=BG, linewidth=1.5)
    for bar, val in zip(bars, [reis_jaren, tau]):
        ax1.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(reis_jaren, tau) * 0.03,
                f"{fmt(val, 1)} jr",
                ha="center", color=TEXT, fontsize=11, fontweight="600")

    # Rechts: γ curve
    ax2 = axes[1]
    apply_style(ax2)
    betas = np.linspace(0, 0.999, 300)
    gammas = 1.0 / np.sqrt(1 - betas**2)
    tau_curve = reis_jaren / gammas

    ax2.fill_between(betas, tau_curve, reis_jaren, alpha=0.1, color=ACCENT2,
                    label=f"Tijdsverschil (bij {fmt(reis_jaren,0)} jaar op aarde)")
    ax2.plot(betas, tau_curve, color=ACCENT2, linewidth=2,
            label="Eigen tijd astronaut")
    ax2.axhline(reis_jaren, color=ACCENT1, linewidth=1.5, linestyle="--",
               label=f"Tijd op aarde ({fmt(reis_jaren,0)} jr)")
    ax2.scatter([beta], [tau], color=ACCENT4, s=120, zorder=6)
    ax2.axvline(beta, color=ACCENT4, linewidth=0.8, linestyle=":", alpha=0.5)
    ax2.set_xlabel("Snelheid β = v/c")
    ax2.set_ylabel("Tijd (jaar)")
    ax2.set_title("Hoe sneller, hoe jonger de astronaut terugkomt", color=TEXT, fontsize=9)
    ax2.legend(fontsize=8, facecolor=BG2, edgecolor="#2d2d3d", labelcolor=TEXT)

    plt.tight_layout(pad=1.5)
    st.pyplot(fig, width='stretch')
    plt.close(fig)

    # Echte voorbeelden
    st.markdown("**Echte voorbeelden van tijdsvertraging:**")
    c1, c2, c3 = st.columns(3)
    for col, (icon, naam, detail) in zip([c1, c2, c3], [
        ("⚛️", "Muonen in de atmosfeer",
         "Muonen leven 2,2 μs maar bereiken de aarde dankzij tijdsvertraging bij 0,998c"),
        ("🛰️", "GPS-satellieten",
         "Klokken lopen 7 μs/dag trager door snelheid — gecorrigeerd in elk GPS-systeem"),
        ("✈️", "Hafele-Keating (1971)",
         "Atoomklokken in vliegtuigen liepen meetbaar anders — eerste directe bevestiging"),
    ]):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="text-align:center;padding:1rem;">
                <div style="font-size:1.5rem;">{icon}</div>
                <div style="color:{ACCENT1};font-size:0.88rem;font-weight:600;margin:0.3rem 0;">{naam}</div>
                <div style="color:#475569;font-size:0.78rem;line-height:1.4;">{detail}</div>
            </div>""", unsafe_allow_html=True)


def _pres_tweeling():
    """Slide 3: tweelingparadox."""
    beta = st.slider("Snelheid van de reizende tweeling",
                     0.1, 0.999, 0.8, 0.001,
                     format="%.3f β", key="pres_tw_beta")
    t_totaal = st.slider("Totale reistijd op aarde (jaren)",
                        2.0, 60.0, 20.0, 1.0, key="pres_tw_t")

    gamma = gamma_from_beta(beta)
    tau_B = t_totaal / gamma
    verschil = t_totaal - tau_B
    afstand = beta * t_totaal / 2

    c1, c2, c3, c4 = st.columns(4)
    for col, (lbl, val, color) in zip([c1, c2, c3, c4], [
        ("Tweeling A (thuis)", f"{fmt(t_totaal, 1)} jaar", ACCENT1),
        ("Tweeling B (reizend)", f"{fmt(tau_B, 2)} jaar", ACCENT2),
        ("B is jonger met", f"{fmt(verschil, 2)} jaar", ACCENT4),
        ("Afstand gereisd", f"{fmt(afstand, 1)} lichtjaar", ACCENT3),
    ]):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="text-align:center;padding:1.2rem;">
                <div class="metric-label">{lbl}</div>
                <div style="font-size:1.6rem;font-weight:600;color:{color};">{val}</div>
            </div>""", unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    apply_style(ax, fig)
    t_omdraaipunt = t_totaal / 2

    t_A = np.linspace(0, t_totaal, 200)
    ax.plot([0]*len(t_A), t_A, color=ACCENT1, linewidth=3, label="👤 Tweeling A (thuis)")

    t_heen = np.linspace(0, t_omdraaipunt, 100)
    t_terug = np.linspace(t_omdraaipunt, t_totaal, 100)
    x_max = beta * t_omdraaipunt
    ax.plot(beta * t_heen, t_heen, color=ACCENT2, linewidth=2.5,
           label="🚀 Tweeling B (heen)")
    ax.plot(x_max - beta*(t_terug - t_omdraaipunt), t_terug,
           color=ACCENT2, linewidth=2.5, linestyle="--",
           label="🚀 Tweeling B (terug)")

    ax.scatter([0], [0], color=ACCENT4, s=150, zorder=10)
    ax.scatter([0], [t_totaal], color=ACCENT4, s=150, zorder=10)
    ax.scatter([x_max], [t_omdraaipunt], color=ACCENT3, s=120, zorder=9)

    ax.annotate(f"Vertrek\n(beiden {0} jaar)", (0, 0),
               xytext=(0.3, 0.5), color=ACCENT4, fontsize=9,
               arrowprops=dict(arrowstyle="->", color=ACCENT4, lw=1))
    ax.annotate(f"Terugkomst\nA: {fmt(t_totaal,1)} jr\nB: {fmt(tau_B,1)} jr",
               (0, t_totaal),
               xytext=(0.3, t_totaal - 2), color=ACCENT4, fontsize=9,
               arrowprops=dict(arrowstyle="->", color=ACCENT4, lw=1))
    ax.text(x_max + 0.1, t_omdraaipunt, "Omdraaipunt\n(versnelling!)",
           color=ACCENT3, fontsize=9)

    ax.set_xlabel("Ruimte (lichtjaar)")
    ax.set_ylabel("Tijd (jaar)")
    ax.set_title("Worldlines van de twee tweelingen", color=TEXT, fontsize=11)
    ax.legend(fontsize=9, facecolor=BG2, edgecolor="#2d2d3d", labelcolor=TEXT,
             loc="upper left")

    st.pyplot(fig, width='stretch')
    plt.close(fig)

    st.markdown(f"""
    <div class="metric-card" style="margin-top:1rem;border-color:#f472b644;">
        <div style="color:#94a3b8;font-size:0.9rem;line-height:1.7;">
            <strong style="color:#f472b6;">Waarom is het geen paradox?</strong><br>
            Tweeling B moet <em>versnellen</em> om om te draaien. Dat versnelling maakt de situatie
            asymmetrisch — B zit in een niet-inertiaal frame, A niet. Daarom veroudert B echt minder.
            Dit is experimenteel bevestigd met atoomklokken in vliegtuigen.
        </div>
    </div>
    """, unsafe_allow_html=True)


def _pres_gps():
    """Slide 4: GPS en relativiteit."""
    G = 6.674e-11
    M_earth = 5.972e24
    R_earth = 6.371e6
    c = SPEED_OF_LIGHT
    h_gps = 20200  # km
    v_gps = 3874   # m/s

    r_sat = R_earth + h_gps * 1000
    beta_gps = v_gps / c
    gamma_gps = gamma_from_beta(beta_gps)

    sr_effect = -(1 - 1/gamma_gps) * 86400 * 1e6
    phi_diff = G * M_earth * (1/R_earth - 1/r_sat)
    gr_effect = phi_diff / c**2 * 86400 * 1e6
    netto = sr_effect + gr_effect
    fout_km = abs(netto) * 1e-6 * c / 1000

    st.markdown("""
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;margin-bottom:1rem;">
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, (icon, lbl, val, sub, color) in zip([c1, c2, c3], [
        ("🔴", "SR-effect (snelheid)", f"{fmt(sr_effect, 2)} μs/dag",
         "Klok loopt trager door beweging", ACCENT2),
        ("🟢", "GR-effect (hoogte)", f"+{fmt(gr_effect, 2)} μs/dag",
         "Klok loopt sneller door minder zwaartekracht", ACCENT3),
        ("⚡", "Netto effect", f"{fmt(netto, 2)} μs/dag",
         f"= {fmt(fout_km, 1)} km fout per dag zonder correctie", ACCENT4),
    ]):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="text-align:center;padding:1.3rem;">
                <div style="font-size:1.8rem;">{icon}</div>
                <div class="metric-label">{lbl}</div>
                <div style="font-size:1.4rem;font-weight:600;color:{color};">{val}</div>
                <div style="color:#475569;font-size:0.78rem;margin-top:0.3rem;">{sub}</div>
            </div>""", unsafe_allow_html=True)

    # Visualisatie: aarde + satelliet
    fig, ax = plt.subplots(figsize=(7, 5))
    apply_style(ax, fig)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.set_title("GPS-satelliet: twee relativiteitseffecten in één", color=TEXT, fontsize=10)
    ax.axis("off")

    # Aarde
    aarde = plt.Circle((0, 0), 0.7, color="#1a3a5c", zorder=3)
    ax.add_patch(aarde)
    ax.text(0, 0, "🌍", ha="center", va="center", fontsize=28, zorder=4)

    # Atmosfeer ringen
    for r, alpha in [(0.85, 0.15), (1.0, 0.08)]:
        ring = plt.Circle((0, 0), r, fill=False, color=ACCENT1,
                         linewidth=1, alpha=alpha)
        ax.add_patch(ring)

    # Satelliet orbit
    orbit = plt.Circle((0, 0), 2.2, fill=False, color="#334155",
                       linewidth=1.5, linestyle="--")
    ax.add_patch(orbit)

    # Satelliet positie
    angle_sat = math.pi / 4
    sx, sy = 2.2 * math.cos(angle_sat), 2.2 * math.sin(angle_sat)
    ax.text(sx, sy, "🛰️", ha="center", va="center", fontsize=22, zorder=5)

    # Pijl voor snelheid (SR)
    ax.annotate("", xy=(sx - 0.3, sy + 0.3),
               xytext=(sx, sy),
               arrowprops=dict(arrowstyle="-|>", color=ACCENT2, lw=2))
    ax.text(sx - 0.8, sy + 0.5, f"SR: {fmt(sr_effect, 1)} μs/dag",
           color=ACCENT2, fontsize=8, ha="center")

    # Pijl voor hoogte (GR)
    ax.annotate("", xy=(0.5, 1.1), xytext=(0.5, 0.7),
               arrowprops=dict(arrowstyle="-|>", color=ACCENT3, lw=2))
    ax.text(1.2, 0.9, f"GR: +{fmt(gr_effect, 1)} μs/dag",
           color=ACCENT3, fontsize=8)

    # Signaal naar telefoon
    ax.plot([sx*0.85, 0.15], [sy*0.85, -0.5], color=ACCENT4,
           linewidth=1.5, linestyle=":", alpha=0.7)
    ax.text(0.3, -0.7, "📱", ha="center", fontsize=16)
    ax.text(0.3, -1.1, f"Zonder correctie:\n{fmt(fout_km, 1)} km/dag fout!",
           ha="center", color=ACCENT4, fontsize=8)

    st.pyplot(fig, width='stretch')
    plt.close(fig)


def _pres_emc2():
    """Slide 5: E=mc² met concrete voorbeelden."""
    st.markdown("**Kies een object en zie hoeveel energie erin zit:**")

    objecten = {
        "Paperclip (1 gram)": 0.001,
        "Suikerklontje (4 gram)": 0.004,
        "Flesje water (500 ml)": 0.5,
        "Jijzelf (70 kg)": 70.0,
        "Een auto (1500 kg)": 1500.0,
        "Empire State Building (~365.000 ton)": 365_000_000.0,
    }

    keuze = st.selectbox("Object", list(objecten.keys()), key="pres_emc_obj")
    massa = objecten[keuze]
    energie_J = massa * SPEED_OF_LIGHT**2

    vergelijkingen = [
        ("⚡", "kWh elektriciteit", energie_J / 3.6e6, "kWh"),
        ("💣", "× atoombom Hiroshima", energie_J / 6.3e13, "×"),
        ("🏙️", "jaar stroom voor steden (250MW)", energie_J / (250e6 * 365.25 * 24 * 3600), "steden·jaar"),
        ("☀️", "seconden zonne-energie op aarde", energie_J / 1.74e17, "s"),
    ]

    st.markdown(f"""
    <div class="metric-card" style="text-align:center;padding:1.5rem;margin-bottom:1rem;
         border-color:#f472b644;">
        <div style="color:#475569;font-size:0.85rem;">Rustenergie van {keuze}</div>
        <div style="font-size:2.2rem;font-weight:600;color:#f472b6;margin:0.5rem 0;">
            {energie_J:.3e} Joule
        </div>
        <div style="color:#475569;font-size:0.8rem;">E = mc² = {massa} kg × (3×10⁸ m/s)²</div>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(4)
    for col, (icon, lbl, val, eenheid) in zip(cols, vergelijkingen):
        with col:
            if val >= 1e9:
                val_str = f"{val:.2e}"
            elif val >= 1e6:
                val_str = f"{val/1e6:.1f} miljoen"
            elif val >= 1000:
                val_str = f"{val:,.0f}"
            elif val >= 1:
                val_str = f"{val:.2f}"
            else:
                val_str = f"{val:.2e}"
            st.markdown(f"""
            <div class="metric-card" style="text-align:center;padding:1rem;">
                <div style="font-size:1.5rem;">{icon}</div>
                <div style="color:{ACCENT1};font-size:1rem;font-weight:600;">{val_str}</div>
                <div style="color:#475569;font-size:0.75rem;margin-top:0.2rem;">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    # Visualisatie: E vs massa
    fig, ax = plt.subplots(figsize=(7, 3.5))
    apply_style(ax, fig)
    massa_range = np.logspace(-3, 6, 300)
    energie_range = massa_range * SPEED_OF_LIGHT**2

    ax.loglog(massa_range, energie_range, color=ACCENT2, linewidth=2.5,
             label="E = mc²")
    ax.scatter([massa], [energie_J], color=ACCENT4, s=150, zorder=6,
              label=keuze)

    bekende_punten = [
        (0.001, "📎 Paperclip"),
        (70.0, "👤 Mens"),
        (1500.0, "🚗 Auto"),
    ]
    for m, lbl in bekende_punten:
        e = m * SPEED_OF_LIGHT**2
        ax.scatter([m], [e], color=ACCENT3, s=60, zorder=5, alpha=0.7)
        ax.text(m * 1.5, e * 1.5, lbl, color=ACCENT3, fontsize=8)

    ax.set_xlabel("Massa (kg)")
    ax.set_ylabel("Energie (Joule)")
    ax.set_title("E = mc² — energie in massa", color=TEXT, fontsize=10)
    ax.legend(fontsize=9, facecolor=BG2, edgecolor="#2d2d3d", labelcolor=TEXT)
    st.pyplot(fig, width='stretch')
    plt.close(fig)


def _pres_lichtsnelheid():
    """Slide 6: constante lichtsnelheid visualisatie."""
    st.markdown("""
    <div class="metric-card" style="text-align:center;padding:1.5rem;margin-bottom:1rem;
         border-color:#60a5fa44;">
        <div style="font-size:1.1rem;color:#60a5fa;font-weight:500;">De lichtsnelheid</div>
        <div style="font-size:2.5rem;font-weight:700;color:#f1f5f9;margin:0.5rem 0;">
            299.792.458 m/s
        </div>
        <div style="color:#475569;">Precies hetzelfde voor elke waarnemer, in elk frame, altijd.</div>
    </div>
    """, unsafe_allow_html=True)

    beta_waarnemer = st.slider(
        "Snelheid van de waarnemer",
        -0.95, 0.95, 0.0, 0.01,
        format="%.2f β", key="pres_lk_beta"
    )

    # Relativistische snelheidsoptelling: licht (β=1) + waarnemer
    beta_licht_klassiek = 1.0 + beta_waarnemer  # klassiek
    beta_licht_rel = relativistic_velocity_add(
        1.0 * SPEED_OF_LIGHT, beta_waarnemer * SPEED_OF_LIGHT
    ) / SPEED_OF_LIGHT

    c1, c2, c3 = st.columns(3)
    for col, (lbl, val, sub, color) in zip([c1, c2, c3], [
        ("Waarnemer beweegt met", f"{fmt(abs(beta_waarnemer)*100, 0)}% c",
         "van de lichtsnelheid", ACCENT1),
        ("Lichtsnelheid klassiek gemeten", f"{fmt(beta_licht_klassiek, 4)} c",
         "simpele optelling → FOUT", ACCENT2),
        ("Lichtsnelheid relativistisch", f"{fmt(beta_licht_rel, 6)} c",
         "altijd exact 1,000000 c → CORRECT", ACCENT3),
    ]):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="text-align:center;padding:1.2rem;">
                <div class="metric-label">{lbl}</div>
                <div style="font-size:1.3rem;font-weight:600;color:{color};">{val}</div>
                <div style="color:#475569;font-size:0.78rem;margin-top:0.2rem;">{sub}</div>
            </div>""", unsafe_allow_html=True)

    # Visualisatie: Minkowski diagram met lichtkegel
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.patch.set_facecolor(BG)

    t_arr = np.linspace(0, 5, 200)

    for ax_lk, (title, beta_ref, show_tilt) in zip(axes, [
        ("Klassiek (Newton) — lichtsnelheid verandert", 0.0, False),
        ("Relativistisch (Einstein) — lichtsnelheid constant", beta_waarnemer, True),
    ]):
        apply_style(ax_lk)
        ax_lk.set_xlim(-5.5, 5.5)
        ax_lk.set_ylim(0, 5)
        ax_lk.set_xlabel("Ruimte x")
        ax_lk.set_ylabel("Tijd t")
        ax_lk.set_title(title, color=TEXT, fontsize=9)

        # Lichtkegel vanuit oorsprong
        ax_lk.plot( t_arr, t_arr, color=ACCENT4, linewidth=2.5,
                   label="Lichtpuls (+c)")
        ax_lk.plot(-t_arr, t_arr, color=ACCENT4, linewidth=2.5,
                   label="Lichtpuls (−c)")
        ax_lk.fill_between(t_arr, t_arr, 5, alpha=0.06, color=ACCENT4)
        ax_lk.fill_between(-t_arr, t_arr, 5, alpha=0.06, color=ACCENT4)

        # Waarnemer worldline
        if abs(beta_ref) > 0.01:
            beta_w_view = 0.0 if not show_tilt else transform_beta_to_frame(beta_waarnemer, beta_waarnemer)
            beta_w_lab = beta_waarnemer if not show_tilt else 0.0
            ax_lk.plot(beta_w_lab * t_arr, t_arr, color=ACCENT1, linewidth=2,
                      label=f"Waarnemer (β={fmt(beta_waarnemer,2)})")
            if not show_tilt:
                # Klassiek: licht lijkt sneller/langzamer
                beta_lk_class = 1.0 + beta_waarnemer
                ax_lk.plot(beta_lk_class * t_arr, t_arr, color="#ef4444",
                          linewidth=2, linestyle="--",
                          label=f"Licht gemeten: {fmt(beta_lk_class,2)}c (FOUT)")
        else:
            ax_lk.plot([0]*len(t_arr), t_arr, color=ACCENT1, linewidth=2,
                      label="Waarnemer (stilstaand)")

        ax_lk.legend(fontsize=7, facecolor=BG2, edgecolor="#2d2d3d",
                    labelcolor=TEXT, loc="upper left")
        ax_lk.axhline(0, color="#374151", linewidth=0.5, alpha=0.4)
        ax_lk.axvline(0, color="#374151", linewidth=0.5, alpha=0.4)

    plt.tight_layout(pad=1.5)
    st.pyplot(fig, width='stretch')
    plt.close(fig)

    st.markdown(f"""
    <div class="metric-card" style="margin-top:1rem;border-color:#60a5fa44;">
        <div style="color:#94a3b8;font-size:0.9rem;line-height:1.7;">
            <strong style="color:#60a5fa;">Waarom is dit zo revolutionair?</strong><br>
            Newton ging ervan uit dat snelheden gewoon optellen. Als licht 300.000 km/s gaat
            en jij vliegt erachteraan met 150.000 km/s, dan zou je licht met 150.000 km/s
            zien bewegen. Maar dat klopt niet — alle experimenten (Michelson-Morley, 1887)
            lieten zien dat licht altijd exact c is. Einstein nam dit als axioma en bouwde
            daar de hele relativiteitstheorie op.
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_presentatie():
    if "slide" not in st.session_state:
        st.session_state.slide = 0
    render_slide(st.session_state.slide)


# ==========================
# Routing
# ==========================

if st.session_state.page == "home":
    render_home()
elif st.session_state.page == "presentatie":
    render_presentatie()
else:
    render_toolkit()
