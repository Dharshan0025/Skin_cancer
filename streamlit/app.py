"""
DermAI  —  AI-Based Skin Cancer Detection
Dataset : HAM10000 (ISIC Archive)
Design  : Precision Noir — Jeton-quality medical SaaS
"""

import os
import time
import numpy as np
import streamlit as st
from PIL import Image
from datetime import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

st.set_page_config(
    page_title="DermAI — Skin Cancer Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  PRECISION NOIR — DESIGN SYSTEM
#  Palette  : #080C14 bg · #0E1420 surface · #00C2FF cyan · semantic risk colors
#  Geometry : 0–4px radius — sharp, clinical, precise
#  Motion   : fadeUp entrance · shimmer loading · hover lift
#  Font     : Inter — Google Fonts
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">

<style>
/* ── Reset ─────────────────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: #080C14 !important;
    color: #CBD5E1 !important;
    -webkit-font-smoothing: antialiased;
}

/* ── Hide chrome ───────────────────────────────────────────────────────────── */
#MainMenu, footer, header, .stDeployButton { visibility: hidden !important; }

/* ── Scrollbar ─────────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #080C14; }
::-webkit-scrollbar-thumb { background: #1A2540; border-radius: 2px; }

/* ── Animations ────────────────────────────────────────────────────────────── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes shimmer {
    0%   { background-position: -400px 0; }
    100% { background-position: 400px 0; }
}
@keyframes pulse-border {
    0%, 100% { border-color: rgba(0,194,255,0.3); }
    50%       { border-color: rgba(0,194,255,0.7); }
}
@keyframes barGrow {
    from { width: 0 !important; }
    to   { width: var(--target-w); }
}

/* ── Main container ────────────────────────────────────────────────────────── */
.main .block-container {
    padding: 0 2rem 4rem 2rem !important;
    max-width: 1280px !important;
}

/* ── Sidebar ───────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #060A12 !important;
    border-right: 1px solid #0E1420 !important;
}
[data-testid="stSidebar"] * { color: #94A3B8 !important; }
[data-testid="stSidebar"] .stButton > button {
    background: #00C2FF !important;
    color: #080C14 !important;
    font-weight: 700 !important;
    border-radius: 2px !important;
    letter-spacing: 0.04em !important;
    font-size: 0.82rem !important;
    text-transform: uppercase !important;
    border: none !important;
    padding: 0.7rem 1rem !important;
    transition: background 0.15s, box-shadow 0.15s !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #00A8DB !important;
    box-shadow: 0 0 20px rgba(0,194,255,0.25) !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #0A0F1A !important;
    border: 1px solid #1A2540 !important;
    border-radius: 2px !important;
    color: #CBD5E1 !important;
    font-size: 0.82rem !important;
}
[data-testid="stFileUploader"] {
    background: #0A0F1A !important;
    border: 1.5px dashed #1A2540 !important;
    border-radius: 2px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #00C2FF !important;
    animation: pulse-border 2s infinite !important;
}

/* ── Alert overrides ───────────────────────────────────────────────────────── */
[data-testid="stAlert"] {
    background: #0A0F1A !important;
    border: 1px solid #1A2540 !important;
    border-radius: 2px !important;
    color: #64748B !important;
    font-size: 0.82rem !important;
}

/* ══════════════════════════════════════════════════════════════════════════
   CUSTOM COMPONENTS
══════════════════════════════════════════════════════════════════════════ */

/* ── Topbar ────────────────────────────────────────────────────────────── */
.topbar {
    background: #060A12;
    border-bottom: 1px solid #0E1420;
    padding: 0 2rem;
    margin: 0 -2rem 0 -2rem;
    display: flex;
    align-items: stretch;
    justify-content: space-between;
    height: 52px;
    animation: fadeIn 0.4s ease both;
    position: sticky;
    top: 0;
    z-index: 999;
}
.topbar-left { display: flex; align-items: center; gap: 0.75rem; }
.topbar-dot {
    width: 8px; height: 8px;
    background: #00C2FF;
    border-radius: 50%;
    box-shadow: 0 0 8px rgba(0,194,255,0.6);
}
.topbar-name {
    font-size: 1rem;
    font-weight: 800;
    color: #F1F5F9;
    letter-spacing: -0.02em;
}
.topbar-version {
    font-size: 0.65rem;
    font-weight: 600;
    color: #00C2FF;
    background: rgba(0,194,255,0.08);
    border: 1px solid rgba(0,194,255,0.2);
    padding: 0.15rem 0.45rem;
    border-radius: 2px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* Nav tabs inside topbar */
.topbar-nav {
    display: flex;
    align-items: stretch;
    gap: 0;
}
.nav-tab {
    display: flex;
    align-items: center;
    padding: 0 1.25rem;
    font-size: 0.8rem;
    font-weight: 600;
    color: #475569;
    letter-spacing: 0.04em;
    cursor: pointer;
    border-bottom: 2px solid transparent;
    transition: color 0.15s, border-color 0.15s;
    text-transform: uppercase;
    user-select: none;
    white-space: nowrap;
}
.nav-tab:hover { color: #94A3B8; }
.nav-tab.active { color: #F1F5F9; border-bottom-color: #00C2FF; }

/* Nav button overrides — make Streamlit buttons look like nav tabs */
.nav-btn-wrap .stButton > button {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    color: #475569 !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    padding: 0 1rem !important;
    height: 52px !important;
    width: 100% !important;
    transition: color 0.15s, border-color 0.15s !important;
    box-shadow: none !important;
}
.nav-btn-wrap .stButton > button:hover {
    color: #94A3B8 !important;
    background: transparent !important;
    box-shadow: none !important;
}
.nav-btn-wrap.active .stButton > button {
    color: #F1F5F9 !important;
    border-bottom-color: #00C2FF !important;
}

/* ── Sidebar brand ─────────────────────────────────────────────────────────── */
.sb-brand {
    padding: 1.5rem 1rem 1.25rem;
    border-bottom: 1px solid #0E1420;
    margin-bottom: 0.25rem;
}
.sb-brand-name {
    font-size: 1rem;
    font-weight: 800;
    color: #F1F5F9 !important;
    letter-spacing: -0.02em;
}
.sb-brand-sub {
    font-size: 0.7rem;
    color: #334155 !important;
    margin-top: 0.2rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.sb-label {
    font-size: 0.62rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #334155 !important;
    margin: 1.25rem 0 0.4rem 0;
    padding: 0 1rem;
}
.sb-divider { height: 1px; background: #0E1420; margin: 1rem 0; }
.sb-meta {
    padding: 0 1rem;
    font-size: 0.7rem;
    color: #334155 !important;
    line-height: 2;
}
.sb-meta b { color: #475569 !important; font-weight: 600; }

/* Model chip */
.model-chip {
    margin: 0.5rem 1rem 0;
    background: #0A0F1A;
    border: 1px solid #1A2540;
    border-radius: 2px;
    padding: 0.6rem 0.75rem;
    font-size: 0.72rem;
    color: #475569 !important;
    line-height: 1.8;
}
.model-chip b { color: #64748B !important; font-weight: 500; }

/* Speed/accuracy bar */
.tier-bar-wrap {
    margin: 0.5rem 1rem 0;
    background: #0A0F1A;
    border: 1px solid #1A2540;
    border-radius: 2px;
    padding: 0.6rem 0.75rem;
}
.tier-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.62rem;
    color: #334155 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.35rem;
}
.tier-track {
    height: 3px;
    background: #1A2540;
    border-radius: 1px;
    overflow: hidden;
}
.tier-fill {
    height: 3px;
    background: linear-gradient(90deg, #00C2FF, #0080AA);
    border-radius: 1px;
    transition: width 0.4s ease;
}

/* ── Hero section ──────────────────────────────────────────────────────────── */
.hero {
    padding: 3.5rem 0 2.5rem;
    animation: fadeUp 0.5s ease both;
}
.hero-eyebrow {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: #00C2FF;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.hero-eyebrow::before {
    content: '';
    display: inline-block;
    width: 20px;
    height: 2px;
    background: #00C2FF;
}
.hero-headline {
    font-size: 3rem;
    font-weight: 800;
    color: #F1F5F9;
    letter-spacing: -0.04em;
    line-height: 1.1;
    margin-bottom: 1.25rem;
}
.hero-headline span { color: #00C2FF; }
.hero-sub {
    font-size: 0.95rem;
    color: #475569;
    line-height: 1.7;
    max-width: 480px;
    margin-bottom: 2rem;
}
.hero-stats {
    display: flex;
    gap: 2rem;
    padding-top: 1.5rem;
    border-top: 1px solid #0E1420;
}
.hero-stat-val {
    font-size: 1.5rem;
    font-weight: 800;
    color: #F1F5F9;
    letter-spacing: -0.03em;
    line-height: 1;
}
.hero-stat-lbl {
    font-size: 0.68rem;
    color: #334155;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.3rem;
    font-weight: 600;
}

/* ── Condition strip (horizontal) ─────────────────────────────────────────── */
.cond-strip {
    display: grid;
    grid-template-columns: repeat(7, 1fr);
    gap: 1px;
    background: #0E1420;
    border: 1px solid #0E1420;
    margin: 0 0 0 0;
    animation: fadeUp 0.5s ease 0.1s both;
}
.cond-tile {
    background: #0A0F1A;
    padding: 1.1rem 0.9rem;
    border-top: 2px solid transparent;
    transition: background 0.15s, border-color 0.15s;
    cursor: default;
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
}
.cond-tile:hover { background: #0D1525; }
.cond-tile.malignant    { border-top-color: #EF4444; }
.cond-tile.precancerous { border-top-color: #F59E0B; }
.cond-tile.benign       { border-top-color: #22C55E; }
.cond-name {
    font-size: 0.78rem;
    font-weight: 600;
    color: #CBD5E1;
    line-height: 1.3;
}
.cond-short {
    font-size: 0.65rem;
    color: #334155;
    line-height: 1.5;
    flex: 1;
}

/* ── CTA button ────────────────────────────────────────────────────────────── */
.cta-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.75rem;
    padding: 2.5rem 0 0.5rem;
    animation: fadeUp 0.5s ease 0.25s both;
}
.cta-hint {
    font-size: 0.72rem;
    color: #334155;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
}
.cta-arrow {
    font-size: 0.8rem;
    color: #1A2540;
    animation: bounceDown 1.5s ease infinite;
}
@keyframes bounceDown {
    0%, 100% { transform: translateY(0); }
    50%       { transform: translateY(5px); }
}

/* ── Risk pills ────────────────────────────────────────────────────────────── */
.pill {
    display: inline-block;
    padding: 0.12rem 0.5rem;
    border-radius: 2px;
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.pill-malignant    { background: rgba(239,68,68,0.1);  color: #F87171; border: 1px solid rgba(239,68,68,0.2); }
.pill-precancerous { background: rgba(245,158,11,0.1); color: #FBBF24; border: 1px solid rgba(245,158,11,0.2); }
.pill-benign       { background: rgba(34,197,94,0.1);  color: #4ADE80; border: 1px solid rgba(34,197,94,0.2); }

/* ── How it works ──────────────────────────────────────────────────────────── */
.how-section { margin: 2.5rem 0; animation: fadeUp 0.5s ease 0.15s both; }
.section-eyebrow {
    font-size: 0.62rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: #334155;
    margin-bottom: 0.6rem;
}
.section-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #F1F5F9;
    letter-spacing: -0.02em;
    margin-bottom: 1.25rem;
}
.how-strip {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1px;
    background: #0E1420;
    border: 1px solid #0E1420;
}
.how-tile {
    background: #0A0F1A;
    padding: 1.25rem;
    border-top: 2px solid #0E1420;
    transition: border-color 0.15s, background 0.15s;
}
.how-tile:hover { background: #0D1525; border-top-color: #00C2FF; }
.how-num {
    font-size: 0.62rem;
    font-weight: 700;
    color: #00C2FF;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.6rem;
}
.how-title {
    font-size: 0.88rem;
    font-weight: 600;
    color: #E2E8F0;
    margin-bottom: 0.4rem;
}
.how-desc { font-size: 0.75rem; color: #334155; line-height: 1.6; }

/* ── Image panel ───────────────────────────────────────────────────────────── */
.img-panel {
    background: #0A0F1A;
    border: 1px solid #0E1420;
    border-radius: 0;
    padding: 1.25rem;
    animation: fadeUp 0.4s ease both;
}
.panel-label {
    font-size: 0.62rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #334155;
    margin-bottom: 0.75rem;
}
.meta-row {
    display: flex;
    align-items: baseline;
    padding: 0.5rem 0;
    border-bottom: 1px solid #0E1420;
    gap: 1rem;
}
.meta-row:last-child { border-bottom: none; }
.meta-key {
    font-size: 0.7rem;
    color: #334155;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    width: 100px;
    flex-shrink: 0;
}
.meta-val {
    font-size: 0.82rem;
    color: #94A3B8;
    font-variant-numeric: tabular-nums;
}
.status-ready {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.7rem;
    font-weight: 600;
    color: #4ADE80;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.75rem;
}
.status-ready::before {
    content: '';
    width: 6px; height: 6px;
    background: #22C55E;
    border-radius: 50%;
    box-shadow: 0 0 6px rgba(34,197,94,0.6);
}

/* ── Results ───────────────────────────────────────────────────────────────── */
.results-section { animation: fadeUp 0.4s ease both; }

/* Result hero card */
.result-hero {
    border: 1px solid;
    padding: 2rem;
    margin-bottom: 1px;
    position: relative;
    overflow: hidden;
}
.result-hero.malignant    { background: #0C0808; border-color: rgba(239,68,68,0.25); }
.result-hero.precancerous { background: #0C0B06; border-color: rgba(245,158,11,0.25); }
.result-hero.benign       { background: #060C08; border-color: rgba(34,197,94,0.25); }
.result-hero::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.result-hero.malignant::before    { background: #EF4444; }
.result-hero.precancerous::before { background: #F59E0B; }
.result-hero.benign::before       { background: #22C55E; }
.result-eyebrow {
    font-size: 0.62rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: #334155;
    margin-bottom: 0.5rem;
}
.result-name {
    font-size: 2rem;
    font-weight: 800;
    color: #F1F5F9;
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin-bottom: 0.75rem;
}
.result-conf {
    font-size: 3.5rem;
    font-weight: 800;
    letter-spacing: -0.05em;
    line-height: 1;
    font-variant-numeric: tabular-nums;
}
.result-conf.malignant    { color: #F87171; }
.result-conf.precancerous { color: #FBBF24; }
.result-conf.benign       { color: #4ADE80; }
.result-conf-lbl {
    font-size: 0.62rem;
    color: #334155;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
    margin-top: 0.3rem;
}

/* Metrics strip */
.metrics-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: #0E1420;
    border: 1px solid #0E1420;
    border-top: none;
    margin-bottom: 1.5rem;
}
.metric-tile {
    background: #0A0F1A;
    padding: 1rem 1.25rem;
    text-align: center;
    transition: background 0.15s;
}
.metric-tile:hover { background: #0D1525; }
.metric-val {
    font-size: 1.4rem;
    font-weight: 700;
    color: #F1F5F9;
    letter-spacing: -0.02em;
    font-variant-numeric: tabular-nums;
    line-height: 1;
}
.metric-lbl {
    font-size: 0.6rem;
    color: #334155;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 700;
    margin-top: 0.35rem;
}

/* Probability section */
.prob-panel {
    background: #0A0F1A;
    border: 1px solid #0E1420;
    padding: 1.25rem 1.5rem;
    animation: fadeUp 0.5s ease 0.1s both;
}
.prob-panel-title {
    font-size: 0.62rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #334155;
    margin-bottom: 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid #0E1420;
}
.prob-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.5rem 0;
    border-bottom: 1px solid #080C14;
    transition: background 0.1s;
}
.prob-row:last-child { border-bottom: none; }
.prob-name {
    font-size: 0.78rem;
    color: #64748B;
    width: 180px;
    flex-shrink: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.prob-name.top { color: #F1F5F9; font-weight: 600; }
.prob-track {
    flex: 1;
    height: 4px;
    background: #0E1420;
    border-radius: 1px;
    overflow: hidden;
}
.prob-fill {
    height: 4px;
    border-radius: 1px;
    animation: barGrow 0.8s cubic-bezier(0.4,0,0.2,1) both;
}
.prob-pct {
    font-size: 0.75rem;
    color: #475569;
    width: 42px;
    text-align: right;
    flex-shrink: 0;
    font-variant-numeric: tabular-nums;
}
.prob-pct.top { color: #E2E8F0; font-weight: 600; }

/* Clinical panel */
.clinical-panel {
    background: #0A0F1A;
    border: 1px solid #0E1420;
    padding: 1.25rem 1.5rem;
    animation: fadeUp 0.5s ease 0.15s both;
}
.clinical-panel-title {
    font-size: 0.62rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #334155;
    margin-bottom: 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid #0E1420;
}
.clin-block {
    padding: 0.65rem 0;
    border-bottom: 1px solid #080C14;
}
.clin-block:last-child { border-bottom: none; }
.clin-key {
    font-size: 0.6rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #334155;
    margin-bottom: 0.25rem;
}
.clin-val {
    font-size: 0.82rem;
    color: #94A3B8;
    line-height: 1.6;
}
.clin-val.urgent { color: #F87171; font-weight: 500; }
.clin-val.highlight { color: #F1F5F9; font-weight: 600; }

/* Disclaimer */
.disclaimer {
    background: #0A0F1A;
    border: 1px solid #0E1420;
    border-left: 2px solid #1A2540;
    padding: 0.9rem 1.25rem;
    font-size: 0.75rem;
    color: #334155;
    line-height: 1.7;
    margin-top: 1.5rem;
    animation: fadeIn 0.6s ease 0.3s both;
}
.disclaimer b { color: #475569; }

/* Divider */
.section-divider {
    height: 1px;
    background: #0E1420;
    margin: 2rem 0;
}

/* ── Predict Dashboard ──────────────────────────────────────────────────── */
.dash-header { padding: 2.5rem 0 1.5rem; text-align: center; }
.dash-eyebrow { font-size: 0.65rem; font-weight: 700; letter-spacing: 0.18em;
    text-transform: uppercase; color: #00C2FF; margin-bottom: 0.5rem; }
.dash-title { font-size: 2rem; font-weight: 800; color: #F1F5F9; line-height: 1.15; margin-bottom: 0.5rem; }
.dash-sub { font-size: 0.85rem; color: #475569; max-width: 520px; margin: 0 auto; }

.zone-label { font-size: 0.62rem; font-weight: 700; letter-spacing: 0.18em;
    text-transform: uppercase; color: #334155; margin: 1.5rem 0 0.75rem;
    padding-left: 0.25rem; border-left: 2px solid #00C2FF; padding-left: 0.6rem; }
.zone-divider { height: 1px; background: #0E1420; margin: 1.5rem 0; }

/* Model cards */
.model-card { background: #0A0F1A; padding: 1rem 1rem 0.75rem;
    border: 1px solid #0E1420; cursor: pointer;
    transition: background 0.15s, border-color 0.15s; min-height: 130px; }
.model-card:hover { background: #0D1525; border-color: #1E293B; }
.model-card.selected { background: #0D1A2A; border-color: #00C2FF; }
.mc-name { font-size: 0.75rem; font-weight: 700; color: #F1F5F9;
    margin-bottom: 0.3rem; line-height: 1.3; }
.mc-desc { font-size: 0.62rem; color: #475569; margin-bottom: 0.4rem; line-height: 1.4; }
.mc-meta { font-size: 0.58rem; color: #334155; font-family: 'JetBrains Mono', monospace;
    margin-bottom: 0.5rem; }
.mc-tier-wrap { display: flex; align-items: center; gap: 0.5rem; }
.mc-tier-track { flex: 1; height: 3px; background: #0E1420; }
.mc-tier-fill { height: 100%; transition: width 0.4s ease; }
.mc-tier-val { font-size: 0.6rem; font-weight: 700; font-variant-numeric: tabular-nums;
    flex-shrink: 0; }

/* Upload zone */
.upload-zone { border: 1.5px dashed #1E293B; padding: 2rem 1.5rem;
    text-align: center; background: #0A0F1A; margin-bottom: 0.75rem;
    transition: border-color 0.2s; }
.upload-zone:hover { border-color: #00C2FF; }
.upload-icon { font-size: 1.8rem; margin-bottom: 0.5rem; opacity: 0.4; }
.upload-title { font-size: 0.85rem; font-weight: 600; color: #94A3B8; margin-bottom: 0.25rem; }
.upload-hint { font-size: 0.65rem; color: #334155; }

/* Preview panel */
.preview-panel { background: #0A0F1A; border: 1px solid #0E1420; padding: 1rem; min-height: 200px; }
.preview-panel.empty { display: flex; flex-direction: column;
    align-items: center; justify-content: center; min-height: 220px; }
.preview-empty-icon { font-size: 2.5rem; opacity: 0.15; margin-bottom: 0.5rem; }
.preview-empty-text { font-size: 0.72rem; color: #334155; }
.img-meta-strip { display: flex; gap: 1rem; flex-wrap: wrap; margin-top: 0.6rem;
    font-size: 0.6rem; color: #475569; font-family: 'JetBrains Mono', monospace; }

/* Status chips */
.status-chip { display: inline-block; padding: 0.5rem 1rem; font-size: 0.7rem;
    font-weight: 600; letter-spacing: 0.05em; border-radius: 2px; margin-top: 0.25rem; }
.status-chip.waiting { background: #0E1420; color: #475569; border: 1px solid #1E293B; }
.status-chip.ready   { background: #0D1A2A; color: #00C2FF; border: 1px solid #00C2FF44; }

/* Hide Streamlit sidebar entirely on non-sidebar pages */
[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

/* ══════════════════════════════════════════════════════════════════════════
   MOBILE RESPONSIVE BREAKPOINTS
══════════════════════════════════════════════════════════════════════════ */
@media (max-width: 1024px) {
    .cond-strip { grid-template-columns: repeat(4, 1fr); }
}

@media (max-width: 768px) {
    .metrics-strip { flex-wrap: wrap; gap: 0.5rem; display: flex; }
    .metrics-strip > div { min-width: 45%; flex: 1; }
    
    [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { 
      min-width: 48% !important; 
      flex: 1 1 48% !important;
    }
}

@media (max-width: 640px) {
    .cond-strip { grid-template-columns: repeat(2, 1fr); }
    
    .result-hero h1, .result-hero .result-name { font-size: 1.4rem; }
    .result-hero .result-conf { font-size: 2rem; }
    
    .main .block-container, .hero { 
      padding: 1rem !important; 
      max-width: 100% !important; 
    }
    
    .topbar { flex-direction: column; gap: 0.5rem; padding: 0.75rem; height: auto; }
    .topbar-nav { flex-wrap: wrap; justify-content: center; }
    .topbar-right { display: none; }
}

@media (max-width: 480px) {
    .metrics-strip > div { min-width: 100%; }
    
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { 
      min-width: 100% !important; 
      flex: 1 1 100% !important;
    }
}

</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  DATA LAYER
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

HF_MODEL_REPO = "dharshan0025/skin-cancer-models"

ALL_MODEL_FILES = [
    "ensemble_model.h5", "DenseNet201_retrained.h5", "InceptionV3_retrained.h5",
    "DenseNet201_finetuned.h5", "InceptionV3_finetuned.h5",
    "InceptionResNetV2_finetuned.h5", "VGG16_finetuned.h5", "baseline_cnn.h5",
]

# (file, width, height, tier_pct, description)
MODEL_CONFIG = {
    "Baseline CNN":                         ("baseline_cnn.h5",                64,  64,  15, "Fastest · ~25 MB"),
    "DenseNet201 — Finetuned":              ("DenseNet201_finetuned.h5",       256, 192, 55, "Balanced · ~92 MB"),
    "InceptionV3 — Finetuned":             ("InceptionV3_finetuned.h5",       256, 192, 60, "Balanced · ~142 MB"),
    "VGG16 — Finetuned":                   ("VGG16_finetuned.h5",             256, 192, 50, "Classic · ~119 MB"),
    "InceptionResNetV2 — Finetuned":       ("InceptionResNetV2_finetuned.h5", 256, 192, 80, "High accuracy · ~295 MB"),
    "DenseNet201 — Retrained":             ("DenseNet201_retrained.h5",       256, 192, 75, "Full retrain · ~233 MB"),
    "InceptionV3 — Retrained":            ("InceptionV3_retrained.h5",       256, 192, 78, "Full retrain · ~275 MB"),
    "Ensemble (DenseNet201 + InceptionV3)": ("ensemble_model.h5",             256, 192, 95, "Best accuracy · ~171 MB"),
}

CLASS_LABELS = {
    0: "Actinic Keratosis", 1: "Basal Cell Carcinoma", 2: "Benign Keratosis",
    3: "Dermatofibroma",    4: "Melanoma",              5: "Melanocytic Nevi",
    6: "Vascular Lesion",
}

CLASS_INFO = {
    0: {"risk": "Pre-cancerous", "rk": "precancerous", "short": "UV-damaged skin patch",
        "desc": "A rough, scaly patch from years of UV exposure. Can progress to squamous cell carcinoma if untreated.",
        "action": "Consult a dermatologist for cryotherapy, topical medications, or photodynamic therapy.",
        "prevalence": "Very common in fair-skinned individuals over 40.", "bar": "#F59E0B"},
    1: {"risk": "Malignant", "rk": "malignant", "short": "Most common skin cancer",
        "desc": "Most common form of skin cancer. Rarely metastasizes but locally destructive without treatment.",
        "action": "Requires surgical excision or Mohs surgery. Excellent prognosis with early treatment.",
        "prevalence": "~3 million cases annually in the US.", "bar": "#EF4444"},
    2: {"risk": "Benign", "rk": "benign", "short": "Non-cancerous skin growth",
        "desc": "Non-cancerous growths including seborrheic keratoses and solar lentigines. Harmless.",
        "action": "No treatment required. Can be removed for cosmetic reasons.",
        "prevalence": "Extremely common, especially after age 50.", "bar": "#22C55E"},
    3: {"risk": "Benign", "rk": "benign", "short": "Fibrous nodule, usually on legs",
        "desc": "Common benign fibrous nodule, typically found on the legs. Feels like a hard lump under the skin.",
        "action": "No treatment necessary unless symptomatic or for cosmetic concerns.",
        "prevalence": "More common in women, usually on lower legs.", "bar": "#22C55E"},
    4: {"risk": "Malignant — High Risk", "rk": "malignant", "short": "Most dangerous skin cancer",
        "desc": "Most dangerous form of skin cancer. Can metastasize rapidly if not caught early.",
        "action": "URGENT: Immediate dermatologist consultation required. Wide surgical excision needed.",
        "prevalence": "~100,000 new cases annually in the US, 7,000+ deaths.", "bar": "#EF4444"},
    5: {"risk": "Benign", "rk": "benign", "short": "Common moles",
        "desc": "Common moles. Benign melanocyte proliferations. Atypical nevi require monitoring.",
        "action": "Monitor for changes using the ABCDE rule. Regular skin checks recommended.",
        "prevalence": "Most adults have 10–40 moles.", "bar": "#22C55E"},
    6: {"risk": "Benign", "rk": "benign", "short": "Vascular skin lesion",
        "desc": "Includes cherry angiomas, pyogenic granulomas, and other vascular lesions. Generally not dangerous.",
        "action": "Usually no treatment needed. Can be removed via laser or cauterization.",
        "prevalence": "Cherry angiomas increase with age; present in most adults over 30.", "bar": "#22C55E"},
}

# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE — NAV ROUTING
# ══════════════════════════════════════════════════════════════════════════════
if "page" not in st.session_state:
    st.session_state.page = "Home"

NAV_PAGES = ["Home", "Predict", "Models"]

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    import tensorflow as tf
    import logging
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import keras
    keras.config.enable_unsafe_deserialization()
    return tf.keras.models.load_model(path, compile=False, safe_mode=False)

@st.cache_resource(show_spinner=False)
def load_ensemble(models_dir):
    import os
    import tensorflow as tf
    m1 = tf.keras.models.load_model(
        os.path.join(models_dir, "DenseNet201_finetuned.h5"),
        compile=False, safe_mode=False)
    m2 = tf.keras.models.load_model(
        os.path.join(models_dir, "InceptionV3_finetuned.h5"),
        compile=False, safe_mode=False)
    return m1, m2

def safe_predict(model, tensor):
    """Handles Ensemble Lambda output_shape issue in Keras 3"""
    import tensorflow as tf
    try:
        return model.predict(tensor, verbose=0)
    except Exception:
        sub_preds = []
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                sub_preds.append(layer.predict(tensor, verbose=0))
        if len(sub_preds) == 2:
            # Weighted ensemble: DenseNet201-FT (0.40) + InceptionV3-FT (0.60)
            return sub_preds[0] * 0.40 + sub_preds[1] * 0.60
        elif sub_preds:
            return sum(sub_preds) / len(sub_preds)
        raise


def preprocess(img: Image.Image, w: int, h: int) -> np.ndarray:
    arr = np.array(img.convert("RGB").resize((w, h), Image.LANCZOS), dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)





def remove_annotations(img):
    import cv2
    arr = np.array(img.convert("RGB"))
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    red1  = cv2.inRange(hsv, np.array([0,150,100]),   np.array([10,255,255]))
    red2  = cv2.inRange(hsv, np.array([170,150,100]), np.array([180,255,255]))
    blue  = cv2.inRange(hsv, np.array([100,150,100]), np.array([130,255,255]))
    green = cv2.inRange(hsv, np.array([40,150,100]),  np.array([80,255,255]))
    color_mask = cv2.bitwise_or(cv2.bitwise_or(red1, red2), cv2.bitwise_or(blue, green))
    gray    = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges   = cv2.Canny(gray, 50, 150)
    kernel  = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shape_mask = np.zeros(gray.shape, dtype=np.uint8)
    found = False
    h_img, w_img = gray.shape
    for cnt in contours:
        area  = cv2.contourArea(cnt)
        perim = cv2.arcLength(cnt, True)
        if perim == 0 or area < 50 or area > 0.4 * h_img * w_img:
            continue
        circularity = 4 * np.pi * area / (perim ** 2)
        if 0.55 < circularity <= 1.25 and area < 8000:
            cv2.drawContours(shape_mask, [cnt], -1, 255, thickness=cv2.FILLED)
            found = True
    combined_mask = cv2.bitwise_or(shape_mask, color_mask)
    has_annotation = found and int(np.sum(color_mask)) > 1000  # Stricter threshold
    if not has_annotation:
        return img, False
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=3)
    cleaned = cv2.inpaint(arr, combined_mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    return Image.fromarray(cleaned), True


def generate_gradcam_overlay(model, img_pil, model_w, model_h, pred_index=None):
    """True Grad-CAM — auto-detects last Conv2D layer; falls back to gradient saliency."""
    import cv2
    import tensorflow as tf
    try:
        arr = np.array(
            img_pil.convert('RGB').resize((model_w, model_h), Image.LANCZOS),
            dtype=np.float32) / 255.0
        tensor = np.expand_dims(arr, 0)

        # AUTO-DETECT last Conv2D layer — no hardcoded names
        last_conv_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer
                break

        # If no Conv2D found, raise to trigger saliency fallback
        if last_conv_layer is None:
            raise ValueError("No Conv2D layer found — using saliency fallback")

        # Build grad model: inputs → [last_conv_output, final_output]
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[last_conv_layer.output, model.output]
        )

        # GradientTape watching conv layer output
        tensor_tf = tf.cast(tensor, tf.float32)
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(tensor_tf)
            tape.watch(conv_outputs)
            if pred_index is None:
                pred_index = int(tf.argmax(predictions[0]))
            class_score = predictions[:, pred_index]

        # Gradients of class score w.r.t. conv feature maps
        grads = tape.gradient(class_score, conv_outputs)

        # Global average pool gradients → importance weights per channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight each feature map by its gradient importance
        conv_outputs_np = conv_outputs[0].numpy()
        pooled_grads_np = pooled_grads.numpy()
        for i in range(pooled_grads_np.shape[-1]):
            conv_outputs_np[:, :, i] *= pooled_grads_np[i]

        # Mean across channels + ReLU + normalize
        heatmap = np.mean(conv_outputs_np, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / (np.max(heatmap) + 1e-8)

        # Resize heatmap to original image size and colorize
        orig_w, orig_h = img_pil.size
        heatmap_resized = cv2.resize(heatmap, (orig_w, orig_h))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Overlay on ORIGINAL image (not preprocessed)
        orig_arr = np.array(img_pil.convert('RGB'))
        overlay = cv2.addWeighted(orig_arr, 0.55, heatmap_colored, 0.45, 0)
        return Image.fromarray(overlay)

    except Exception:
        # Fallback: gradient saliency — works with ALL Keras 3 models
        try:
            arr = np.array(
                img_pil.convert('RGB').resize((model_w, model_h), Image.LANCZOS),
                dtype=np.float32) / 255.0
            input_var = tf.Variable(arr[np.newaxis, ...])
            with tf.GradientTape() as tape:
                preds = model(input_var, training=False)
                loss = preds[:, pred_index] if pred_index is not None else tf.reduce_max(preds)
            grads = tape.gradient(loss, input_var)
            if grads is None:
                return None
            saliency = tf.reduce_max(tf.abs(grads), axis=-1)[0].numpy()
            s_min, s_max = saliency.min(), saliency.max()
            if s_max - s_min < 1e-8:
                return None
            saliency = (saliency - s_min) / (s_max - s_min)
            orig_arr = np.array(img_pil.convert('RGB'))
            h, w = orig_arr.shape[:2]
            heatmap = cv2.resize(saliency, (w, h))
            heatmap_colored = cv2.applyColorMap(
                np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(orig_arr, 0.55, heatmap_colored, 0.45, 0)
            return Image.fromarray(overlay)
        except Exception:
            return None


def generate_pdf_report(class_label, confidence, uncertainty, model_name,
                         preds, class_labels, gradcam_img, preprocessed_img):
    """Generate a PDF diagnostic report and return raw bytes."""
    from fpdf import FPDF
    import tempfile, os

    # top_idx resolved locally so the probability table can bold the top class
    import numpy as _np
    top_idx_local = int(_np.argmax(list(preds)))

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    def clean_text(text):
        if not isinstance(text, str):
            from numbers import Number
            if isinstance(text, Number): 
                return str(text)
            text = str(text)
        # fpdf uses latin-1, replace common unicode chars that fail
        return text.replace("—", "-").replace("·", "-").replace("🔴", "").replace("🟡", "").replace("🔵", "").replace("✅", "").replace("⚠️", "").replace("σ", "SD").encode('latin-1', 'replace').decode('latin-1')

    # ── HEADER ──────────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 12, clean_text("DermAI - Skin Lesion Analysis Report"), ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, clean_text(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"),
             ln=True, align="C")
    pdf.ln(4)

    # ── DISCLAIMER BOX ──────────────────────────────────────────────────────
    pdf.set_fill_color(254, 242, 242)
    pdf.set_draw_color(220, 38, 38)
    pdf.set_text_color(153, 27, 27)
    pdf.set_font("Helvetica", "B", 9)
    pdf.multi_cell(0, 7,
        clean_text("DISCLAIMER: This report is NOT a medical diagnosis. "
        "DermAI is an AI research tool for educational purposes only. "
        "Always consult a qualified dermatologist for medical advice."),
        border=1, fill=True, align="C")
    pdf.ln(6)

    # ── PRIMARY DIAGNOSIS ───────────────────────────────────────────────────
    pdf.set_text_color(30, 30, 30)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, clean_text("Primary Diagnosis"), ln=True)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)

    unc_level = ('High' if uncertainty >= 0.18
                 else 'Moderate' if uncertainty >= 0.08
                 else 'Low')
    rows = [
        ("Diagnosis",       class_label),
        ("Confidence",      f"{confidence:.1f}%"),
        ("TTA Uncertainty", f"{uncertainty:.4f} ({unc_level})"),
        ("Model Used",      str(model_name).replace("—", "-").replace("·", "-")),
        ("Report Date",     datetime.now().strftime('%Y-%m-%d')),
    ]
    for lbl, val in rows:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(60, 8, clean_text(lbl + ":"), ln=False)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 8, clean_text(val), ln=True)
    pdf.ln(6)

    # ── CLASS PROBABILITY TABLE ─────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, clean_text("Class Probability Breakdown"), ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)

    pdf.set_fill_color(240, 240, 240)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(120, 8, clean_text("Diagnosis Class"), border=1, fill=True)
    pdf.cell(0,   8, clean_text("Probability"),     border=1, fill=True, ln=True)

    sorted_preds = sorted(enumerate(preds), key=lambda x: x[1], reverse=True)
    for idx, prob in sorted_preds:
        pdf.set_font("Helvetica", "B" if idx == top_idx_local else "", 10)
        pdf.cell(120, 7, clean_text(class_labels[idx]), border=1)
        pdf.cell(0,   7, clean_text(f"{prob * 100:.2f}%"), border=1, ln=True)
    pdf.ln(6)

    # ── PREPROCESSING PIPELINE ──────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, clean_text("Preprocessing Pipeline Applied"), ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 10)
    for step in [
        "Blur Detection",
        "Annotation Removal",
        "Hair Removal (DullRazor)",
        "Lighting Normalization (CLAHE)",
        "10-Pass Test-Time Augmentation (TTA)",
    ]:
        pdf.cell(0, 7, clean_text(f"  +  {step}"), ln=True)
    pdf.ln(6)

    # ── PAGE 2: GRAD-CAM IMAGE ──────────────────────────────────────────────
    if gradcam_img is not None:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(0, 10, clean_text("AI Explainability - Grad-CAM Heatmap"), ln=True)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(3)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(100, 100, 100)
        pdf.multi_cell(0, 6,
            clean_text("The heatmap shows which regions the AI focused on when making "
            "its prediction. Red/yellow areas indicate high activation."))

        pdf.ln(4)
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                gradcam_img.save(tmp.name)
                tmp_path = tmp.name
            pdf.image(tmp_path, x=25, w=160)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return bytes(pdf.output())


def detect_blur(img):
    """Returns True if image is too blurry to analyze"""
    import cv2
    gray = cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < 50.0, round(variance, 1)

def remove_hair(img):
    """DullRazor: safe hair removal, preserves skin colors exactly"""
    import cv2
    arr = np.array(img.convert('RGB'))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    hair_pixel_count = int(np.sum(hair_mask > 0))
    if hair_pixel_count < 500:
        return img, False
    result = cv2.inpaint(arr, hair_mask, 3, cv2.INPAINT_TELEA)
    return Image.fromarray(result), True

def correct_lighting(img):
    """Ultra-gentle CLAHE - only brightens dark images"""
    import cv2
    arr = np.array(img.convert('RGB'))
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    avg_brightness = float(np.mean(l))
    if avg_brightness > 100:
        return img
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
    l = clahe.apply(l)
    result = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

def tta_predict(model, img: "Image.Image", w: int, h: int, n: int = 10, model2=None):
    """Run Test-Time Augmentation — n passes with random flips/rotations."""
    import cv2
    arr = np.array(img.convert("RGB").resize((w, h), Image.LANCZOS), dtype=np.float32) / 255.0
    preds_list = []
    for _ in range(n):
        aug = arr.copy()
        # Random horizontal flip
        if np.random.rand() > 0.5:
            aug = aug[:, ::-1, :]
        # Random vertical flip
        if np.random.rand() > 0.5:
            aug = aug[::-1, :, :]
        # Random 90-degree rotation
        k = np.random.randint(0, 4)
        aug = np.rot90(aug, k)
        if aug.shape[0] != h or aug.shape[1] != w:
            aug = np.array(
                Image.fromarray((aug * 255).astype(np.uint8)).resize((w, h), Image.LANCZOS),
                dtype=np.float32) / 255.0
        # Random brightness shift
        aug = np.clip(aug + np.random.uniform(-0.05, 0.05), 0, 1)
        tensor = np.expand_dims(aug, 0)
        if model2 is not None:
            p1 = model.predict(tensor, verbose=0)[0]
            p2 = model2.predict(tensor, verbose=0)[0]
            # Weighted ensemble: DenseNet201-FT (0.40) + InceptionV3-FT (0.60)
            preds_list.append(p1 * 0.40 + p2 * 0.60)
        else:
            preds_list.append(model.predict(tensor, verbose=0)[0])
    preds_arr   = np.array(preds_list)          # shape (n, 7)
    mean_preds  = preds_arr.mean(axis=0)        # average prediction
    std_preds   = preds_arr.std(axis=0)         # per-class std deviation
    uncertainty = float(std_preds.max())        # max std = overall uncertainty
    return mean_preds, uncertainty


def preprocess_pipeline(img):
    """Blur detection, hair removal, lighting correction."""
    import cv2
    arr = np.array(img.convert("RGB"))
    issues = []

    # 1. Blur Detection (Laplacian variance)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_blurry = blur_score < 80.0
    if is_blurry:
        issues.append(("blur", blur_score))

    # 2. Hair Removal (DullRazor algorithm)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_h)
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    hair_pixels = int(np.sum(hair_mask > 0))
    has_hair = hair_pixels > 500
    if has_hair:
        arr = cv2.inpaint(arr, hair_mask, inpaintRadius=6, flags=cv2.INPAINT_TELEA)
        issues.append(("hair", hair_pixels))

    # 3. Lighting Correction (CLAHE on L channel)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    mean_l = float(l.mean())
    needs_correction = mean_l < 80 or mean_l > 200
    if needs_correction:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        arr = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        issues.append(("lighting", mean_l))

    return Image.fromarray(arr), issues, is_blurry

def pill(rk: str, label: str) -> str:
    return f"<span class='pill pill-{rk}'>{label}</span>"


def ensure_models() -> None:
    if st.session_state.get("_ready"):
        return
    missing = [f for f in ALL_MODEL_FILES if not os.path.exists(os.path.join(MODELS_DIR, f))]
    if missing:
        try:
            from huggingface_hub import hf_hub_download
            with st.status(f"⬇️ Downloading {len(missing)} model(s)…", expanded=True) as status:
                bar = st.progress(0)
                for i, f in enumerate(missing):
                    st.write(f"Fetching `{f}` from HuggingFace…")
                    hf_hub_download(repo_id=HF_MODEL_REPO, filename=f, local_dir=MODELS_DIR)
                    bar.progress((i + 1) / len(missing), text=f"✓ {f}")
                status.update(label="✅ All models ready!", state="complete", expanded=False)
                bar.empty()
        except Exception as e:
            st.warning(f"⚠️ Model download failed: {e}")
    st.session_state["_ready"] = True


# ── No sidebar: all controls live on the Predict page ──────────────────────
uploaded   = None
run_btn    = False
model_name = list(MODEL_CONFIG.keys())[0]
mfile, mw, mh, tier_pct, mdesc = MODEL_CONFIG[model_name]
model_path = os.path.join(MODELS_DIR, mfile)


# ══════════════════════════════════════════════════════════════════════════════
#  TOPBAR WITH NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════
ensure_models()

# Topbar: brand left | nav center | badges right
st.markdown("""
<div class='topbar'>
    <div class='topbar-left'>
        <div class='topbar-dot'></div>
        <span class='topbar-name'>DermAI</span>
        <span class='topbar-version'>v2.0</span>
    </div>
    <div class='topbar-nav' id='nav-placeholder'></div>
    <div class='topbar-right'>
        <span class='topbar-badge'>HAM10000</span>
        <span class='topbar-badge'>7 Classes</span>
        <span class='topbar-badge'>8 Models</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Nav buttons rendered as Streamlit columns (positioned below topbar, styled to look inline)
_nav_cols = st.columns([3, 1, 1, 1, 3])
with _nav_cols[1]:
    _active0 = "active" if st.session_state.page == "Home" else ""
    st.markdown(f"<div class='nav-btn-wrap {_active0}'>", unsafe_allow_html=True)
    if st.button("Home", key="nav_home", use_container_width=True):
        st.session_state.page = "Home"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

with _nav_cols[2]:
    _active1 = "active" if st.session_state.page == "Predict" else ""
    st.markdown(f"<div class='nav-btn-wrap {_active1}'>", unsafe_allow_html=True)
    if st.button("Predict", key="nav_predict", use_container_width=True):
        st.session_state.page = "Predict"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

with _nav_cols[3]:
    _active2 = "active" if st.session_state.page == "Models" else ""
    st.markdown(f"<div class='nav-btn-wrap {_active2}'>", unsafe_allow_html=True)
    if st.button("Models", key="nav_models", use_container_width=True):
        st.session_state.page = "Models"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# Thin separator below nav buttons
st.markdown("<div style='height:1px;background:#0E1420;margin:0 -2rem;'></div>", unsafe_allow_html=True)

current_page = st.session_state.page


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if current_page == "Home":

    st.markdown("""
    <div class='hero' style='text-align:center;padding:3.5rem 0 2rem;'>
        <div class='hero-eyebrow' style='justify-content:center;'>Medical AI &middot; Dermatology</div>
        <div class='hero-headline' style='font-size:3.2rem;max-width:700px;margin:0 auto 1.25rem;'>
            Skin Cancer Detection,<br><span>Powered by AI</span>
        </div>
        <div class='hero-sub' style='max-width:560px;margin:0 auto 2rem;text-align:center;'>
            Upload a dermoscopy image and receive an instant AI-powered classification
            across 7 lesion types \u2014 from benign moles to high-risk melanoma.
        </div>
        <div class='hero-stats' style='justify-content:center;'>
            <div>
                <div class='hero-stat-val'>10K+</div>
                <div class='hero-stat-lbl'>Training Images</div>
            </div>
            <div style='width:1px;background:#0E1420;'></div>
            <div>
                <div class='hero-stat-val'>7</div>
                <div class='hero-stat-lbl'>Lesion Classes</div>
            </div>
            <div style='width:1px;background:#0E1420;'></div>
            <div>
                <div class='hero-stat-val'>8</div>
                <div class='hero-stat-lbl'>AI Models</div>
            </div>
            <div style='width:1px;background:#0E1420;'></div>
            <div>
                <div class='hero-stat-val'>HAM10000</div>
                <div class='hero-stat-lbl'>Dataset</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-eyebrow' style='margin-bottom:0.5rem;'>What DermAI Detects</div>", unsafe_allow_html=True)
    tiles_html = "<div class='cond-strip'>"
    for idx, label in CLASS_LABELS.items():
        info = CLASS_INFO[idx]
        p = pill(info["rk"], info["risk"])
        tiles_html += f"""
        <div class='cond-tile {info["rk"]}'>
            <div class='cond-name'>{label}</div>
            <div class='cond-short'>{info["short"]}</div>
            {p}
        </div>"""
    tiles_html += "</div>"
    st.markdown(tiles_html, unsafe_allow_html=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='how-section'>
        <div class='section-eyebrow'>Workflow</div>
        <div class='section-title'>How It Works</div>
        <div class='how-strip'>
            <div class='how-tile'>
                <div class='how-num'>Step 01</div>
                <div class='how-title'>Upload a Skin Image</div>
                <div class='how-desc'>Go to the Predict tab, upload a JPG, PNG, or BMP dermoscopy image. Standard clinical photos also work.</div>
            </div>
            <div class='how-tile'>
                <div class='how-num'>Step 02</div>
                <div class='how-title'>Select an AI Model</div>
                <div class='how-desc'>Choose from 8 trained architectures \u2014 from the fast Baseline CNN to the high-accuracy Ensemble model.</div>
            </div>
            <div class='how-tile'>
                <div class='how-num'>Step 03</div>
                <div class='how-title'>Review Diagnostic Report</div>
                <div class='how-desc'>Get a full probability breakdown, confidence score, inference time, and structured clinical guidance.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='cta-wrap'>", unsafe_allow_html=True)
    st.markdown("<div class='cta-hint'>Click Predict in the navigation bar to start</div>", unsafe_allow_html=True)
    st.markdown("<div class='cta-arrow'>&#9660;</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='disclaimer'>
        <b>Medical Disclaimer:</b> DermAI is a research and educational tool. It is not a substitute for professional
        medical advice, diagnosis, or treatment. Always consult a qualified dermatologist for any skin concerns.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: PREDICT  — Full-page dashboard
# ══════════════════════════════════════════════════════════════════════════════
elif current_page == "Predict":

    # ── Section header ─────────────────────────────────────────────────────────
    st.markdown("""
    <div class='dash-header'>
        <div class='dash-eyebrow'>AI Diagnostic Tool</div>
        <div class='dash-title'>Skin Lesion Analysis Dashboard</div>
        <div class='dash-sub'>Select a model, upload a dermoscopy image, and run the analysis — all in one place.</div>
    </div>
    """, unsafe_allow_html=True)

    # ── ZONE 1: Model selector cards ──────────────────────────────────────────
    st.markdown("<div class='zone-label'>① Select Model</div>", unsafe_allow_html=True)

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = list(MODEL_CONFIG.keys())[0]

    model_keys = list(MODEL_CONFIG.keys())
    # Render model cards in rows of 4
    for row_start in range(0, len(model_keys), 4):
        row_keys = model_keys[row_start:row_start+4]
        cols = st.columns(len(row_keys), gap="small")
        for col, mkey in zip(cols, row_keys):
            mf_, mw_, mh_, tier_, mdesc_ = MODEL_CONFIG[mkey]
            is_sel = st.session_state.selected_model == mkey
            tier_color = "#EF4444" if tier_ < 30 else "#F59E0B" if tier_ < 60 else "#22C55E" if tier_ < 85 else "#00C2FF"
            sel_class = "model-card selected" if is_sel else "model-card"
            with col:
                st.markdown(f"""
                <div class='{sel_class}' style='border-top:2px solid {tier_color};'>
                    <div class='mc-name'>{mkey}</div>
                    <div class='mc-desc'>{mdesc_}</div>
                    <div class='mc-meta'>{mw_}×{mh_} px</div>
                    <div class='mc-tier-wrap'>
                        <div class='mc-tier-track'>
                            <div class='mc-tier-fill' style='width:{tier_}%;background:{tier_color};'></div>
                        </div>
                        <div class='mc-tier-val' style='color:{tier_color};'>{tier_}%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"Select", key=f"sel_{mkey}", use_container_width=True):
                    st.session_state.selected_model = mkey
                    st.rerun()

    # Resolve selected model config
    sel_model_name = st.session_state.selected_model
    mfile, mw, mh, tier_pct, mdesc = MODEL_CONFIG[sel_model_name]
    model_path = os.path.join(MODELS_DIR, mfile)

    st.markdown("<div class='zone-divider'></div>", unsafe_allow_html=True)

    # ── ZONE 2: Upload + Preview ───────────────────────────────────────────────
    st.markdown("<div class='zone-label'>② Upload Image</div>", unsafe_allow_html=True)

    col_up, col_prev = st.columns([1, 1], gap="large")

    with col_up:
        st.markdown("""
        <div class='upload-zone'>
            <div class='upload-icon'>⬆</div>
            <div class='upload-title'>Drop or Browse</div>
            <div class='upload-hint'>JPG · PNG · BMP · TIFF · up to 10 MB</div>
        </div>
        """, unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload dermoscopy image",
            type=["jpg", "jpeg", "png", "bmp", "tiff"],
            label_visibility="collapsed",
        )

    # ── Upload validation ──────────────────────────────────────────────────────
    if uploaded is not None:
        if uploaded.size > 10 * 1024 * 1024:
            st.error("⚠️ File too large. Please upload an image under 10 MB.")
            st.stop()
        if uploaded.size == 0:
            st.error("⚠️ Uploaded file appears to be empty or corrupted.")
            st.stop()
        try:
            _img_check = Image.open(uploaded)
            _img_check.verify()          # Exhausts stream — must seek(0) after
            uploaded.seek(0)             # Reset stream pointer before real open
        except Exception:
            st.error("⚠️ Could not read this image. Please upload a valid JPG or PNG.")
            st.stop()

    with col_prev:
        if uploaded is not None:
            image = Image.open(uploaded)  # Safe: stream reset by seek(0) above
            iw, ih = image.size
            size_kb = uploaded.size / 1024
            st.markdown("""
            <div class='preview-panel'>
                <div class='panel-label'>Image Preview</div>
            """, unsafe_allow_html=True)
            
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                st.markdown("**Original Image**")
                st.image(image, use_container_width=True)
            with p_col2:
                st.markdown("**Processed Image**")
                if "preprocessed_image" in st.session_state:
                    st.image(st.session_state["preprocessed_image"], use_container_width=True)
                else:
                    st.info("Processing preview appears after analysis")
                    
            st.markdown(f"""
                <div class='img-meta-strip'>
                    <span>{uploaded.name}</span>
                    <span>{iw}×{ih} px</span>
                    <span>{image.mode}</span>
                    <span>{size_kb:.1f} KB</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="preview-panel empty">
                <div class="preview-empty-icon">🩺</div>
                <div class="preview-empty-text">Image preview will appear here</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='zone-divider'></div>", unsafe_allow_html=True)

    # ── ZONE 3: Run button + status ────────────────────────────────────────────
    st.markdown("<div class='zone-label'>③ Run Analysis</div>", unsafe_allow_html=True)

    col_run, col_status = st.columns([1, 2], gap="large")
    with col_run:
        run_btn = st.button(
            "▶  Run Analysis",
            use_container_width=True,
            type="primary",
            disabled=(uploaded is None),
        )
    with col_status:
        if uploaded is None:
            st.markdown("<div class='status-chip waiting'>Waiting for image upload…</div>", unsafe_allow_html=True)
        elif not run_btn:
            st.markdown(f"<div class='status-chip ready'>Ready · Model: {sel_model_name}</div>", unsafe_allow_html=True)

    # ── ZONE 4: Results ────────────────────────────────────────────────────────
    if uploaded is not None and run_btn:
        if not os.path.exists(model_path):
            st.error(f"Model file **{mfile}** not found. Ensure models are downloaded.")
            st.stop()

        with st.spinner(f"Running inference with {sel_model_name}…"):
            t0 = time.perf_counter()
            try:
                is_ensemble = sel_model_name == "Ensemble (DenseNet201 + InceptionV3)"
                if is_ensemble:
                    model, model2 = load_ensemble(MODELS_DIR)
                else:
                    model = load_model(model_path)
                    model2 = None
                # BLUR CHECK
                is_blurry, blur_score = detect_blur(image)
                if is_blurry:
                    st.error(f"⚠️ Image too blurry (sharpness: {blur_score}) — upload a clearer photo for accurate results.")
                    st.stop()
            
                # ANNOTATION REMOVAL
                clean_image, had_annotation = remove_annotations(image)
                if had_annotation:
                    st.markdown('<div style="background:rgba(0,194,255,0.08);border:1px solid rgba(0,194,255,0.3);padding:0.5rem 1rem;font-size:0.75rem;color:#00C2FF;margin-bottom:0.75rem">✔ Drawn annotation detected and removed before analysis</div>', unsafe_allow_html=True)
            
                # HAIR REMOVAL
                clean_image, hair_removed = remove_hair(clean_image)
                if hair_removed:
                    st.markdown('<div style="background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.3);padding:0.5rem 1rem;font-size:0.75rem;color:#4ADE80;margin-bottom:0.75rem">✂️ Hair detected and removed for cleaner analysis</div>', unsafe_allow_html=True)
            
                # LIGHTING CORRECTION
                clean_image = correct_lighting(clean_image)
            
                # SAVE CRISP PREVIEW (512x384 for display)
                st.session_state["preprocessed_image"] = clean_image.resize((512, 384), Image.LANCZOS)
            
                # MODEL INPUT
                model_input = clean_image.resize((mw, mh), Image.LANCZOS)
                tensor = preprocess(model_input, mw, mh)
                preds, tta_uncertainty = tta_predict(model, clean_image, mw, mh, n=10, model2=model2)
                try:
                    # For ensemble, always use DenseNet201 (primary model) for Grad-CAM
                    primary_model = model[0] if isinstance(model, tuple) else model
                    gradcam_img = generate_gradcam_overlay(primary_model, clean_image, mw, mh, int(np.argmax(preds)))
                except Exception:
                    gradcam_img = None
                st.session_state["gradcam_image"] = gradcam_img
                elapsed = time.perf_counter() - t0
            except Exception as e:
                st.error(f"Inference failed: {e}")
                st.stop()

        top_idx    = int(np.argmax(preds))
        top_conf   = float(preds[top_idx]) * 100
        top_label  = CLASS_LABELS[top_idx]
        info       = CLASS_INFO[top_idx]
        rk         = info["rk"]
        second_idx = int(np.argsort(preds)[-2])
        second_conf= float(preds[second_idx]) * 100
        second_lbl = CLASS_LABELS[second_idx]

        st.markdown("<div class='zone-divider'></div>", unsafe_allow_html=True)
        st.markdown("<div class='zone-label'>④ Diagnostic Report</div>", unsafe_allow_html=True)

        # Result hero
        pill_html = pill(rk, info["risk"])
        st.markdown(f"""
        <div class='result-hero {rk}'>
            <div style='display:flex;align-items:flex-start;justify-content:space-between;gap:2rem;'>
                <div>
                    <div class='result-eyebrow'>Primary Diagnosis</div>
                    <div class='result-name'>{top_label}</div>
                    {pill_html}
                </div>
                <div style='text-align:right;flex-shrink:0;'>
                    <div class='result-conf {rk}'>{top_conf:.1f}%</div>
                    <div class='result-conf-lbl'>Confidence Score</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Metrics strip
        risk_display = info["risk"].replace(" — High Risk", "")
        st.markdown(f"""
        <div class='metrics-strip'>
            <div class='metric-tile'>
                <div class='metric-val'>{top_conf:.1f}%</div>
                <div class='metric-lbl'>Confidence</div>
            </div>
            <div class='metric-tile'>
                <div class='metric-val'>{elapsed:.2f}s</div>
                <div class='metric-lbl'>Inference Time</div>
            </div>
            <div class='metric-tile'>
                <div class='metric-val'>{second_conf:.1f}%</div>
                <div class='metric-lbl'>2nd Prediction</div>
            </div>
            <div class='metric-tile'>
                <div class='metric-val'>{risk_display}</div>
                <div class='metric-lbl'>Risk Level</div>
            </div>
            <div class='metric-tile'>
                <div class='metric-val'>{sel_model_name.split("—")[0].strip()}</div>
                <div class='metric-lbl'>Model Used</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Prob + Clinical
        # ── TTA Uncertainty Badge ──
        if tta_uncertainty < 0.08:
            unc_color='#22C55E'; unc_icon='✅'; unc_label='High Confidence'
            unc_desc='All 10 augmented passes agreed — result is stable.'
        elif tta_uncertainty < 0.18:
            unc_color='#F59E0B'; unc_icon='⚠️'; unc_label='Moderate Uncertainty'
            unc_desc='Some variation across passes — consider a second opinion.'
        else:
            unc_color='#EF4444'; unc_icon='🔴'; unc_label='High Uncertainty'
            unc_desc='High variation — please consult a dermatologist.'
        st.markdown(
            f'<div style="background:rgba(0,0,0,0.3);border:1px solid {unc_color}44;'
            f'border-left:3px solid {unc_color};padding:0.75rem 1.25rem;'
            f'margin:0.75rem 0;display:flex;align-items:center;gap:1rem;>'
            f'<span style="font-size:1.3rem">{unc_icon}</span>'
            f'<div><div style="font-size:0.75rem;font-weight:700;color:{unc_color};'
            f'text-transform:uppercase;letter-spacing:0.08em">{unc_label}'
            f' &nbsp;·&nbsp; σ = {tta_uncertainty:.3f}</div>'
            f'<div style="font-size:0.72rem;color:#64748B;margin-top:0.2rem">{unc_desc}</div>'
            f'</div></div>',
            unsafe_allow_html=True)

        # ── PDF Report Download ───────────────────────────────────────────────
        try:
            pdf_bytes = generate_pdf_report(
                class_label      = CLASS_LABELS[top_idx],
                confidence       = top_conf,
                uncertainty      = tta_uncertainty,
                model_name       = sel_model_name,
                preds            = preds,
                class_labels     = CLASS_LABELS,
                gradcam_img      = st.session_state.get("gradcam_image"),
                preprocessed_img = st.session_state.get("preprocessed_image"),
            )
            st.download_button(
                label               = "📄 Download Diagnosis Report (PDF)",
                data                = pdf_bytes,
                file_name           = f"DermAI_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime                = "application/pdf",
                use_container_width = True,
            )
        except Exception as _pdf_err:
            print(f"PDF GENERATION ERROR: {_pdf_err}")
            st.caption(f"PDF generation unavailable: {_pdf_err}")


        st.markdown('<div class="zone-label">🔬 AI Explainability — Grad-CAM</div>', unsafe_allow_html=True)
        gcam = st.session_state.get("gradcam_image", None)
        if gcam is not None:
            gc1, gc2 = st.columns(2)
            with gc1:
                st.markdown("**Original Image**")
                st.image(clean_image, use_container_width=True)
            with gc2:
                st.markdown("**AI Focus Heatmap**")
                st.image(gcam, use_container_width=True)
            st.markdown(
                '<div style="font-size:0.72rem;color:#475569;padding:0.5rem 0.75rem;'
                'background:#0A0F1A;border:1px solid #0E1420;margin-bottom:0.75rem">'
                '🔴 <b style="color:#F87171">Red</b> = highest AI focus &nbsp;|&nbsp;'
                '🟡 <b style="color:#FBBF24">Yellow</b> = moderate &nbsp;|&nbsp;'
                '🔵 <b style="color:#60A5FA">Blue</b> = low attention</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                '<div style="font-size:0.75rem;color:#334155;padding:0.75rem 1rem;'
                'background:#0A0F1A;border:1px solid #0E1420;margin-bottom:0.75rem">'
                'Grad-CAM not available for this model.</div>',
                unsafe_allow_html=True)
        st.markdown('<div class="zone-divider"></div>', unsafe_allow_html=True)

        # CBIR Similar Cases - P2.1
        try:
            from cbir_engine import init_cbir
            with st.expander("🔍 Similar Cases (CBIR)", expanded=False):
                engine = init_cbir()
                if engine.load_index():
                    similar_cases = engine.find_similar(image, top_k=5)
                    if similar_cases:
                        st.info(f"🎯 Found {len(similar_cases)} visual matches")
                        _CBIR_BADGE_COLORS = {
                            'Melanoma':              ('#dc2626', '#fff'),
                            'Basal Cell Carcinoma':  ('#dc2626', '#fff'),
                            'Actinic Keratosis':     ('#b45309', '#fff'),
                            'Benign Keratosis':      ('#0284c7', '#fff'),
                            'Dermatofibroma':        ('#0369a1', '#fff'),
                            'Vascular Lesion':       ('#7c3aed', '#fff'),
                            'Melanocytic Nevi':      ('#16a34a', '#fff'),
                        }
                        for i, case in enumerate(similar_cases):
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                if case.get('path') and os.path.exists(case['path']):
                                    st.image(case['path'], width=100)
                                else:
                                    st.markdown("🖼️", unsafe_allow_html=True)
                            with col2:
                                diag  = case.get('diagnosis', 'Unknown')
                                sim   = case.get('similarity', 0.0)
                                bg, fg = _CBIR_BADGE_COLORS.get(diag, ('#6b7280', '#fff'))
                                st.markdown(
                                    f"<span style='background:{bg};color:{fg};"
                                    f"padding:3px 10px;border-radius:12px;"
                                    f"font-size:0.82em;font-weight:600;'>"
                                    f"{diag}</span>"
                                    f"&nbsp;&nbsp;<span style='font-size:1.1em;"
                                    f"font-weight:700;color:#f1f5f9;'>{sim:.1%}</span>",
                                    unsafe_allow_html=True,
                                )
                                st.caption(case.get('image_id', ''))
                    else:
                        st.warning("No similar cases found")
                else:
                    st.info("📥 Build CBIR index first: streamlit run build_cbir_index.py")
        except Exception as e:
            st.error(f"CBIR Module Error: {str(e)}")

        col_prob, col_clin = st.columns([1, 1], gap="large")

        with col_prob:
            st.markdown("<div class='prob-panel'><div class='prob-panel-title'>Class Probabilities</div>", unsafe_allow_html=True)
            for idx in np.argsort(preds)[::-1]:
                label_  = CLASS_LABELS[idx]
                pct     = float(preds[idx]) * 100
                color   = CLASS_INFO[idx]["bar"]
                is_top  = idx == top_idx
                nc = "prob-name top" if is_top else "prob-name"
                pc = "prob-pct top"  if is_top else "prob-pct"
                st.markdown(f"""
                <div class='prob-row'>
                    <div class='{nc}'>{label_}</div>
                    <div class='prob-track'>
                        <div class='prob-fill' style='width:{pct:.2f}%;background:{color};--target-w:{pct:.2f}%;'></div>
                    </div>
                    <div class='{pc}'>{pct:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_clin:
            action_cls = "clin-val urgent" if rk == "malignant" else "clin-val"
            ts = datetime.now().strftime("%d %b %Y, %H:%M:%S")
            st.markdown(f"""
            <div class='clinical-panel'>
                <div class='clinical-panel-title'>Clinical Information</div>
                <div class='clin-block'>
                    <div class='clin-key'>Diagnosis</div>
                    <div class='clin-val highlight'>{top_label}</div>
                </div>
                <div class='clin-block'>
                    <div class='clin-key'>Risk Category</div>
                    <div class='clin-val'>{info["risk"]}</div>
                </div>
                <div class='clin-block'>
                    <div class='clin-key'>Description</div>
                    <div class='clin-val'>{info["desc"]}</div>
                </div>
                <div class='clin-block'>
                    <div class='clin-key'>Recommended Action</div>
                    <div class='{action_cls}'>{info["action"]}</div>
                </div>
                <div class='clin-block'>
                    <div class='clin-key'>Prevalence</div>
                    <div class='clin-val'>{info["prevalence"]}</div>
                </div>
                <div class='clin-block'>
                    <div class='clin-key'>2nd Most Likely</div>
                    <div class='clin-val'>{second_lbl} — {second_conf:.1f}%</div>
                </div>
                <div class='clin-block'>
                    <div class='clin-key'>Analyzed At</div>
                    <div class='clin-val'>{ts}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class='disclaimer'>
            <b>Medical Disclaimer:</b> DermAI is a research and educational tool built on the HAM10000 dataset.
            It is not a substitute for professional medical advice, diagnosis, or treatment.
            Always consult a qualified dermatologist for any skin concerns.
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: MODELS
# ══════════════════════════════════════════════════════════════════════════════
elif current_page == "Models":
    st.markdown("""
    <div class='hero' style='text-align:center;padding:3rem 0 1.5rem;'>
        <div class='hero-eyebrow' style='justify-content:center;'>Architecture Overview</div>
        <div class='hero-headline' style='font-size:2.2rem;max-width:600px;margin:0 auto 1rem;'>
            8 AI Models,<br><span>One Platform</span>
        </div>
        <div class='hero-sub' style='max-width:480px;margin:0 auto;text-align:center;'>
            From lightweight CNNs to state-of-the-art ensemble architectures \u2014 choose the right model for your use case.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-eyebrow' style='margin-bottom:0.5rem;'>Available Models</div>", unsafe_allow_html=True)

    for name, (mf, mw_, mh_, tier, desc) in MODEL_CONFIG.items():
        tier_color = "#EF4444" if tier < 30 else "#F59E0B" if tier < 60 else "#22C55E" if tier < 85 else "#00C2FF"
        st.markdown(f"""
        <div class='how-tile' style='margin-bottom:1px;border-top:2px solid {tier_color};padding:1.25rem 1.5rem;'>
            <div style='display:flex;justify-content:space-between;align-items:flex-start;gap:2rem;'>
                <div>
                    <div class='how-title' style='font-size:0.95rem;'>{name}</div>
                    <div class='how-desc' style='margin-top:0.3rem;'>{desc} &nbsp;&middot;&nbsp; Input: {mw_}\u00d7{mh_} px &nbsp;&middot;&nbsp; File: {mf}</div>
                </div>
                <div style='text-align:right;flex-shrink:0;'>
                    <div style='font-size:1.1rem;font-weight:800;color:{tier_color};font-variant-numeric:tabular-nums;'>{tier}%</div>
                    <div style='font-size:0.62rem;color:#334155;text-transform:uppercase;letter-spacing:0.08em;'>Accuracy Tier</div>
                </div>
            </div>
            <div class='tier-track' style='margin-top:0.75rem;'>
                <div class='tier-fill' style='width:{tier}%;background:{tier_color};'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='disclaimer' style='margin-top:1.5rem;'>
        <b>Note:</b> Accuracy tiers are relative indicators based on architecture complexity and training depth.
        For clinical use, always prefer the Ensemble model. Individual model performance may vary by lesion type.
    </div>
    """, unsafe_allow_html=True)
