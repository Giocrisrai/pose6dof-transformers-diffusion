"""Streamlit dashboard interactivo del TFM.

Ejecucion:
    .venv/bin/streamlit run dashboard.py

Muestra:
- Resumen ejecutivo de hipotesis aceptadas
- Metricas FP con bootstrap CI 95%
- Curvas de robustez (oclusion + ruido)
- Profiling pipeline (cuello de botella)
- Diversidad multimodal Diffusion
- Convergencia PBVS
- Per-object analysis
- Video demo MP4 + diagrama arquitectura
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parent

st.set_page_config(
    page_title="TFM Pose 6-DoF Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS custom — tema profesional cohesionado
st.markdown("""
<style>
    /* Tema principal */
    .stApp {
        background: linear-gradient(180deg, #F8FAFC 0%, #FFFFFF 100%);
        color: #0F172A;
    }

    /* === GARANTIZAR LEGIBILIDAD DEL CUERPO DE TEXTO ===
       Streamlit a veces hereda color claro del sistema; forzamos
       explicitamente texto oscuro sobre fondo claro en todo el body. */
    .stApp, .stApp p, .stApp li, .stApp span, .stApp div, .stApp label,
    .stApp .stMarkdown, .stApp .stMarkdown p, .stApp .stMarkdown li,
    .stApp .stMarkdown ul, .stApp .stMarkdown ol,
    .stApp .stMarkdown span, .stApp .stMarkdown strong, .stApp .stMarkdown b,
    .stApp .stText, .stApp .stCaption,
    .stApp [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stMarkdownContainer"] p,
    .stApp [data-testid="stMarkdownContainer"] li,
    .stApp [data-testid="stMarkdownContainer"] span,
    .stApp [data-testid="stMarkdownContainer"] strong,
    .stApp [data-testid="stExpander"] p,
    .stApp [data-testid="stExpander"] li {
        color: #0F172A !important;
    }
    /* Tablas legibles */
    .stApp table, .stApp table td, .stApp table th {
        color: #0F172A !important;
    }
    .stApp table th { background: #F1F5F9 !important; font-weight: 700 !important; }

    /* === EXCEPCION: bloques de codigo tienen fondo oscuro ===
       Aqui forzamos texto CLARO. Especificidad maxima para ganar
       sobre la regla global anterior. */
    .stApp pre, .stApp pre *, .stApp pre code, .stApp pre code *,
    .stApp [data-testid="stCodeBlock"], .stApp [data-testid="stCodeBlock"] *,
    .stApp [data-testid="stCodeBlock"] code,
    .stApp [data-testid="stCodeBlock"] code *,
    .stApp .stCode, .stApp .stCode *,
    .stApp .highlight, .stApp .highlight *,
    .stApp .highlight .o, .stApp .highlight .n,
    .stApp .highlight .k, .stApp .highlight .s,
    .stApp .highlight .p, .stApp .highlight .nb {
        color: #E2E8F0 !important;
    }
    .stApp pre, .stApp [data-testid="stCodeBlock"], .stApp .stCode {
        background: #0F172A !important;
        border-radius: 8px !important;
        padding: 12px 16px !important;
    }
    /* Inline code (entre backticks) — fondo claro y texto oscuro */
    .stApp p code, .stApp li code,
    .stApp [data-testid="stMarkdownContainer"] p code,
    .stApp [data-testid="stMarkdownContainer"] li code {
        background: #F1F5F9 !important;
        color: #0F172A !important;
        padding: 1px 6px !important;
        border-radius: 4px !important;
        font-size: 0.9em !important;
    }
    /* Excepcion: el sidebar mantiene texto claro porque es oscuro.
       Especificidad alta para ganar sobre .stApp .stMarkdown p (0,2,1) */
    .stApp [data-testid="stSidebar"] *,
    .stApp [data-testid="stSidebar"] p,
    .stApp [data-testid="stSidebar"] li,
    .stApp [data-testid="stSidebar"] span,
    .stApp [data-testid="stSidebar"] label,
    .stApp [data-testid="stSidebar"] strong,
    .stApp [data-testid="stSidebar"] b,
    .stApp [data-testid="stSidebar"] .stMarkdown,
    .stApp [data-testid="stSidebar"] .stMarkdown p,
    .stApp [data-testid="stSidebar"] .stMarkdown li,
    .stApp [data-testid="stSidebar"] .stMarkdown span,
    .stApp [data-testid="stSidebar"] .stMarkdown strong,
    .stApp [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    .stApp [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li,
    .stApp [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
    .stApp [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong {
        color: #F1F5F9 !important;
    }

    /* Headers con peso visual */
    h1 {
        color: #0F172A !important;
        font-weight: 800 !important;
        letter-spacing: -0.025em;
        margin-bottom: 0.5rem !important;
    }
    h2 {
        color: #0F172A !important;
        font-weight: 700 !important;
        border-bottom: 3px solid #0098CD;
        padding-bottom: 0.4rem;
        margin-top: 1.8rem !important;
    }
    h3 {
        color: #334155 !important;
        font-weight: 600 !important;
        margin-top: 1.2rem !important;
    }

    /* Tarjetas de metricas con sombra */
    [data-testid="stMetric"] {
        background: #FFFFFF;
        padding: 1.1rem 1.3rem;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(15, 23, 42, 0.06), 0 1px 2px rgba(15, 23, 42, 0.04);
        border-left: 4px solid #0098CD;
        transition: transform 0.15s ease-out;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(15, 23, 42, 0.10), 0 2px 4px rgba(15, 23, 42, 0.06);
    }
    [data-testid="stMetricLabel"] {
        color: #64748B !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="stMetricValue"] {
        color: #0F172A !important;
        font-weight: 800 !important;
        font-size: 1.8rem !important;
    }

    /* Sidebar mas elegante */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div {
        color: #F1F5F9 !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: #334155 !important;
    }

    /* Botones primarios */
    .stButton button {
        background: linear-gradient(135deg, #0098CD 0%, #006B92 100%);
        color: white !important;
        border: none;
        font-weight: 600;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 152, 205, 0.20);
    }
    .stButton button:hover {
        box-shadow: 0 4px 8px rgba(0, 152, 205, 0.30);
        transform: translateY(-1px);
    }

    /* Info/success/warning con estilo cohesionado */
    [data-testid="stAlert"] {
        border-radius: 10px;
        border-left-width: 5px;
    }

    /* Tablas con look profesional */
    [data-testid="stTable"], [data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
    }

    /* Code blocks */
    pre, code {
        font-family: 'JetBrains Mono', 'DejaVu Sans Mono', monospace !important;
        border-radius: 8px !important;
    }

    /* Eliminar el padding superior excesivo */
    .block-container {
        padding-top: 2rem !important;
        max-width: 1400px;
    }
</style>
""", unsafe_allow_html=True)


# Sidebar
with st.sidebar:
    st.title("🤖 TFM Pose 6-DoF")
    st.markdown("**Transformer + Diffusion**\n\nPara *Bin Picking* Robótico")
    st.markdown("---")
    st.markdown("**Autores:**\n- Giocrisrai Godoy Bonillo\n- José Miguel Carrasco")
    st.markdown("**Director/a:** Iván Oristela Benítez González")
    st.markdown("**UNIR · Junio 2026**")
    st.markdown("---")
    section = st.radio(
        "Sección",
        ["📊 Resumen", "🔬 Exploraciones post-TFM",
         "🤖 Robustez y clutter", "💡 Innovación y SOTA",
         "🧠 Decisiones del pipeline", "🗣️ Lenguaje natural",
         "🎯 Hipótesis", "📈 Métricas FP",
         "🛡️ Robustez", "⚙️ Profiling", "🌳 Diversidad",
         "🎮 PBVS", "📦 Per-Object", "🎬 Video", "📚 Recursos"]
    )


# Helpers
@st.cache_data
def load_json(path: str):
    p = REPO / path
    if not p.exists():
        return None
    return json.loads(p.read_text())


def load_image(path: str):
    p = REPO / path
    if p.exists():
        return str(p)
    return None


def render_language_tab(instruction: str) -> dict:
    """Lógica del tab de lenguaje natural. Devuelve el payload de grounding.

    Separada de los widgets para ser testeable sin servidor Streamlit.
    """
    from src.language import make_parser
    from src.language.grounding import Grounder
    from src.language.schema import ObjectView
    objs = [
        ObjectView(0, (-0.20, 0.0, 0.5), {"color": "red", "shape": "cube", "size": "large"}),
        ObjectView(1, (0.00, 0.0, 0.5), {"color": "blue", "shape": "cube", "size": "small"}),
        ObjectView(2, (0.20, 0.0, 0.5), {"color": "red", "shape": "sphere", "size": "small"}),
    ]
    instr = make_parser("deterministic").parse(instruction)
    res = Grounder(method="attribute").ground(instr, objs)
    return {
        "parsed": {"color": instr.target.color, "shape": instr.target.shape,
                   "size": instr.target.size,
                   "spatial": instr.spatial.relation if instr.spatial else None},
        "grounding": {"target_obj_id": res.target_obj_id, "scores": res.scores,
                      "ambiguous": res.ambiguous, "method": res.method},
    }


# === SECCIONES ===

if section == "📊 Resumen":
    st.title("Resumen ejecutivo")
    st.markdown(
        "**¿Qué es esto?** Un sistema que permite a un brazo robótico ver un objeto, "
        "decidir cómo cogerlo y planificar el movimiento — todo en menos de 7 s y sin "
        "necesitar un ordenador con GPU dedicada de gama alta. Integra tres tecnologías "
        "(Transformer para visión 3D, Diffusion Policy para planificación, Visual Servoing "
        "para control fino) y valida tres hipótesis sobre datasets industriales reconocidos "
        "(YCB-Video, T-LESS)."
    )

    # Figura hero: arquitectura del pipeline
    hero_pipeline = REPO / "docs/figures_hero/01_pipeline_architecture.png"
    if hero_pipeline.exists():
        st.image(str(hero_pipeline), width="stretch")

    # Diagrama Graphviz: pipeline detallado con flechas
    diag_pipeline = REPO / "docs/figures_hero/10_pipeline_full.png"
    if diag_pipeline.exists():
        with st.expander("🔍 Diagrama detallado del pipeline (Graphviz)", expanded=False):
            st.image(str(diag_pipeline), width="stretch")

    col1, col2, col3 = st.columns(3)
    col1.metric("Páginas TFM", "63")
    col2.metric("Experimentos", "26", "+25 vs entrega 1")
    col3.metric("Tests pasando", "173", "✓ CI verde")

    col4, col5, col6 = st.columns(3)
    col4.metric("AUC ADD-S YCB-V", "0.908", "[CI 95% 0.901–0.916]")
    col5.metric("AUC ADD-S T-LESS", "0.957", "[CI 95% 0.954–0.959]")
    col6.metric("Cycle p95 E2E (ultra)", "6.29 s / 6.68 s", "<10 s ✓")

    col7, col8, col9 = st.columns(3)
    col7.metric("MSE Diffusion ultra", "0.0022", "-89 % vs original")
    col8.metric("Jerk RMS ultra", "0.053", "-93 % vs original")
    col9.metric("Coste hardware", "~2.000 USD", "vs 15-150k USD industrial")

    st.markdown("---")
    st.subheader("Diagrama de arquitectura")
    img = load_image("experiments/results/pipeline_e2e/fig_pipeline_arquitectura.png")
    if img:
        st.image(img, caption="Pipeline TFM: FoundationPose + Diffusion Policy + CoppeliaSim")

elif section == "🔬 Exploraciones post-TFM":
    st.title("13 contribuciones novedosas sobre el TFM entregado")

    # Figura hero: dashboard de exploraciones
    hero_explor = REPO / "docs/figures_hero/02_exploraciones_dashboard.png"
    if hero_explor.exists():
        st.image(str(hero_explor), width="stretch")

    # Diagrama Graphviz: dependencias entre exploraciones
    diag_workflow = REPO / "docs/figures_hero/12_exploraciones_workflow.png"
    if diag_workflow.exists():
        with st.expander("🔗 Dependencias entre exploraciones (DAG)", expanded=False):
            st.image(str(diag_workflow), width="stretch")
            st.caption("Cada flecha indica que el resultado de una exploración se reutiliza en la siguiente.")
    st.markdown("---")

    st.markdown(
        "Tras entregar el TFM se planificaron y ejecutaron 5 exploraciones con "
        "criterios numéricos de éxito. **Las 4 se mergearon a `main`** porque "
        "cumplen los criterios. Documentación completa: "
        "[`docs/PLAN_EXPLORACIONES_POST_TFM.md`](https://github.com/Giocrisrai/pose6dof-transformers-diffusion/blob/main/docs/PLAN_EXPLORACIONES_POST_TFM.md)."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tests totales", "173", "+50 vs TFM")
    col2.metric("Modelos Diffusion", "10", "+7 nuevos")
    col3.metric("Paquete PyPI publicado", "✓", "bop-bootstrap-ci 0.1.0")
    col4.metric("Exploraciones éxitos", "13/13", "100 %")

    st.markdown("---")

    # Exploración 1
    st.subheader("1️⃣  Bootstrap-CI BOP toolkit (PyPI)  ✅")
    st.markdown(
        "Paquete `bop-bootstrap-ci` 0.1.0 que extrae el framework de evaluación a "
        "librería standalone. **27 tests pasando, 97 % cobertura, twine check PASSED**."
    )
    st.code(
        "pip install bop-bootstrap-ci\n\n"
        "from bop_bootstrap_ci import bootstrap_auc_adds\n"
        "ci = bootstrap_auc_adds(add_s_errors_mm).as_dict()\n"
        '# {"point": 0.908, "lo": 0.901, "hi": 0.916, "B": 1000, "alpha": 0.05}',
        language="python",
    )

    # Exploración 2
    st.subheader("2️⃣  Distillation Diffusion 1-NFE  ✅")
    st.markdown(
        "El modelo `ultra` (DDIM-25, 93 ms) se destila a un student `ultra_fast` de "
        "**1 forward pass**. Verificado en vivo en la API REST."
    )
    st.table(pd.DataFrame({
        "Métrica": ["MSE vs GT heurístico", "Jerk RMS", "Latencia por trayectoria", "NFE inference"],
        "Teacher ultra (DDIM-25)": ["0.0129", "0.064", "48.5 ms", "25"],
        "Student ultra_fast": ["**0.0124**", "**0.018**", "**0.09 ms**", "**1**"],
        "Mejora": ["−4 %", "−71 %", "**×517 speedup**", "×25 menos"],
    }))
    st.info(
        "**Hallazgo metodológico**: el 'MSE 0.0022' reportado en el TFM original "
        "era *noise-prediction loss* durante el training, no MSE de trayectoria "
        "reconstruida (que para el teacher es 0.0129). Corregido honestamente."
    )

    # Exploración 3
    st.subheader("3️⃣  Pipeline 100 % open-license  ✅")
    st.markdown(
        "Comparativa cuantitativa con bootstrap CI 95 % usando el paquete de #1, "
        "sobre alternativas open a FoundationPose (que tiene licencia NC NVIDIA). "
        "Pipeline ahora es agnóstico de estimador via `PoseEstimator` protocol."
    )

    img_pareto = REPO / "experiments/results/exp15_open_license/fig_pareto.png"
    if img_pareto.exists():
        st.image(str(img_pareto), caption="Pareto licencia × performance — bootstrap CI 95 %",
                  width="stretch")

    st.markdown(
        "**Conclusión**: cambiar a **FreeZeV2 (Apache-2.0)** cuesta solo **−3 pp AUC** "
        "y abre la puerta a comercialización del pipeline."
    )

    # Exploración 4
    st.subheader("4️⃣  VLA-lite con CLIP  ✅")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown(
            "Añade **lenguaje natural** al pipeline. El usuario escribe "
            "*'pick the red object'* → CLIP encode → TextGroundedGate selecciona "
            "el objeto → Diffusion genera trayectoria.\n\n"
            "**Selection accuracy 98.6 %** sobre escenas multi-objeto sintéticas. "
            "Coste: **1000× menor que RDT-1B / π0**.\n\n"
            "Pruébalo en vivo en Gradio → http://127.0.0.1:7860 → tab "
            "**🗣️ Hablar al robot**."
        )
    with col_b:
        st.metric("Selection accuracy", "98.6 %", "+23.6 pp sobre objetivo")
        st.metric("Gate accuracy", "100 %", "perfecto")
        st.metric("Latencia total", "~50 ms", "CLIP+gate+DDIM")

    # Exploración 5
    st.subheader("5️⃣  Robustez lingüística del VLA-lite  ✅")
    st.markdown(
        "Extensión cuantitativa del exp 4: evaluación del modelo CLIP+gate "
        "sobre **900 frases en 6 familias lingüísticas no vistas en training** "
        "(sinónimos, modificadores, frases largas, casos extremos como solo `{color}`)."
    )
    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.metric("Familias evaluadas", "6/6 ≥ 75 % acc", "100 % éxito")
    col_r2.metric("Accuracy media", "100.0 %", "900/900 frases")
    col_r3.metric("Min confidence", "99.96 %", "todas las familias")
    st.info(
        "**Implicación**: CLIP aporta comprensión real del lenguaje, no plantilla "
        "memorizada. El modelo generaliza incluso a frases tipo *'between the two, "
        "pick the red object'* o *'go red'* sin re-entrenar."
    )

    # Exploración 6
    st.subheader("6️⃣  VLA-lite multi-atributo color + forma  ✅")
    st.markdown(
        "Extiende el gate a 7-D (RGB + shape one-hot). Soporta "
        "*'pick the red sphere'*, *'grab the blue cube'*, *'select the green cylinder'*. "
        "Modelo: `diffusion_policy_clip_shapes.pth`."
    )
    col_6a, col_6b, col_6c = st.columns(3)
    col_6a.metric("Accuracy global", "99.9 %", "1199/1200")
    col_6b.metric("Color-only", "100 %", "n=389")
    col_6c.metric("Combinado color+forma", "100 %", "n=393")

    # Exploración 7
    st.subheader("7️⃣  Simulaciones visuales 3D  ✅")
    st.markdown(
        "12 renders 3D con objetos reales (cubos, esferas, cilindros, cajas) "
        "mostrando la decisión del pipeline. Incluye casos límite como "
        "*'the round one'* (entiende sinónimo de sphere)."
    )
    img_grid = REPO / "experiments/results/exp19_visual_sims/grid_overview.png"
    if img_grid.exists():
        st.image(str(img_grid), caption="12 escenas demostrativas — 12/12 correctas",
                  width="stretch")

    # Exploración 8
    st.subheader("8️⃣  VLA-lite multi-objeto N=2..5  ✅")
    st.markdown(
        "Generaliza el gate a **hasta 5 objetos simultáneos en la escena** "
        "(shared-weight scorer + mask + softmax). Aplicable directamente a "
        "logística (cintas con varios paquetes), reciclaje (bins con categorías), "
        "manufactura (bandejas)."
    )
    col_8a, col_8b, col_8c, col_8d = st.columns(4)
    col_8a.metric("N=2", "100 %", "n=394")
    col_8b.metric("N=3", "100 %", "n=365")
    col_8c.metric("N=4", "100 %", "n=399")
    col_8d.metric("N=5", "100 %", "n=342")

    # Exploración 9
    st.subheader("9️⃣  Atributo continuo TAMAÑO  ✅")
    st.markdown(
        "Añade dimensión continua **size** (small / medium / large = 30/50/80 mm). "
        "Soporta *'pick the large red box'*, *'the small one'*, *'take the medium object'*."
    )
    col_9a, col_9b = st.columns(2)
    col_9a.metric("Accuracy global", "99.9 %", "n=1500")
    col_9b.metric("8 templates ≥ 99.5 %", "100 %", "todos pasan")

    # Exploración 10
    st.subheader("🔟  Instrucciones secuenciales multi-step  ✅")
    st.markdown(
        "Parser de connectors (*then*, *after*, *followed by*, *in order:*) + "
        "ejecutor que reutiliza el modelo del exp 8 **sin re-train**. "
        "Soporta *'first the red cube, then the blue sphere'* hasta 4 pasos."
    )
    col_10a, col_10b = st.columns(2)
    col_10a.metric("Secuencias completas", "8/8 = 100 %", "")
    col_10b.metric("Pasos individuales", "20/20 = 100 %", "")

    # Exploración 11 — VISUAL GROUNDING (la más importante)
    st.subheader("1️⃣1️⃣  CLIP-image visual grounding  ✅ **(crítico para producción)**")
    st.markdown(
        "**Cierre del pipeline industrial real**: el sistema selecciona el objeto "
        "basándose en la **apariencia visual** (crop CLIP-image 768-D), no en "
        "atributos sintéticos declarados. Esto es lo que permite enchufar una "
        "cámara RGB-D real al sistema."
    )
    col_11a, col_11b = st.columns(2)
    col_11a.metric("Selection accuracy", "100 %", "n=500 val")
    col_11b.metric("Latencia training", "1.4 min M1 Pro", "")
    st.code("""
    Camera RGB-D → segmentación (SAM2) → crops 64×64 RGB
                                                ↓
                          CLIP-image (frozen, 86M) → embedding 768-D
                                                ↓
                          CLIP-text(instrucción) + VisualGate
                                                ↓
                          Diffusion Policy → trayectoria
    """, language="text")

    # Exploración 12
    st.subheader("1️⃣2️⃣  Robustez CLIP-image con domain randomization  ✅")
    st.markdown(
        "Valida que el modelo del exp 11 sobrevive a condiciones reales: "
        "oclusión, ruido sensor e iluminación variable. **Bootstrap CI 95 %** "
        "usando el paquete PyPI publicado `bop-bootstrap-ci`."
    )
    img_robust = REPO / "experiments/results/exp25_robustness/fig_robustness_curves.png"
    if img_robust.exists():
        st.image(str(img_robust), caption="Curvas de robustez con CI 95 % B=1000",
                  width="stretch")
    col_12a, col_12b, col_12c = st.columns(3)
    col_12a.metric("Condiciones robustas", "12/12 ≥ 75 %", "")
    col_12b.metric("Accuracy media", "95.3 %", "")
    col_12c.metric("Combinación realista", "93.7 %", "(occ 20 % + ruido + atenuación)")

    # Exploración 13
    st.subheader("1️⃣3️⃣  Razonamiento espacial  ✅")
    st.markdown(
        "Extiende el gate con **coordenadas XYZ normalizadas** para que el modelo "
        "razone sobre referencias espaciales: *'the leftmost red object'*, "
        "*'the closest one'*, *'the highest object'*, *'pick the one closest to me'*."
    )
    col_13a, col_13b = st.columns(2)
    col_13a.metric("Accuracy global", "98.4 %", "n=1500")
    col_13b.metric("Templates ≥ 94.9 %", "13/13", "todos pasan")
    st.info(
        "5 templates en **100 %**: *the lowest one*, *the closest one*, "
        "*the farthest object*, *select the leftmost one*, *pick the one closest to me*."
    )

    st.markdown("---")
    st.success(
        "**Resumen completo**: 173 tests pasando · 10 modelos Diffusion · "
        "**bop-bootstrap-ci 0.1.0 publicado en PyPI** · 4 hallazgos metodológicos · "
        "13 documentos de cierre · 22 simulaciones visuales 3D. "
        "Etapas físicas post-TFM detalladas en "
        "[ROADMAP_POSTTFM.md](https://github.com/Giocrisrai/pose6dof-transformers-diffusion/blob/main/docs/ROADMAP_POSTTFM.md)."
    )

elif section == "🤖 Robustez y clutter":
    st.title("Robustez, selección en clutter y verificación previa")
    st.markdown(
        "Tres iteraciones post-TFM sobre el pipeline de bin-picking, todas con la "
        "misma palanca barata: **randomizar en las capturas lo que NO debe importar** "
        "(apariencia, clutter) compra invariancia visual por minutos de cómputo. La "
        "narrativa industrial: *agarrar lo que se pida (Iter 8), entre lo que haya "
        "(Iter 9), sin golpear el resto (verificar antes de actuar)*. Todas las tablas "
        "se calculan en vivo desde los JSON de evaluación pareada (n=25, seed 2026). "
        "Detalle: [`docs/INTEGRATION_PIPELINE.md`](https://github.com/Giocrisrai/pose6dof-transformers-diffusion/blob/main/docs/INTEGRATION_PIPELINE.md)."
    )

    d8 = load_json("experiments/results/pick_with_diffusion/eval_v8_robustez.json")
    d9 = load_json("experiments/results/pick_with_diffusion/eval_v9_clutter.json")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precisión de agarre", "1.4 cm", "−2.2 cm vs original (Iter 8)")
    c2.metric("Invariante a forma/color", "✓", "esfera/cilindro/cubo")
    c3.metric("Elige la pieza indicada", "100 %", "entre 3 piezas (Iter 9)")
    c4.metric("Golpes a otras piezas", "0 %", "con verificación (v8)")
    st.markdown("---")

    # ── Iter 8: robustez a forma y color ──────────────────────────
    st.subheader("1️⃣  Robustez a forma y color (Iter 8)  ✅")
    st.markdown(
        "La política original se entrenó solo con cubos rojos y se degrada fuera de "
        "distribución. Reentrenar con apariencias randomizadas (3 formas × 6 colores, "
        "~5 min de datos + 46 s de fine-tune) **triplica la precisión de agarre y la "
        "hace invariante**, sin sacrificar el caso original."
    )
    if d8:
        def _r8(cond):
            r = d8["condiciones"][cond]["resumen"]
            return [f"{r['pick_place_pct']:.0f} %", f"{r['grasp_pct']:.0f} %",
                    f"{r['deposit_pct']:.0f} %", f"{r['mean_grasp_proximity_m']*100:.1f} cm"]
        st.table(pd.DataFrame({
            "Política × condición": ["original × cubo rojo", "original × randomizada",
                                     "robusta × cubo rojo", "robusta × randomizada"],
            "Pick&Place E2E": [_r8("v7a_phase2__cubo_rojo")[0], _r8("v7a_phase2__randomizada")[0],
                               _r8("v8_randomized__cubo_rojo")[0], _r8("v8_randomized__randomizada")[0]],
            "Agarre": [_r8("v7a_phase2__cubo_rojo")[1], _r8("v7a_phase2__randomizada")[1],
                       _r8("v8_randomized__cubo_rojo")[1], _r8("v8_randomized__randomizada")[1]],
            "Depósito": [_r8("v7a_phase2__cubo_rojo")[2], _r8("v7a_phase2__randomizada")[2],
                         _r8("v8_randomized__cubo_rojo")[2], _r8("v8_randomized__randomizada")[2]],
            "Proximidad": [_r8("v7a_phase2__cubo_rojo")[3], _r8("v7a_phase2__randomizada")[3],
                           _r8("v8_randomized__cubo_rojo")[3], _r8("v8_randomized__randomizada")[3]],
        }))
        st.caption("El depósito bajo en randomizada (esferas/cilindros) es física: ruedan al soltarse, no es la percepción.")
    else:
        st.warning("No se encontró `eval_v8_robustez.json`.")

    # ── Iter 9: selección en clutter ──────────────────────────────
    st.subheader("2️⃣  Selección de la pieza indicada entre varias (Iter 9)  ✅")
    st.markdown(
        "Una mesa real tiene varias piezas. Añadir 0-2 distractoras de apariencia "
        "aleatoria a las capturas enseña al sistema a **ir por la pieza indicada (su "
        "pose) ignorando el resto**. La política sin este entrenamiento se degrada "
        "con distractores; la de clutter recupera el 100 % de agarre."
    )
    if d9:
        def _r9(cond):
            r = d9["condiciones"][cond]["resumen"]
            return [f"{r['grasp_pct']:.0f} %", f"{r.get('mean_grasp_proximity_m', 0)*100:.1f} cm"]
        st.table(pd.DataFrame({
            "Política × condición": ["robusta (Iter 8) × con distractores",
                                     "clutter (Iter 9) × sin distractores",
                                     "clutter (Iter 9) × con distractores"],
            "Agarre": [_r9("v8_randomized__con_distractores")[0],
                       _r9("v9_clutter__sin_distractores")[0],
                       _r9("v9_clutter__con_distractores")[0]],
            "Proximidad": [_r9("v8_randomized__con_distractores")[1],
                           _r9("v9_clutter__sin_distractores")[1],
                           _r9("v9_clutter__con_distractores")[1]],
        }))
    else:
        st.warning("No se encontró `eval_v9_clutter.json`.")

    # ── Verify-then-act ───────────────────────────────────────────
    st.subheader("3️⃣  Verificar antes de actuar — no golpear las otras piezas  ✅")
    st.markdown(
        "Al ejecutar, el brazo podría rozar otras piezas (como en una celda real). La "
        "difusión genera las 8 trayectorias como números dentro de la red **sin mover "
        "el brazo**; la verificación descarta las que pasarían a <7 cm de otra pieza y "
        "ejecuta la mejor de las seguras. Métrica sobre distractores **cubo** "
        "(desplazamiento = golpe real; las piezas redondas ruedan y dan outliers)."
    )
    if d9:
        def _cubos(cond):
            rs = d9["condiciones"].get(cond, {}).get("resultados", [])
            desp = [dd for r in rs if "error" not in r
                    for di, dd in zip(r.get("distractores", []),
                                      r.get("desplazamiento_distractores_m", []))
                    if di["forma"] == "cubo"]
            if not desp:
                return "—", "—"
            a = np.array(desp)
            return f"{100*float((a > 0.02).mean()):.0f} %", f"{100*float(np.percentile(a, 90)):.1f} cm"
        st.table(pd.DataFrame({
            "Política × condición": ["robusta × con distractores", "robusta × con verificación",
                                     "clutter × con distractores", "clutter × con verificación"],
            "Cubos golpeados >2cm": [_cubos("v8_randomized__con_distractores")[0],
                                     _cubos("v8_randomized__con_distractores_seguro")[0],
                                     _cubos("v9_clutter__con_distractores")[0],
                                     _cubos("v9_clutter__con_distractores_seguro")[0]],
            "Peor caso (p90)": [_cubos("v8_randomized__con_distractores")[1],
                                _cubos("v8_randomized__con_distractores_seguro")[1],
                                _cubos("v9_clutter__con_distractores")[1],
                                _cubos("v9_clutter__con_distractores_seguro")[1]],
        }))
    st.info(
        "**Honestidad**: verify-then-act reduce los golpes (v8 robusta: 18 %→0 %, peor "
        "caso 13.6→0.2 cm) a un costo pequeño de precisión. NO cubre piezas redondas "
        "(ruedan al tocarse) ni golpes del cuerpo del brazo (los links, no solo la "
        "punta): la seguridad completa requiere planificación de colisiones de cuerpo "
        "entero (RRT/OMPL). Es la versión barata y honesta del 'mirar antes de actuar'."
    )
    st.caption(
        "Pruébalo en vivo: `streamlit`/Gradio demo → http://127.0.0.1:7860 → pestaña "
        "CoppeliaSim → selector de política + distractoras + verificación 🛡️."
    )

elif section == "💡 Innovación y SOTA":
    st.title("Innovación, valor diferencial y estado del arte (mayo 2026)")

    st.markdown("### En 60 segundos")
    st.markdown(
        "Este TFM **no inventa** FoundationPose ni Diffusion Policy — esas piezas ya "
        "existen (CVPR 2024 y RSS 2023). Lo que hace es **integrarlas en un pipeline "
        "E2E reproducible, estadísticamente riguroso y entrenable en un portátil**. "
        "A fecha de mayo 2026, **nadie ha publicado en abierto** esta combinación con "
        "este nivel de detalle estadístico."
    )

    st.markdown("### Mapa del estado del arte (mayo 2026)")

    col_p, col_d = st.columns(2)
    with col_p:
        st.markdown("#### Percepción 6-DoF post-FoundationPose")
        st.markdown(
            "| Método | Año | Aporte |\n"
            "|---|---|---|\n"
            "| FreeZeV2 | 2025 | Training-free, 8× speedup, +5 % accuracy |\n"
            "| Any6D | 2025 | Una sola imagen RGB-D, model-free |\n"
            "| SamPose | 2025 | Open-world, single-view prompt (sin CAD) |\n"
            "| NBV active perception | 2025 | 95 % success en escenarios ambiguos |\n"
            "| BOP Challenge 2025 | Nov 2025 | Resultados ICCV'25 Honolulu |\n"
        )
    with col_d:
        st.markdown("#### Planificación post-Diffusion Policy")
        st.markdown(
            "| Método | Año | Aporte |\n"
            "|---|---|---|\n"
            "| RDT-1B | Oct 2024 | 1,2 B params, bimanual, 46 datasets |\n"
            "| π0 / π0-FAST / π0.5 | 2024-2025 | VLA flow matching, control 50 Hz |\n"
            "| OpenVLA | 2024 | 7 B params, abre RT-2 paradigm |\n"
            "| Two-Steps Diffusion | Oct 2025 | Solo 2 NFE (vs 25-100) |\n"
            "| On-Device DiT Policy | ICCV 2025 | Latencia edge para tiempo real |\n"
        )

    st.markdown("---")
    st.markdown("### Posicionamiento")

    pos_col1, pos_col2 = st.columns([1, 1])
    with pos_col1:
        st.success(
            "**Lo que SÍ aporta valor diferencial:**\n\n"
            "1. Pipeline E2E **open-source MIT** que une FP + Diffusion + PBVS\n"
            "2. **Bootstrap CI 95 %** B=1000 en todas las métricas\n"
            "3. **Re-entrenamiento progresivo** demostrado: -89 % MSE, -93 % jerk\n"
            "4. **Robustez cuantificada** (oclusión hasta 70 %, ruido hasta σ=10 mm)\n"
            "5. **Validación E2E live** sobre 60+ instancias en CoppeliaSim\n"
            "6. **Entrenamiento en M1 Pro MPS en 3,3 min** — sin datacenter\n"
            "7. **Aplicaciones industriales mapeadas** por sector"
        )
    with pos_col2:
        st.warning(
            "**Lo que NO es novedoso (transparencia):**\n\n"
            "- FoundationPose es de NVIDIA Labs (CVPR 2024)\n"
            "- Diffusion Policy es de Columbia/Toyota (RSS 2023)\n"
            "- PBVS en SE(3) existe desde los 90\n"
            "- Combinar percepción + planificación + control lo hace cualquier pipeline\n\n"
            "**El aporte es la integración pulida + auditoría estadística + democratización**, "
            "no la novedad algorítmica."
        )

    st.markdown("### Limitaciones honestas")
    st.markdown(
        "- Solo simulación (CoppeliaSim). Sin robot físico. Sim-to-real no validado.\n"
        "- Requiere CAD model — no funciona con objetos novel sin modelo.\n"
        "- Single-arm, sin lenguaje natural (vs RDT-1B / π0).\n"
        "- Cycle 6-7 s — superable: métodos on-device 2025 logran 1-3 s.\n"
        "- FoundationPose tiene licencia NC (NVIDIA) — restricción comercial.\n"
        "- Subset BOP-19 evaluado, no challenge completo — no comparable con leaderboard oficial."
    )

    st.markdown("### Comparativa de posicionamiento")
    st.table(pd.DataFrame({
        "Aspecto": ["Hardware", "Código", "Rigor estadístico", "Latencia",
                    "Lenguaje natural", "Coste estación industrial"],
        "Este TFM": ["~2 000 USD (M1 Pro)", "Open MIT", "Bootstrap CI 95 %",
                     "6-7 s", "No", "< 5 000 USD"],
        "SOTA puntera (RDT-1B, π0)": ["20-200 k USD cluster", "Open mayoritario",
                                       "Habitualmente puntos", "1-3 s", "Sí (VLA)", "N/A"],
        "Industriales (Isaac, Symbotic)": ["15-150 k USD", "Cerrado", "N/A publicado",
                                            "< 1 s", "A veces", "15-150 k USD"],
    }))

    st.markdown("---")
    st.markdown(
        "**El nicho**: democratizar bin picking robótico para PYMES y educación, "
        "manteniendo rigor académico.\n\n"
        "**Documento completo**: "
        "[`docs/INNOVACION_Y_ESTADO_DEL_ARTE.md`](https://github.com/Giocrisrai/pose6dof-transformers-diffusion/blob/main/docs/INNOVACION_Y_ESTADO_DEL_ARTE.md) "
        "(incluye 11 referencias clave verificadas en mayo 2026)."
    )

elif section == "🧠 Decisiones del pipeline":
    st.title("Cómo decide el sistema, paso a paso")
    st.markdown(
        "Estas tarjetas muestran qué hace el pipeline para cada objeto detectado: "
        "(1) lo identifica, (2) mide su pose 6-DoF, (3) genera múltiples trayectorias "
        "de agarre y (4) las ejecuta con el robot — todo dentro del presupuesto de 10 s."
    )

    cards_dir = REPO / "experiments/results/pipeline_e2e/decision_cards"
    cards = sorted(cards_dir.glob("decision_*.png")) if cards_dir.exists() else []

    if not cards:
        st.warning(
            "No hay tarjetas generadas todavía. Ejecuta:\n\n"
            "`.venv/bin/python experiments/make_decision_visualization.py`"
        )
    else:
        ds_filter = st.radio("Filtrar dataset", ["Todos", "YCB-V", "T-LESS"], horizontal=True)
        filtered = [c for c in cards if
                    (ds_filter == "Todos" or
                     (ds_filter == "YCB-V" and "ycbv" in c.name) or
                     (ds_filter == "T-LESS" and "tless" in c.name))]
        for card in filtered:
            obj_id = card.stem.split("_obj")[-1]
            dataset = "YCB-V" if "ycbv" in card.name else "T-LESS"
            st.subheader(f"{dataset} · objeto #{int(obj_id)}")
            st.image(str(card), width="stretch")
            st.markdown("---")

    st.info(
        "**Cómo leerlo**: cada tarjeta tiene 4 paneles. Arriba-izq: qué objeto se detectó "
        "(nombre, dataset, escena origen). Arriba-der: la pose 6-DoF predicha mostrada como "
        "ejes XYZ en 3D. Abajo-izq: las 10 trayectorias que Diffusion Policy propone para "
        "el agarre. Abajo-der: línea de tiempo con la latencia real de cada fase y el margen "
        "que queda contra el límite de 10 s (H3)."
    )

elif section == "🗣️ Lenguaje natural":
    st.title("🗣️ Bin picking guiado por lenguaje natural")
    st.markdown(
        "Escribe una instrucción y observa el *parsing* y la selección de objetivo "
        "(grounding). Consolida las exploraciones VLA/CLIP (exp16–26). Escena demo: "
        "3 objetos — cubo rojo (izq.), cubo azul (centro), esfera roja (der.)."
    )
    instr_text = st.text_input("Instrucción", value="dame el cubo rojo de la izquierda")
    if st.button("Interpretar y seleccionar"):
        payload = render_language_tab(instr_text)
        st.json(payload["parsed"])
        tgt = payload["grounding"]["target_obj_id"]
        if tgt is None:
            st.warning("Ningún objeto coincide con la instrucción.")
        elif payload["grounding"]["ambiguous"]:
            st.info(f"Ambiguo — sugerido objeto #{tgt}. Añade una relación espacial.")
        else:
            st.success(f"Objeto seleccionado: #{tgt}  (método: {payload['grounding']['method']})")
        if tgt is not None:
            st.bar_chart(payload["grounding"]["scores"])

elif section == "🎯 Hipótesis":
    st.title("Hipótesis verificables")

    st.markdown("### H1 — Precisión de pose")
    st.success("✅ ACEPTADA con bootstrap CI 95% (B=1000)")
    st.markdown("""
    - **Criterio**: Δ Mean AR ≥ 3 pp vs GDR-Net++ y Recall@10mm > 95%
    - **Resultado**: AUC ADD-S 0.908/0.957, Recall 95.8%/99.7%, Δ +3.0/+3.6 pp
    - **Test estadístico**: Wilcoxon p<0.0001, Cohen's d = -1.89 (efecto grande)
    """)

    st.markdown("### H2 — Planificación multimodal")
    st.success("✅ ACEPTADA")
    st.markdown("""
    - **Criterio**: score ≥ 0.95 y muestreo < 50 ms
    - **Resultado**: score medio 0.96, sampling 1.88 ms (DDPM) / 133 ms (DDIM-25 MPS)
    - **Modelo entrenado local**: M1 Pro/MPS, 30 epochs, MSE 0.020
    - **Diversidad emergente**: 2 modos detectados (silhouette 0.476), endpoint dispersión 58.6 cm
    """)

    st.markdown("### H3 — Viabilidad industrial sin GPU dedicada")
    st.success("✅ ACEPTADA con margen ≥ 3.14 s")
    st.markdown("""
    - **Criterio**: Cycle p95 < 10 s/instancia
    - **Resultado E2E live**: 6.12 s YCB-V (margen 3.88 s) / 6.86 s T-LESS (margen 3.14 s)
    - **Hardware**: Apple M1 Pro + Colab T4 (~$1.9k vs ~$15-150k industrial)
    """)

elif section == "📈 Métricas FP":
    st.title("Métricas FoundationPose con Bootstrap CI 95%")
    data = load_json("experiments/results/local_metrics_with_bootstrap.json")
    if data:
        for ds, d in data.get("datasets", {}).items():
            st.subheader(ds.upper())
            cols = st.columns(4)
            cols[0].metric("n", d["n_evaluated"])
            cols[1].metric("AUC ADD-S",
                f"{d['auc_adds_50mm']:.3f}",
                f"[{d['auc_adds_50mm_ci95']['lo']:.3f}, {d['auc_adds_50mm_ci95']['hi']:.3f}]")
            cols[2].metric("Recall@10mm",
                f"{d['recall_adds_10mm']*100:.1f}%",
                f"[{d['recall_adds_10mm_ci95']['lo']*100:.1f}%, {d['recall_adds_10mm_ci95']['hi']*100:.1f}%]")
            cols[3].metric("Median ADD-S", f"{d['adds_median_mm']:.2f} mm")
    else:
        st.warning("Ejecuta `python experiments/recompute_metrics_with_bootstrap.py`")

elif section == "🛡️ Robustez":
    st.title("Robustez ante oclusión y ruido")
    col1, col2 = st.columns(2)
    img1 = load_image("experiments/results/exp6_robustness/fig_robustness_occlusion.png")
    img2 = load_image("experiments/results/exp6_robustness/fig_robustness_noise.png")
    if img1: col1.image(img1, caption="Oclusión simulada {0, 30, 50, 70} %")
    if img2: col2.image(img2, caption="Ruido sensor sigma {0, 2, 5, 10} mm")

    data = load_json("experiments/results/exp6_robustness/exp6_results.json")
    if data:
        st.subheader("Tabla de degradación")
        for ds, r in data.get("datasets", {}).items():
            st.markdown(f"#### {ds.upper()}")
            df_occ = pd.DataFrame(r["occlusion"])
            st.dataframe(df_occ.style.format({"auc_adds_50mm": "{:.4f}", "recall_adds_10mm": "{:.1%}"}))

elif section == "⚙️ Profiling":
    st.title("Profiling del pipeline — Cuello de botella")
    img = load_image("experiments/results/exp10_profiling/fig_profiling.png")
    if img: st.image(img, caption="Distribución de tiempo de ciclo + latencia DDIM por n_steps")
    data = load_json("experiments/results/exp10_profiling/exp10_results.json")
    if data:
        b = data.get("bottleneck_analysis", {})
        col1, col2, col3 = st.columns(3)
        col1.metric("FoundationPose", f"{b.get('FP_fraction_pct', 0):.1f}%", f"{b.get('FoundationPose_ms', 0):.0f} ms")
        col2.metric("Diffusion DDIM-25", f"{b.get('Diff_fraction_pct', 0):.1f}%", f"{b.get('Diffusion_DDIM25_ms', 0):.0f} ms")
        col3.metric("CoppeliaSim", f"{b.get('Sim_fraction_pct', 0):.1f}%", f"{b.get('CoppeliaSim_50steps_ms', 0):.0f} ms")
        st.info(f"📊 **Conclusión**: FoundationPose es el cuello de botella dominante "
                f"({b.get('FP_fraction_pct', 0):.0f}% del ciclo). Optimización debe priorizar FP.")

elif section == "🌳 Diversidad":
    st.title("Diversidad multimodal Diffusion Policy")
    img1 = load_image("experiments/results/exp8_diversity/fig_diversity_pca.png")
    img2 = load_image("experiments/results/exp9_3d_viz/fig_trajectories_3d.png")
    if img1: st.image(img1, caption="PCA 2D + comparación jerk Diffusion vs heurístico")
    if img2: st.image(img2, caption="50 trayectorias 3D — multimodal")

    data = load_json("experiments/results/exp8_diversity/exp8_results.json")
    if data:
        col1, col2, col3 = st.columns(3)
        col1.metric("# muestras", data.get("n_samples_per_scene", 0) * data.get("n_scenes", 0))
        col2.metric("Modos detectados", data.get("best_n_modes_silhouette", "?"))
        col3.metric("Jerk Diffusion", f"{data.get('diffusion_jerk_rms', {}).get('mean', 0):.3f}")

elif section == "🎮 PBVS":
    st.title("Visual Servoing PBVS — Convergencia")
    img = load_image("experiments/results/exp7_pbvs/fig_pbvs_convergence.png")
    if img: st.image(img)
    data = load_json("experiments/results/exp7_pbvs/exp7_results.json")
    if data:
        s = data.get("summary", {})
        col1, col2, col3 = st.columns(3)
        col1.metric("Tasa convergencia", f"{s.get('convergence_rate_pct', 0):.0f}%")
        col2.metric("Iter mediana", f"{s.get('median_iters', 0):.0f}")
        col3.metric("Tiempo p95", f"{s.get('mean_time_s_p95', 0):.2f} s")

elif section == "📦 Per-Object":
    st.title("Análisis de error por categoría de objeto")
    col1, col2 = st.columns(2)
    img1 = load_image("experiments/results/exp12_per_object/fig_per_object_ycbv.png")
    img2 = load_image("experiments/results/exp12_per_object/fig_per_object_tless.png")
    if img1: col1.image(img1)
    if img2: col2.image(img2)
    data = load_json("experiments/results/exp12_per_object/exp12_results.json")
    if data:
        for ds, r in data.get("datasets", {}).items():
            st.subheader(f"{ds.upper()} - {r['n_objects']} objetos")
            cols = st.columns(2)
            cols[0].markdown("**Peores 3 objetos** (failure cases)")
            for o in r["worst_3"]:
                cols[0].markdown(f"- obj_id={o['obj_id']}: AUC {o['auc_adds_50mm']:.3f}, R@10mm {o['recall_10mm']:.1%}")
            cols[1].markdown("**Mejores 3 objetos**")
            for o in r["best_3"]:
                cols[1].markdown(f"- obj_id={o['obj_id']}: AUC {o['auc_adds_50mm']:.3f}, R@10mm {o['recall_10mm']:.1%}")

elif section == "🎬 Video":
    st.title("Demo del pipeline E2E")

    st.markdown(
        "Este video muestra el **ciclo completo de bin picking** en simulacion "
        "CoppeliaSim sobre el robot Ragnar (cinematica delta) con piezas del "
        "dataset YCB-Video. La camara cenital reproduce una linea de produccion "
        "tipo conveyor."
    )

    st.markdown("### Lo que vas a ver en el video")
    colA, colB, colC = st.columns(3)
    with colA:
        st.markdown(
            "**1. Percepcion 6-DoF**\n\n"
            "FoundationPose recibe RGB-D + CAD y devuelve la pose `(R, t)` en "
            "**~ 3.8 s**. El HUD lateral marca *FASE ACTUAL: PERCEPCION* y la "
            "latencia acumulada."
        )
    with colB:
        st.markdown(
            "**2. Planificacion Diffusion**\n\n"
            "Diffusion Policy genera **10 trayectorias multimodales** "
            "condicionadas a la pose en **~ 90 ms (DDIM-25)**. El HUD pasa "
            "a *PLANIFICACION DIFFUSION*."
        )
    with colC:
        st.markdown(
            "**3. Ejecucion + control**\n\n"
            "Una trayectoria se ejecuta en CoppeliaSim (**~ 1.2 s**) cerrando "
            "el lazo con PBVS en SE(3). El HUD muestra TOTAL ciclo y el "
            "veredicto **H3 PASA** si el total < 10 s."
        )

    st.markdown("### Que demuestra (vinculo con las 3 hipotesis del TFM)")
    st.markdown(
        "- **H1** *(precision de pose)*: el cubo virtual de FoundationPose se "
        "superpone correctamente con la pieza real -> AUC ADD-S = 0.908 YCB-V "
        "/ 0.957 T-LESS.\n"
        "- **H2** *(planificacion multimodal)*: cada ejecucion genera multiples "
        "trayectorias alternativas, no una sola -> diversidad emergente sin "
        "modos colapsados.\n"
        "- **H3** *(viabilidad sin GPU dedicada)*: el HUD lateral muestra TOTAL "
        "ciclo < 10 s ejecutado en MacBook Pro M1 Pro + Colab T4."
    )

    st.markdown("---")
    st.markdown("### Reproduccion completa (pantalla completa recomendado)")
    video_path = REPO / "experiments/results/pipeline_e2e/demo_v2.mp4"
    if video_path.exists():
        st.video(str(video_path))
        st.caption(
            "Demo v2 — Robot Ragnar (delta) en CoppeliaSim Edu V4.10 sobre "
            "escena `pickAndPlaceDemo`. Pulsa el icono de pantalla completa "
            "para ver todos los detalles del HUD."
        )

    st.markdown("---")
    st.subheader("Resumen visual en una sola imagen")
    img = load_image("experiments/results/pipeline_e2e/highlights_v2/composite_v2_3phases.png")
    if img:
        st.image(img, caption=(
            "Composite v2: las 3 fases (percepcion -> planificacion -> "
            "ejecucion) en un solo cuadro. Util para slides de defensa."
        ), width="stretch")

elif section == "📚 Recursos":
    st.title("Recursos y entregables")
    st.markdown("""
    ### Entregables `docs/entrega2/`
    - 📄 [TFM_Entrega2_UNIR.docx](docs/entrega2/TFM_Entrega2_UNIR.docx)
    - 📄 [TFM_Entrega2_UNIR.pdf](docs/entrega2/TFM_Entrega2_UNIR.pdf) (62 págs)
    - 📄 TFM_Entrega2_UNIR.md
    - 🎬 TFM_Defensa_Slides.pptx (20 slides)
    - 📋 FAQ_DEFENSA.md (30+ preguntas anticipadas)

    ### Repositorio público
    https://github.com/Giocrisrai/pose6dof-transformers-diffusion

    ### CLI ejecutable
    ```bash
    python scripts/run_experiment.py --list
    python scripts/run_experiment.py bootstrap
    python scripts/run_experiment.py all
    ```

    ### Reproducibilidad
    - 173 tests pasando (`pytest tests/ packages/bop_bootstrap_ci/tests/`)
    - 12 experimentos commiteados con resultados
    - RUN_CARD trazable hasta commit
    - Bootstrap CI 95% B=1000, semilla 42
    """)
