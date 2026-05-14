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

import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parent

st.set_page_config(
    page_title="TFM Pose 6-DoF Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS custom
st.markdown("""
<style>
    .metric-card {
        background-color: #f5f5dc;
        padding: 15px;
        border-radius: 10px;
        margin: 5px;
    }
    .h1-status {
        font-size: 18px;
        font-weight: bold;
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
        ["📊 Resumen", "🧠 Decisiones del pipeline", "🎯 Hipótesis", "📈 Métricas FP",
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


# === SECCIONES ===

if section == "📊 Resumen":
    st.title("Resumen ejecutivo")

    col1, col2, col3 = st.columns(3)
    st.markdown(
        "**¿Qué es esto?** Un sistema que permite a un brazo robótico ver un objeto, "
        "decidir cómo cogerlo y planificar el movimiento — todo en menos de 7 s y sin "
        "necesitar un ordenador con GPU dedicada de gama alta. Integra tres tecnologías "
        "(Transformer para visión 3D, Diffusion Policy para planificación, Visual Servoing "
        "para control fino) y valida tres hipótesis sobre datasets industriales reconocidos "
        "(YCB-Video, T-LESS)."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Páginas TFM", "63")
    col2.metric("Experimentos", "13", "+12 vs entrega 1")
    col3.metric("Tests pasando", "123", "✓")

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
            st.image(str(card), use_container_width=True)
            st.markdown("---")

    st.info(
        "**Cómo leerlo**: cada tarjeta tiene 4 paneles. Arriba-izq: qué objeto se detectó "
        "(nombre, dataset, escena origen). Arriba-der: la pose 6-DoF predicha mostrada como "
        "ejes XYZ en 3D. Abajo-izq: las 10 trayectorias que Diffusion Policy propone para "
        "el agarre. Abajo-der: línea de tiempo con la latencia real de cada fase y el margen "
        "que queda contra el límite de 10 s (H3)."
    )

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
    video_path = REPO / "experiments/results/pipeline_e2e/demo_v2.mp4"
    if video_path.exists():
        st.video(str(video_path))
    img = load_image("experiments/results/pipeline_e2e/highlights_v2/composite_v2_3phases.png")
    if img: st.image(img, caption="Composite v2: 3 fases del bin picking")

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
    - 110 tests pasando (`pytest tests/`)
    - 12 experimentos commiteados con resultados
    - RUN_CARD trazable hasta commit
    - Bootstrap CI 95% B=1000, semilla 42
    """)
