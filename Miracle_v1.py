"""
miRNA Upregulation Predictor — Random Forest
=============================================
Model : random_forest_microrna_model.pkl  (bundle from train_rf_csv.py)
Run   : streamlit run app_rf.py

Features expected by the model (in order):
  CAT: microrna_group_simplified, parasite, organism, cell type, scenario
  NUM: time
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="miRNA Upregulation Predictor (RF)",
    page_icon="🧬",
    layout="wide"
)

# ── Load model bundle ─────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('rf_mirna_model_v1.pkl')

try:
    bundle       = load_model()
    model        = bundle['model']
    oob_score    = bundle['oob_score']
    options      = bundle['options']
    mirna_lookup = bundle['mirna_lookup']
    fi           = bundle['feature_importance']
    all_groups   = sorted(set(mirna_lookup.values()))
except FileNotFoundError:
    st.error(
        "**Missing file:** `random_forest_microrna_model.pkl` not found. "
        "Run `train_rf_csv.py` first and place the pkl here."
    )
    st.stop()


# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
st.title("🧬 miRNA Upregulation Predictor")
st.markdown(
    "Predicts whether a miRNA is **upregulated** or **downregulated** "
    "during *Leishmania* infection based on experimental conditions."
)
st.caption("Model: Random Forest · OneHotEncoder · OOB evaluation")
st.divider()


# ══════════════════════════════════════════════════════════════
# LAYOUT — two columns
# ══════════════════════════════════════════════════════════════
col_input, col_result = st.columns([1, 1], gap="large")


# ── LEFT: Inputs ──────────────────────────────────────────────
with col_input:
    st.subheader("Experimental conditions")

    mirna_input = st.text_input(
        "miRNA name",
        placeholder="e.g. hsa-miR-155, mmu-let-7f"
    )

    parasite = st.selectbox(
        "Parasite species",
        options=options['parasite']
    )

    organism = st.selectbox(
        "Host organism",
        options=options['organism']
    )

    cell_type = st.selectbox(
        "Cell type",
        options=options['cell_type']
    )

    # Free numeric input — user can type any time point
    time = st.number_input(
        "Time point (hours post-infection)",
        min_value=0,
        max_value=10000,
        value=24,
        step=1
    )

    # Manual group override — if filled, takes priority over the automatic lookup
    group_override = st.text_input(
        "miRNA group (optional — overrides automatic lookup)",
        placeholder="e.g. miR-146b, let-7a"
    )

    predict_btn = st.button("Predict", type="primary", use_container_width=True)


# ── RIGHT: Result ─────────────────────────────────────────────
with col_result:
    st.subheader("Prediction")

    if predict_btn:
        if not mirna_input.strip():
            st.info("Please enter a miRNA name.")
        else:
            mirna_clean = mirna_input.strip()

            # ── Step 1: resolve group ─────────────────────────
            # Priority: manual override → automatic lookup → ask user
            if group_override.strip():
                group = group_override.strip()
                st.info(f"Using manually entered group: **{group}**")
            else:
                group = mirna_lookup.get(mirna_clean)
                if group:
                    st.info(f"miRNA found · group: **{group}**")
                else:
                    st.info(
                        f"**{mirna_clean}** is not in the training data. "
                        "Please enter its miRNA group in the field above."
                    )
                    st.stop()

            # ── Step 2: build input row ───────────────────────
            # parasite and cell type passed exactly as selectbox returns them
            # — no lowercasing — so OneHotEncoder sees the same values as training
            scenario = f"{parasite}_{cell_type}"

            input_df = pd.DataFrame([{
                'microrna_group_simplified': group,
                'parasite':                 parasite,
                'organism':                 organism,
                'cell type':                cell_type,
                'scenario':                 scenario,
                'time':                     time,
            }])

            # ── Step 3: predict ───────────────────────────────
            try:
                proba     = model.predict_proba(input_df)[0]
                pred      = model.predict(input_df)[0]
                prob_up   = proba[1]
                prob_down = proba[0]

                st.divider()

                if pred == 1:
                    st.success("## ⬆ Upregulated")
                else:
                    st.error("## ⬇ Downregulated")

                st.markdown(f"**Confidence:** {max(prob_up, prob_down)*100:.1f}%")

                st.markdown("**Probability breakdown:**")
                prob_col1, prob_col2 = st.columns(2)
                prob_col1.metric("Upregulated",   f"{prob_up   * 100:.1f}%")
                prob_col2.metric("Downregulated", f"{prob_down * 100:.1f}%")

                st.progress(
                    float(prob_up),
                    text=f"↑ {prob_up*100:.1f}%  |  ↓ {prob_down*100:.1f}%"
                )

                with st.expander("Input summary"):
                    display_df = input_df.copy()
                    display_df.insert(0, 'miRNA', mirna_clean)
                    st.dataframe(display_df, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"**Prediction error:** {e}")

    else:
        st.markdown(
            "<div style='color: gray; margin-top: 2rem;'>"
            "Fill in the conditions on the left and click <b>Predict</b>."
            "</div>",
            unsafe_allow_html=True
        )


# ══════════════════════════════════════════════════════════════
# MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════
st.divider()
st.subheader("Model performance")
st.caption("Random Forest · 100 estimators · OOB evaluation")

st.metric("OOB Accuracy", f"{oob_score*100:.2f}%")

st.markdown("**Top 10 feature importances (MDI):**")
fi_df = pd.DataFrame(fi)
fi_df['Importance'] = fi_df['Importance'].round(4)
st.dataframe(
    fi_df.style.background_gradient(subset=['Importance'], cmap='Greens'),
    use_container_width=True,
    hide_index=True
)
