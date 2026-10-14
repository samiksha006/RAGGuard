import os, html

import requests

import pandas as pd

import streamlit as st

from pathlib import Path



# Resolve paths from the project root, not the terminal's working directory.

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ANALYSIS_DIR = PROJECT_ROOT / "artifacts" / "analysis"



st.set_page_config(page_title="RAGGuard", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

API_DEFAULT = os.getenv("RAGGUARD_API_URL", "http://localhost:8000").rstrip("/")

BASELINE = {"Accuracy": .7403704, "Precision": .6132959, "Recall": .6945917, "F1": .6514172, "PR-AUC": .7287233, "ROC-AUC": .7974223}

PROBE = {"Accuracy": .7015, "Precision": .5556, "Recall": .7264, "F1": .6296, "PR-AUC": .7046, "ROC-AUC": .7818}

HYBRID = {"Accuracy": .6693, "Precision": .5180, "Recall": .7614, "F1": .6166, "PR-AUC": .7177, "ROC-AUC": .7863}

BASELINE = {"Accuracy": .7403704, "Precision": .6132959, "Recall": .6945917, "F1": .6514172, "PR-AUC": .7287233, "ROC-AUC": .7974223}

PROBE = {"Accuracy": .7015, "Precision": .5556, "Recall": .7264, "F1": .6296, "PR-AUC": .7046, "ROC-AUC": .7818}

HYBRID = {"Accuracy": .6693, "Precision": .5180, "Recall": .7614, "F1": .6166, "PR-AUC": .7177, "ROC-AUC": .7863}



st.markdown("""

<style>

.stApp{background:radial-gradient(ellipse at 75% -10%,rgba(130,10,35,.2),transparent 42%),#050b14;color:#edf2fa}

[data-testid="stHeader"]{background:rgba(5,11,20,.85)}

[data-testid="stSidebar"]{background:linear-gradient(180deg,#07111e,#050a12);border-right:1px solid #172536}

.block-container{padding-top:1.2rem;padding-bottom:2rem;max-width:1800px}

h1,h2,h3{letter-spacing:-.025em} h1{font-weight:850!important}

.hero{border:1px solid #1a293b;border-radius:16px;padding:1.2rem 1.4rem;background:linear-gradient(110deg,#1c0a17,#06101d 55%,#07111d);margin-bottom:1rem}

.hero-title{font-size:2.5rem;line-height:1.1;font-weight:850;color:#f5f7fb}.hero-title span{color:#ff304d}

.subtle,.tiny{color:#9cacc0;font-size:.84rem}

.panel-title{font-size:1.25rem;font-weight:750;color:#f2f5fb;margin-bottom:.4rem}

.stat-card{border:1px solid #24364b;border-radius:12px;padding:.8rem;background:#07121f;min-height:78px}

.stat-label{color:#9bacc2;font-size:.78rem}.stat-value{font-size:1.45rem;font-weight:800;color:#f7f9fc}

.status-dot{display:inline-block;width:9px;height:9px;border-radius:50%;background:#12d99a;margin-right:8px;box-shadow:0 0 10px #12d99a}

.risk-gauge{width:172px;height:172px;border-radius:50%;background:conic-gradient(#ff304d var(--risk),#1a283a 0);display:flex;align-items:center;justify-content:center;margin:.5rem auto 1rem;box-shadow:0 0 24px #ff304d20}

.risk-inner{width:148px;height:148px;border-radius:50%;background:#07111e;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;border:1px solid #18273a}

.risk-number{font-size:2rem;font-weight:850;color:white}.risk-caption{font-size:.83rem;color:#aabbd0}

.prediction-box{background:#041f1fc0;border:1px solid #15514c;border-radius:13px;padding:1rem}

.good{color:#22d99a;font-weight:800;font-size:1.1rem}.high{color:#ff5368;font-weight:800;font-size:1.1rem}

.example-box{background:#3c0d1a70;border:1px solid #7c2034;border-radius:12px;padding:1rem}

.example-label{color:#ff6176;font-weight:700;font-size:.8rem}

div.stButton>button{border:1px solid #27394d;border-radius:10px;background:#0b1726;color:#edf2fa;min-height:2.5rem;font-weight:650}

div.stButton>button:hover{border-color:#ff304d;color:white;box-shadow:0 0 16px #ff304d30}

div.stButton>button[kind="primary"]{background:linear-gradient(100deg,#ff304d,#e40049);border:1px solid #ff4860;color:white;box-shadow:0 4px 20px #ff244a38}

div[data-testid="stTextArea"] textarea{background:#0c1726;color:#edf2fa;border:1px solid #25364b;border-radius:10px}

div[data-testid="stTextArea"] textarea:focus{border-color:#ff304d;box-shadow:0 0 0 1px #ff304d}

hr{border-color:#1b2a3c!important}

</style>""", unsafe_allow_html=True)



with st.sidebar:

    st.markdown("<div style='text-align:center;font-size:4rem;color:#ff304d'>⬡</div><div style='text-align:center;font-size:1.3rem;font-weight:800'>RAG<span style='color:#ff304d'>Guard</span></div><div style='text-align:center;color:#8fa1b7;font-size:.8rem;margin-bottom:1.5rem'>Research prototype · v1.0</div>", unsafe_allow_html=True)

    page = st.radio("Navigation", ["Overview", "Analyze Answer", "Model Performance", "Error Analysis", "Research Details", "About"], label_visibility="collapsed")

    st.markdown("---")

    api_url = st.text_input("API base URL", API_DEFAULT).strip().rstrip("/")

    if st.button("Check API connection", use_container_width=True):

        try:

            r = requests.get(f"{api_url}/health", timeout=5); r.raise_for_status()

            info = r.json()

            st.success("API online" if info.get("model_available") else "API online; model missing")

            st.json(info)

        except Exception as e: st.error(f"API unavailable: {e}")

    st.markdown("---")

    st.caption("Statistical risk estimate, not a factuality guarantee.")



a,b = st.columns([1.7,2.1], vertical_alignment="center")

with a:

    st.markdown("<div class='hero'><div class='hero-title'>RAG<span>Guard</span></div><div style='font-size:1.08rem'>Hallucination Detection in RAG Answers</div><div class='subtle'>Using Classical Probes on LLM Internals</div></div>", unsafe_allow_html=True)

with b:

    x,y,z=st.columns(3)

    x.markdown("<div class='stat-card'><div class='stat-label'><span class='status-dot'></span>API status</div><div class='stat-value' style='font-size:1.05rem'>Live check</div><div class='stat-label'>Use sidebar check</div></div>",unsafe_allow_html=True)

    y.markdown("<div class='stat-card'><div class='stat-label'>Train samples</div><div class='stat-value'>15.09K</div><div class='stat-label'>RAGTruth</div></div>",unsafe_allow_html=True)

    z.markdown("<div class='stat-card'><div class='stat-label'>Test samples</div><div class='stat-value'>2.70K</div><div class='stat-label'>RAGTruth</div></div>",unsafe_allow_html=True)



if page in ("Overview","Analyze Answer"):

    left,right=st.columns([1.16,.84],gap="large")

    with left:

        st.markdown("<div class='panel-title'>✦ Analyze a RAG Answer</div><div class='subtle'>Enter the question, retrieved context, and generated answer to estimate hallucination risk.</div>",unsafe_allow_html=True)

        for k,v in [("rg_question",""),("rg_context",""),("rg_answer","")]:

            if k not in st.session_state: st.session_state[k]=v

        e1,e2,e3=st.columns(3)

        with e1:

            if st.button("Factual example",use_container_width=True):

                st.session_state.rg_question="What is the boiling point of water at sea level?"

                st.session_state.rg_context="Pure water boils at approximately 100°C at standard atmospheric pressure (1 atmosphere)."

                st.session_state.rg_answer="Water boils at approximately 100°C at standard atmospheric pressure."

        with e2:

            if st.button("Unsupported example",use_container_width=True):

                st.session_state.rg_question="What is the boiling point of water at sea level?"

                st.session_state.rg_context="Pure water boils at approximately 100°C at standard atmospheric pressure (1 atmosphere)."

                st.session_state.rg_answer="Water boils at 180°C at standard atmospheric pressure and was discovered in 1924."

        with e3:

            if st.button("Clear",use_container_width=True):

                for k in ("rg_question","rg_context","rg_answer"): st.session_state[k]=""

        question=st.text_area("⌕  Question",key="rg_question",height=85,max_chars=5000,placeholder="What does the source say?")

        context=st.text_area("▤  Retrieved context / evidence",key="rg_context",height=140,max_chars=20000,placeholder="Paste the source passages retrieved by your RAG system.")

        answer=st.text_area("✎  Generated answer",key="rg_answer",height=110,max_chars=10000,placeholder="Paste the answer to evaluate.")

        analyze=st.button("⚡  Assess hallucination risk",type="primary",use_container_width=True)

    with right:

        st.markdown("<div class='panel-title'>▥ Result</div><div class='subtle'>Model prediction and risk assessment</div>",unsafe_allow_html=True)

        result=st.session_state.get("rg_result")

        if analyze:

            if not question.strip() or not context.strip() or not answer.strip(): st.error("Please provide all three fields.")

            else:

                try:

                    resp=requests.post(f"{api_url}/predict",json={"question":question,"context":context,"answer":answer},timeout=120)

                    resp.raise_for_status(); result=resp.json(); st.session_state.rg_result=result

                except requests.RequestException as e: st.error(f"Prediction failed: {e}"); result=None

        if result:

            risk=max(0,min(1,float(result["hallucination_risk"]))); pct=risk*100; safe=result["prediction"]=="lower-risk"

            st.markdown(f"<div class='risk-gauge' style='--risk:{pct:.2f}%'><div class='risk-inner'><div class='risk-number'>{pct:.1f}%</div><div class='risk-caption'>Hallucination<br>Risk</div></div></div>",unsafe_allow_html=True)

            st.markdown(f"<div class='prediction-box'><div class='stat-label'>Prediction</div><div class='{'good' if safe else 'high'}'>{'✓' if safe else '⚠'} {html.escape(result['prediction'].replace('-',' ').title())}</div><div class='tiny' style='margin-top:.6rem'>Statistical estimate, not proof of correctness.</div></div>",unsafe_allow_html=True)

            m1,m2=st.columns(2); m1.metric("Model latency",f"{result.get('latency_ms',0):.2f} ms"); m2.metric("Threshold",f"{float(result.get('threshold',.5)):.3f}")

            st.caption(f"Model: {result.get('model','Configured model')}"); st.caption(result.get("disclaimer","Verify important claims against evidence."))

        else:

            st.markdown("<div style='padding:2rem 1rem;text-align:center;border:1px dashed #29394d;border-radius:14px;color:#95a8bf'><div style='font-size:2.1rem;color:#ff304d'>◉</div><b>Awaiting analysis</b><br><span class='tiny'>Your model result appears here.</span></div>",unsafe_allow_html=True)

    st.markdown("---")

    c1,c2,c3=st.columns([1.05,1.1,1])

    with c1:

        st.markdown("<div class='panel-title'>⚗ Example Inputs</div><div class='subtle'>Load a sample above to test the workflow.</div><div class='example-box' style='margin-top:1rem'><div class='example-label'>RESEARCH NOTE</div>Compare supported and unsupported answers. One example does not establish model accuracy.</div>",unsafe_allow_html=True)

    with c2:

        st.markdown("<div class='panel-title'>▥ Model Performance</div><div class='subtle'>Recorded experiment test metrics</div>",unsafe_allow_html=True)

        df=pd.DataFrame([

            {"Model":"TF-IDF + Logistic Regression","Accuracy":BASELINE["Accuracy"],"F1":BASELINE["F1"],"PR-AUC":BASELINE["PR-AUC"]},

            {"Model":"DistilBERT hidden-state probe","Accuracy":PROBE["Accuracy"],"F1":PROBE["F1"],"PR-AUC":PROBE["PR-AUC"]},

            {"Model":"TF-IDF + probe hybrid","Accuracy":HYBRID["Accuracy"],"F1":HYBRID["F1"],"PR-AUC":HYBRID["PR-AUC"]},

        ])

        view=df.copy()

        for col in ("Accuracy","F1","PR-AUC"): view[col]=view[col].map(lambda v:f"{v:.2%}")

        st.dataframe(view,use_container_width=True,hide_index=True)

        st.caption("Exploratory official-test results; see report for limitations.")

    with c3:

        st.markdown("<div class='panel-title'>ⓘ About RAGGuard</div>",unsafe_allow_html=True)

        st.markdown("- RAGTruth dataset: 15,090 train / 2,700 test rows\n- Classical ML baseline and hidden-state probes\n- Interactive assessment through FastAPI\n- Research prototype, not a factuality oracle")



elif page=="Model Performance":

    st.title("Model Performance")

    rows=[]

    for name,m in [("TF-IDF + Logistic Regression",BASELINE),("DistilBERT hidden-state probe",PROBE),("TF-IDF + probe hybrid",HYBRID)]:

        rows.append({"Model":name,**m})

    df=pd.DataFrame(rows).set_index("Model")

    st.dataframe(df.style.format("{:.3f}"),use_container_width=True)

    st.bar_chart(df[["F1","PR-AUC","ROC-AUC"]])

    st.warning("Official-test metrics were explored during configuration work. Treat comparisons as exploratory; use an independent holdout for a pristine final evaluation.")

elif page == "Error Analysis":
    st.title("Error Analysis")
    st.caption(
        "Explore model errors, class distribution, and evaluation plots "
        "generated by the RAGGuard analysis pipeline."
    )

    analysis_dir = ANALYSIS_DIR
    st.caption(f"Analysis directory: `{analysis_dir}`")

    if not analysis_dir.is_dir():
        st.error(f"Analysis directory does not exist: {analysis_dir}")
    else:
        available_files = sorted(
            p.name for p in analysis_dir.iterdir() if p.is_file()
        )

        metrics_path = analysis_dir / "error_analysis_metrics.json"
        csv_path = analysis_dir / "misclassified_examples.csv"
        report_path = analysis_dir / "error_analysis_report.md"

        st.subheader("Evaluation Metrics")
        if metrics_path.is_file():
            import json
            try:
                with metrics_path.open("r", encoding="utf-8") as f:
                    metrics = json.load(f)
                if isinstance(metrics, dict):
                    st.json(metrics)
                else:
                    st.write(metrics)
            except (OSError, json.JSONDecodeError) as exc:
                st.error(f"Could not read metrics file: {exc}")
        else:
            st.warning(f"Metrics file not found: {metrics_path}")

        st.subheader("Analysis Visualizations")
        charts = [
            ("Class Distribution", "class_distribution.png"),
            ("Probe Confusion Matrix", "probe_confusion_matrix.png"),
            ("Precision-Recall Curve", "precision_recall_curve.png"),
            ("Error Breakdown", "error_breakdown.png"),
            ("Layer Performance", "layer_performance.png"),
            ("Probe Score Distributions", "probe_score_distributions.png"),
            ("Classification Metrics Comparison", "classification_metrics_comparison.png"),
            ("Ranking Metrics Comparison", "ranking_metrics_comparison.png"),
        ]

        found_charts = 0
        for title, filename in charts:
            image_path = analysis_dir / filename
            if image_path.is_file():
                with st.expander(
                    title,
                    expanded=filename in {
                        "class_distribution.png",
                        "probe_confusion_matrix.png",
                        "precision_recall_curve.png",
                    },
                ):
                    st.image(str(image_path), use_container_width=True)
                found_charts += 1

        if found_charts == 0:
            st.warning("No analysis charts were found in the analysis directory.")

        st.subheader("Misclassified Examples")
        if csv_path.is_file():
            try:
                errors_df = pd.read_csv(csv_path)
                st.caption(f"Total misclassified examples: {len(errors_df):,}")
                st.dataframe(
                    errors_df.head(100),
                    use_container_width=True,
                    hide_index=True,
                )
                st.download_button(
                    "Download misclassified examples (CSV)",
                    data=errors_df.to_csv(index=False).encode("utf-8"),
                    file_name="misclassified_examples.csv",
                    mime="text/csv",
                )
            except (OSError, ValueError, pd.errors.ParserError) as exc:
                st.error(f"Could not read misclassified examples: {exc}")
        else:
            st.warning(f"Misclassified examples not found: {csv_path}")

        if report_path.is_file():
            with st.expander("Research Error Analysis Report"):
                try:
                    st.markdown(report_path.read_text(encoding="utf-8"))
                except OSError as exc:
                    st.error(f"Could not read research report: {exc}")

        with st.expander("Debug: files detected"):
            st.write("Project root:", str(PROJECT_ROOT))
            st.write("Analysis directory:", str(analysis_dir))
            st.write("Files found:", available_files)

elif page == "Research Details":
    st.title("Research Details")
    st.markdown(
        "**Research question:** Can classical text classifiers and probes on "
        "observer-model hidden states estimate hallucination risk in RAG answers?\n\n"
        "**Dataset:** RAGTruth, 15,090 training rows and 2,700 test rows.\n\n"
        "**Methods:** TF-IDF + Logistic Regression, DistilBERT hidden-state probes, "
        "and a TF-IDF/probe hybrid.\n\n"
        "**Interpretation:** Baseline was stronger on recorded F1 and ranking metrics, "
        "while probe/hybrid configurations show different precision/recall trade-offs. "
        "Results are exploratory because official-test metrics were examined during "
        "configuration exploration.\n\n"
        "Hidden states come from an observer DistilBERT model, not the original "
        "answer-generating model."
    )

else:
    st.title("About RAGGuard")
    st.markdown(
        "RAGGuard is a research prototype for estimating hallucination risk in "
        "retrieval-augmented generation answers.\n\n"
        "**Limitations**\n"
        "- A score is not proof that an answer is true or false.\n"
        "- Results depend on training data and distribution.\n"
        "- Verify important claims against source evidence.\n"
        "- Not intended for autonomous high-stakes decisions."
    )
