from __future__ import annotations
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd, streamlit as st, matplotlib.pyplot as plt
from src.data import make_synthetic, TARGET_NAME
from src.model import train_all_models, cross_validate
from src.visualizations import *
st.set_page_config(page_title="AltCredit | Lenddo Alternative Scoring", layout="wide", page_icon="\U0001f4b0")
with st.sidebar:
    st.header("\u2699 Config"); n=st.slider("Samples",2000,20000,10000,1000); tau=st.slider("Threshold",0.05,0.95,0.50,0.05)
    st.caption("Lenddo | Alternative Credit Scoring")
data=make_synthetic(n=n); b=train_all_models(data)
y_test=b["y_test"]; y_probas={n:b["results"][n]["y_proba"] for n in b["results"]}
best=max(b["results"],key=lambda n: b["results"][n]["metrics"].get("roc_auc",0))
c1,c2,c3,c4=st.columns(4)
c1.metric("Samples",f"{n:,}"); c2.metric("Default Rate",f"{data['positive_rate']:.1%}")
c3.metric("Best AUC",f"{b['results'][best]['metrics']['roc_auc']:.4f}"); c4.metric("Best",best)
t1,t2,t3,t4,t5=st.tabs(["\U0001f4ca Explorer","\U0001f52c Model Lab","\U0001f3af Scoring","\U0001f9e0 Behavioral","\U0001f4b0 Inclusion"])
with t1:
    st.dataframe(data["df"].head(50),use_container_width=True,height=200)
    fig,ax=plt.subplots(figsize=(5,3)); _style()
    ax.bar(["Good","Default"],[1-data["positive_rate"],data["positive_rate"]],color=["#22c55e","#f43f5e"])
    for i,v in enumerate([1-data["positive_rate"],data["positive_rate"]]): ax.text(i,v+.01,f"{v:.1%}",ha="center",color="white")
    ax.set_title("Credit Outcome Distribution",color="white"); ax.grid(True,alpha=.2)
    st.pyplot(fig)
with t2:
    rows=[{**{"Model":n},**{k:f"{v:.4f}" for k,v in r["metrics"].items() if k!="confusion_matrix"}} for n,r in b["results"].items()]
    st.dataframe(pd.DataFrame(rows).set_index("Model"),use_container_width=True)
    col_a,col_b=st.columns(2)
    with col_a: st.pyplot(plot_roc_curve(y_test,y_probas))
    with col_b: st.pyplot(plot_calibration_curve(y_test,y_probas))
    st.pyplot(plot_confusion_matrix(y_test,b["results"]["XGBoost"]["y_pred"],"XGBoost"))
    cv=cross_validate(data); cvr=[{"Model":n,"AUC":f"{s['roc_auc']['mean']:.4f}","\u00b1":f"\u00b1{s['roc_auc']['std']:.4f}"} for n,s in cv.items()]
    st.dataframe(pd.DataFrame(cvr).set_index("Model"),use_container_width=True)
with t3:
    st.subheader("Alternative Credit Scorecard")
    st.latex(r"\text{AltScore} = w_1 \cdot S_{\text{social}} + w_2 \cdot S_{\text{behavioral}} + w_3 \cdot S_{\text{device}}")
    scores=data["df"].copy()
    scores["alt_score"]=(0.3*scores["social_connectivity_score"]+0.3*scores["behavioral_score"]+0.2*scores["bill_payment_consistency"]*100+0.1*scores["transaction_regularity"]+0.1*scores["device_freshness"])
    scores["credit_band"]=pd.qcut(scores["alt_score"],q=5,labels=["Very Low","Low","Medium","High","Very High"])
    default_by_band=scores.groupby("credit_band",observed=True)["default_risk"].mean()
    fig,ax=plt.subplots(figsize=(6,4)); _style()
    colors=["#f43f5e","#f97316","#fbbf24","#22c55e","#22d3ee"]
    ax.bar(range(5),default_by_band.values,color=colors)
    ax.set_xticks(range(5)); ax.set_xticklabels(default_by_band.index); ax.set_title("Default Rate by Alt-Score Band",color="white")
    ax.set_ylabel("Default Rate"); ax.grid(True,alpha=.2)
    st.pyplot(fig)
with t4:
    st.subheader("Behavioral Feature Analysis")
    col_a,col_b=st.columns(2)
    with col_a:
        fig,ax=plt.subplots(figsize=(5,4)); _style()
        for label,color in [(0,"#22c55e"),(1,"#f43f5e")]:
            vals=data["df"][data["df"]["default_risk"]==label]["bill_payment_consistency"]
            ax.hist(vals,bins=30,alpha=0.5,color=color,label=f"{'Default' if label else 'Good'}",density=True)
        ax.set_title("Bill Payment Consistency",color="white"); ax.legend(fontsize=8); ax.grid(True,alpha=.2)
        st.pyplot(fig)
    with col_b:
        fig,ax=plt.subplots(figsize=(5,4)); _style()
        for label,color in [(0,"#22c55e"),(1,"#f43f5e")]:
            vals=data["df"][data["df"]["default_risk"]==label]["transaction_regularity"]
            ax.hist(vals,bins=30,alpha=0.5,color=color,label=f"{'Default' if label else 'Good'}",density=True)
        ax.set_title("Transaction Regularity",color="white"); ax.legend(fontsize=8); ax.grid(True,alpha=.2)
        st.pyplot(fig)
with t5:
    st.subheader("Financial Inclusion Metrics")
    st.latex(r"\text{Inclusion Rate} = \frac{\text{Approved}}{\text{Total Applicants}} \times 100\%")
    included=scores[scores["credit_band"].isin(["Medium","High","Very High"])]
    excluded=scores[scores["credit_band"].isin(["Very Low","Low"])]
    inc_rate=len(included)/len(scores)*100
    c1,c2,c3=st.columns(3)
    c1.metric("Applicants Included",f"{len(included):,}"); c2.metric("Inclusion Rate",f"{inc_rate:.1f}%")
    c3.metric("Avg Behavioral Score",f"{data['df']['behavioral_score'].mean():.1f}")
    st.markdown("AltCredit uses non-traditional signals (social connectivity, behavioral data, device freshness) to score the **unbanked and underbanked** populations who lack traditional credit history.")
    sector_incl=scores.groupby("employment_sector")["alt_score"].mean().sort_values()
    fig,ax=plt.subplots(figsize=(6,4)); _style()
    ax.barh(sector_incl.index,sector_incl.values,color="#22d3ee")
    ax.set_title("Avg Alt-Score by Employment Sector",color="white"); ax.grid(True,alpha=.2)
    st.pyplot(fig)
