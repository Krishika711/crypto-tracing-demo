import streamlit as st
import pandas as pd
import numpy as np
import joblib
import networkx as nx
import matplotlib.pyplot as plt

st.set_page_config(page_title="Illicit Bitcoin Transaction Detector", layout="wide")

STRATS = {
    "static": "Static (trained once on steps 1-34)",
    "expanding": "Expanding (retrain on all past steps)",
    "sliding": "Sliding (retrain on last 10 steps)",
}

@st.cache_data
def load_strat():
    return pd.read_csv("strategy_predictions.csv")

def prf(label, prob, thr):
    pred = prob > thr
    tp = int((pred & (label == 1)).sum())
    fp = int((pred & (label == 0)).sum())
    fn = int((~pred & (label == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f

st.title("Illicit Bitcoin Transaction Detector")
st.caption("Research demo on the Elliptic dataset. Scores are model risk estimates for "
           "review, not proof of wrongdoing. Not an investigation tool.")

t_over, t_lab, t_budget, t_unl, t_trace, t_demo = st.tabs([
    "Overview", "Strategy lab", "Review budget",
    "Unlabeled transactions", "Trace graph", "Single transaction demo"])

# ---------- Overview ----------
with t_over:
    sp = load_strat()
    st.subheader("What this project does")
    st.write(
        "Can machine learning flag suspicious Bitcoin transactions? This demo uses the public "
        "Elliptic dataset: 203,769 transactions, of which 46,564 are labeled illicit or licit. "
        "A Random Forest is trained on earlier time steps and tested on later ones, the way a "
        "detector would be used in practice.")
    st.write(
        "**The main finding:** the model works well up to time step 42, then almost stops "
        "working. The pattern of illicit activity changed (concept drift). Retraining on newer "
        "data helps, but only partly.")
    early = sp[sp["time"] <= 42]
    late = sp[sp["time"] >= 43]
    f_e = prf(early["label"].values, early["prob_static"].values, 0.5)[2]
    f_l = prf(late["label"].values, late["prob_static"].values, 0.5)[2]
    f_x = prf(late["label"].values, late["prob_expanding"].values, 0.5)[2]
    a, b, c = st.columns(3)
    a.metric("F1 before step 43 (static model)", f"{f_e:.2f}")
    b.metric("F1 from step 43 on (static model)", f"{f_l:.2f}")
    c.metric("F1 from step 43 on (retrained every step)", f"{f_x:.2f}")
    st.caption("F1 is a score from 0 to 1 that balances catching illicit transactions "
               "(recall) against false alarms (precision).")
    st.markdown(
        "**Try it:** the *Strategy lab* shows how performance changes over time, the "
        "*Review budget* tab lets you act as an investigator with limited time, and "
        "*Trace graph* lets you follow the money around a suspicious transaction.")

# ---------- Strategy lab ----------
with t_lab:
    sp = load_strat()
    st.write("Three ways to train the same Random Forest, evaluated on labeled transactions "
             "from time steps 35 to 49. Pick strategies and move the threshold to see the "
             "trade-off between catching more illicit transactions and raising more false alarms.")
    chosen = st.multiselect("Strategies to compare", list(STRATS), default=list(STRATS),
                            format_func=lambda k: STRATS[k], key="lab_choice")
    thr = st.slider("Decision threshold (flag when probability is above this)",
                    0.05, 0.95, 0.50, 0.05, key="lab_thr")
    if chosen:
        rows = []
        for k in chosen:
            for name, d in [("Steps 35-42", sp[sp["time"] <= 42]),
                            ("Steps 43-49", sp[sp["time"] >= 43]),
                            ("All (35-49)", sp)]:
                p, r, f = prf(d["label"].values, d[f"prob_{k}"].values, thr)
                rows.append({"Strategy": STRATS[k], "Period": name,
                             "Precision": round(p, 3), "Recall": round(r, 3),
                             "F1": round(f, 3)})
        st.dataframe(pd.DataFrame(rows), hide_index=True)

        chart = {}
        for k in chosen:
            s = {}
            for t, g in sp.groupby("time"):
                if g["label"].sum() > 0:
                    s[int(t)] = prf(g["label"].values, g[f"prob_{k}"].values, thr)[2]
            chart[STRATS[k]] = pd.Series(s)
        st.subheader("F1 at each time step")
        st.line_chart(pd.DataFrame(chart))
        st.caption("Later steps contain very few illicit transactions (as few as 2), so the "
                   "per-step lines are noisy. Retraining strategies assume labels for all "
                   "earlier steps are available immediately, which is optimistic for real "
                   "investigations. Results shown are from one random seed.")
    else:
        st.info("Pick at least one strategy.")

# ---------- Review budget ----------
with t_budget:
    sp = load_strat()
    st.write("Imagine you are an investigator who can review only **K transactions per time "
             "step**. The model ranks transactions by risk and you review the top K. How many "
             "illicit transactions do you catch, compared with picking K at random?")
    k_strat = st.selectbox("Model", list(STRATS), format_func=lambda k: STRATS[k],
                           key="bud_strat")
    K = st.slider("Transactions you can review per time step (K)", 1, 100, 20, key="bud_k")
    lo, hi = st.slider("Time steps", 35, 49, (35, 49), key="bud_range")

    rows = []
    for t, g in sp[(sp["time"] >= lo) & (sp["time"] <= hi)].groupby("time"):
        g = g.sort_values(f"prob_{k_strat}", ascending=False)
        n, pos = len(g), int(g["label"].sum())
        kk = min(K, n)
        rows.append({"time step": int(t), "illicit available": pos,
                     "model catches": int(g["label"].head(kk).sum()),
                     "random picking (expected)": kk * pos / n if n else 0.0})
    bud = pd.DataFrame(rows)
    if len(bud):
        tot = int(bud["illicit available"].sum())
        got = int(bud["model catches"].sum())
        rnd = float(bud["random picking (expected)"].sum())
        a, b, c = st.columns(3)
        a.metric("Illicit transactions in range", tot)
        b.metric("Caught by the model's top K", got,
                 f"{(got / tot * 100 if tot else 0):.0f}% of all illicit")
        c.metric("Caught by random picking (expected)", f"{rnd:.1f}")
        st.bar_chart(bud.set_index("time step")[["illicit available", "model catches",
                                                 "random picking (expected)"]])
        st.caption("Uses labeled test transactions only. Try the static model with a "
                   "range that starts at step 43 and compare it with the expanding model.")
        if hi >= 43:
            st.warning("Steps 43 and later: the static model's ranking is much less reliable "
                       "here (concept drift).")

# ---------- Unlabeled transactions ----------
with t_unl:
    @st.cache_data
    def load_scores():
        return pd.read_csv("unknown_scores.csv")

    scores = load_scores()
    st.write("Most transactions in the dataset have no known label. A Random Forest trained on "
             "all labeled transactions (time steps 1-49) gives each one an illicit "
             "probability. These scores cannot be verified, because no ground truth exists for "
             "these transactions.")
    st.subheader("Browse by score")
    lo2, hi2 = st.slider("Time step range", 1, 49, (1, 49), key="unl_range")
    min_prob = st.slider("Minimum illicit probability", 0.0, 1.0, 0.8, 0.05, key="unl_prob")
    sub = scores[(scores["time"] >= lo2) & (scores["time"] <= hi2)
                 & (scores["prob_illicit"] >= min_prob)]
    st.write(f"**{len(sub):,}** of {len(scores):,} unlabeled transactions match "
             "(top 500 shown, highest score first).")
    st.dataframe(sub.head(500))
    if hi2 >= 43:
        st.warning("Time steps 43 and later: in the labeled experiments the model's "
                   "illicit-class F1 dropped from about 0.91 to about 0.03 (concept drift), "
                   "so scores in this range are unreliable.")

    st.subheader("Look up one transaction")
    q = st.text_input("Transaction ID", key="unl_lookup")
    if q.strip():
        try:
            hit = scores[scores["txId"] == int(q.strip())]
        except ValueError:
            hit = scores.iloc[0:0]
        if len(hit):
            r = hit.iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Illicit probability", f"{r['prob_illicit']:.2f}")
            c2.metric("Time step", int(r["time"]))
            pct = (scores["prob_illicit"] < r["prob_illicit"]).mean() * 100
            c3.metric("Higher than", f"{pct:.0f}% of unlabeled")
            if int(r["time"]) >= 43:
                st.warning("Time step 43 or later, so this score is unreliable (concept drift).")
        else:
            st.info("This ID is not in the unlabeled set (it may be labeled, or not in the dataset).")

# ---------- Trace graph ----------
with t_trace:
    @st.cache_data
    def load_nodes():
        return pd.read_csv("nodes.csv").set_index("txId")

    @st.cache_resource
    def load_graph():
        e = pd.read_csv("edges.csv")
        return nx.from_pandas_edgelist(e, "txId1", "txId2", create_using=nx.DiGraph())

    nodes = load_nodes()
    G = load_graph()
    st.write("Follow the money around a transaction. Arrows point in the direction bitcoin "
             "flows. Known labels are shown as they are, and unlabeled transactions are "
             "colored by model score.")

    top = (nodes[(nodes["status"] == "unknown") & (nodes["time"] <= 42)]
           .sort_values("prob_illicit", ascending=False).head(20))
    pick = st.selectbox("Start from a high-risk unlabeled transaction (steps 1-42)",
                        ["(use the ID box below)"] + top.index.tolist(), key="trace_pick")
    typed = st.text_input("...or type a transaction ID", key="trace_id")
    hops = st.radio("Hops", [1, 2], horizontal=True, key="trace_hops")
    cap = st.slider("Max nodes to draw", 10, 60, 30, key="trace_cap")

    start = None
    if typed.strip():
        try:
            start = int(typed.strip())
        except ValueError:
            st.info("Please enter a numeric transaction ID.")
    elif pick != "(use the ID box below)":
        start = int(pick)

    if start is not None:
        if start not in G:
            st.info("This transaction has no connections in the edge list.")
        else:
            seen, frontier = {start}, [start]
            for _ in range(hops):
                nxt = []
                for n in frontier:
                    for m in list(G.predecessors(n)) + list(G.successors(n)):
                        if m not in seen and len(seen) < cap:
                            seen.add(m)
                            nxt.append(m)
                frontier = nxt
            sub_g = G.subgraph(seen)

            def color(tx):
                if tx not in nodes.index:
                    return "#bbbbbb"
                r = nodes.loc[tx]
                if r["status"] == "illicit":
                    return "#b00020"
                if r["status"] == "licit":
                    return "#2e7d32"
                return plt.cm.YlOrRd(float(r["prob_illicit"]))

            pos = nx.spring_layout(sub_g, seed=1)
            fig, ax = plt.subplots(figsize=(8, 6))
            nx.draw_networkx(sub_g, pos, ax=ax, with_labels=False, arrows=True,
                             node_color=[color(n) for n in sub_g.nodes()],
                             node_size=[900 if n == start else 400 for n in sub_g.nodes()],
                             edge_color="#888888")
            nx.draw_networkx_labels(sub_g, pos, ax=ax, font_size=7,
                                    labels={n: str(n)[-4:] for n in sub_g.nodes()})
            ax.axis("off")
            st.pyplot(fig)
            st.caption("Dark red = known illicit, green = known licit, yellow to red = unlabeled "
                       "(redder = higher model score), grey = not in the node file. "
                       "Labels show the last 4 digits of each ID. Large node = the one you picked.")

            rows = []
            for n in sub_g.nodes():
                if n in nodes.index:
                    r = nodes.loc[n]
                    rows.append({"txId": n, "time": int(r["time"]),
                                 "status": r["status"], "prob_illicit": r["prob_illicit"]})
            info = pd.DataFrame(rows)
            others = info[info["txId"] != start]
            c1, c2, c3 = st.columns(3)
            c1.metric("Known illicit nearby", int((others["status"] == "illicit").sum()))
            c2.metric("Known licit nearby", int((others["status"] == "licit").sum()))
            c3.metric("Unlabeled nearby", int((others["status"] == "unknown").sum()))
            st.dataframe(info)
            if start in nodes.index and int(nodes.loc[start, "time"]) >= 43:
                st.warning("This transaction is from time step 43 or later, so its score "
                           "is unreliable (concept drift).")
            st.caption("Being near known illicit transactions is a lead worth reviewing, "
                       "not proof of wrongdoing.")

# ---------- Single transaction demo ----------
with t_demo:
    @st.cache_resource
    def load_model():
        return joblib.load("rf_model.joblib")

    @st.cache_data
    def load_demo():
        return pd.read_csv("demo_data.csv")

    model = load_model()
    demo = load_demo()
    feature_cols = [c for c in demo.columns if c not in ("txId", "time", "label")]
    st.write("Random Forest trained on time steps 1-34, shown on held-out labeled "
             "transactions from steps 35-49.")

    tx = st.selectbox("Pick a held-out transaction ID", demo["txId"].tolist(), key="demo_tx")
    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05, key="demo_thr")
    reveal = st.checkbox("Reveal true label (for evaluation)", key="demo_reveal")

    row = demo[demo["txId"] == tx]
    prob = model.predict_proba(row[feature_cols])[0][1]
    pred = "ILLICIT" if prob > threshold else "LICIT"
    true_label = "Illicit" if int(row["label"].iloc[0]) == 1 else "Licit"
    step = int(row["time"].iloc[0])

    c1, c2, c3 = st.columns(3)
    c1.metric("Illicit probability", f"{prob:.2f}")
    c2.metric("Model prediction", pred)
    c3.metric("True label", true_label if reveal else "hidden")
    st.write(f"Time step: **{step}**")
    if step >= 43:
        st.warning("Time step 43 or later: predictions here are unreliable (concept drift).")

st.caption("Features are anonymized by the dataset providers, so individual feature "
           "values cannot be interpreted.")