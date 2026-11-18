import streamlit as st
import requests
import streamlit.components.v1 as components
import subprocess,sys

st.set_page_config(page_title="PathSnap Evaluator",layout="wide")
st.title("PathSnap — GPS-to-Road Map Matching")
st.caption("Select a reserved trip to run the server-side prediction pipeline.")
api=st.text_input("API URL","http://127.0.0.1:8000")
try:
    trip_payload=requests.get(api+"/trips",timeout=3).json();trips=trip_payload["trip_ids"];partition=trip_payload["partition"]
except Exception:
    trips=[];st.warning("API is not reachable. Start it with: uvicorn src.pathsnap.api:app --host 0.0.0.0 --port 8000")
if trips:
    tid=st.selectbox("Evaluation trip",trips)
    if st.button("Run route prediction",type="primary"):
        try:
            result=requests.post(api+"/predict",json={"trip_id":tid,"partition":partition},timeout=180).json()
            st.session_state["result"]=result
            st.session_state["map_html"]=requests.get(f"{api}/map/{partition}/{tid}",timeout=180).text
        except Exception as e:st.error(str(e))
if "result" in st.session_state:
    r=st.session_state["result"]
    c1,c2=st.columns(2)
    c1.metric("GPS observations",r["observations"])
    c2.metric("Predicted road segments",len(r["predicted_arc_ids"]))
    st.write("Model:",r["model"])
    if "map_html" in st.session_state:
        st.subheader("Predicted route map")
        components.html(st.session_state["map_html"],height=600,scrolling=True)
    metrics=r.get("evaluation_metrics",{})
    if metrics:
        m1,m2,m3=st.columns(3)
        m1.metric("Top-1 accuracy",f"{metrics.get('top1_accuracy',0):.3f}")
        m2.metric("Macro F1",f"{metrics.get('macro_f1',0):.3f}")
        m3.metric("Route continuity",f"{metrics.get('route_continuity',0):.3f}")
    st.write("Viterbi-decoded arc sequence")
    st.code(", ".join(map(str,r["predicted_arc_ids"][:100])))
    st.info("Ground-truth route labels remain server-side and are never sent to the browser or prediction model. Only aggregate evaluation metrics are returned.")
