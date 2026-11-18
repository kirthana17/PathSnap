import numpy as np

def viterbi_decode(candidate_rows, arcs, transition_penalty=2.5, distance_scale=30.0):
    """Decode candidate arc IDs with directed road connectivity.

    Emissions are classifier probabilities. A connected transition receives
    log(0.9); a disconnected transition receives a finite penalty so GPS gaps
    do not make the sequence impossible.
    """
    if candidate_rows.empty: return []
    grouped=[g.sort_values("probability",ascending=False).head(8) for _,g in candidate_rows.groupby("observation",sort=True)]
    if not grouped: return []
    edge_set=set(map(tuple,np.asarray(arcs,dtype=int).tolist()))
    states=[g.arc_id.astype(int).tolist() for g in grouped]
    emit=[np.log(np.maximum(g.probability.to_numpy(float),1e-12)) for g in grouped]
    dp=emit[0].copy(); back=[]
    for t in range(1,len(states)):
        prev=states[t-1]; curr=states[t]
        cur_scores=np.full(len(curr),-np.inf); ptr=np.zeros(len(curr),dtype=int)
        for j,b in enumerate(curr):
            for i,a in enumerate(prev):
                connected=(a<len(arcs) and b<len(arcs) and
                           int(arcs[a,1])==int(arcs[b,0]))
                score=dp[i]+(np.log(.9) if connected else -transition_penalty)+emit[t][j]
                if score>cur_scores[j]: cur_scores[j]=score;ptr[j]=i
        back.append(ptr);dp=cur_scores
    idx=int(np.argmax(dp)); path=[states[-1][idx]]
    for t in range(len(back)-1,-1,-1):
        idx=int(back[t][idx]);path.append(states[t][idx])
    return list(reversed(path))
