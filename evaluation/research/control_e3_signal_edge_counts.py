"""Fixed top-M edge-count control for received signal baselines only.

M is min(original edge counts); ties use lexicographic endpoint IDs.
No label-based threshold selection; no proposed representation graphs inferred.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph-directory',type=Path,required=True)
    p.add_argument('--processed',type=Path,required=True)
    p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path(__file__).with_name('results')/'e3_baselines_current')
    a=p.parse_args();sys.path.insert(0,str(a.runtime))
    import igraph as ig
    import leidenalg
    audit=json.loads((a.output/'e3_baseline_audit.json').read_text(encoding='utf8'))
    m=min(r['n_edges'] for r in audit['graph_metrics'])
    ref=pd.read_csv(a.processed/'honduras_account_splits.csv',dtype={'user_id':str}).set_index('user_id')
    ids=sorted(ref.index);index={u:i for i,u in enumerate(ids)}
    assert len(ids)==224685
    rows=[];inputs={};outputs={}
    for name in ['hash','url','combined']:
        path=a.graph_directory/name/'edges.csv';inputs[str(path)]=sha(path)
        edges=pd.read_csv(path,dtype={'source':str,'target':str},float_precision='round_trip')
        assert (edges.source<edges.target).all() and not edges.duplicated(['source','target']).any()
        assert np.isfinite(edges.weight).all() and (edges.weight>0).all()
        # Original graph weights are retained, never renormalized after trimming.
        selected=edges.sort_values(['weight','source','target'],ascending=[False,True,True],kind='stable').head(m)
        selected=selected.sort_values(['source','target']).reset_index(drop=True)
        g=ig.Graph(n=len(ids),edges=[(index[r.source],index[r.target]) for r in selected.itertuples()],directed=False)
        g.es['weight']=selected.weight.tolist()
        degree=np.array(g.degree());activeidx=np.flatnonzero(degree)
        active=g.induced_subgraph(activeidx.tolist())
        part=leidenalg.find_partition(active,leidenalg.RBConfigurationVertexPartition,weights='weight',
                                     resolution_parameter=1,n_iterations=-1,seed=42)
        membership=np.full(len(ids),-1)
        for cid,group in enumerate(part):membership[activeidx[list(group)]]=cid
        output=a.graph_directory/name/'nodes_equal_edge_count.csv'
        pd.DataFrame({'account_id':ids,'label':ref.loc[ids].label.to_numpy(),'split':ref.loc[ids].split.to_numpy(),
                      'degree':degree,'community':membership}).to_csv(output,index=False);outputs[str(output)]=sha(output)
        selected.to_csv(a.graph_directory/name/'edges_equal_edge_count.csv',index=False)
        outputedge=a.graph_directory/name/'edges_equal_edge_count.csv';outputs[str(outputedge)]=sha(outputedge)
        bad=ref.loc[ids].label.to_numpy()==0
        rows.append(dict(graph=name,n_nodes=len(ids),n_edges=g.ecount(),n_active_nodes=len(activeidx),
            n_isolates=int((degree==0).sum()),largest_component_nodes=max(g.connected_components().sizes()),
            n_communities=len(part),largest_community_nodes=max(part.sizes()),
            modularity=float(active.modularity(part.membership,weights='weight',resolution=1)),
            bad_active=int((bad & (degree>0)).sum()),bad_active_coverage=float((bad & (degree>0)).sum()/bad.sum()),
            seed=42,resolution=1,scope='full_corpus_descriptive_transductive'))
        print(json.dumps(rows[-1]),flush=True)
    pd.DataFrame(rows).to_csv(a.output/'e3_equal_edge_count_signals.csv',index=False)
    (a.output/'e3_equal_edge_count_audit.json').write_text(json.dumps(dict(status='completed_for_signal_baselines_only',
        snapshot_commit=audit['snapshot_commit'],M=m,selection='Descending original weight, ties ascending source/target string IDs',
        node_set='All 224685 manifest IDs, including isolates',algorithm='Leiden RBConfiguration',seed=42,resolution=1,iterations=-1,
        input_sha256=inputs,output_sha256=outputs,metrics=rows,
        limitation='This is a signal ablation. Baseline vs Scratch/Transferred edge-count control still requires proposed graphs.'),indent=2)+'\n',encoding='utf8')


if __name__=='__main__':main()
