"""Rebuild received full-corpus signal baselines with a compact igraph runtime.

Reuses notebook parsing/event helpers and fixed event/edge/Leiden rules.
Does not execute Colab mounting, manual review or owner plotting cells.
Large regenerated edges/nodes belong outside the repository.
"""
import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import re
import sys
import unicodedata
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--processed', type=Path, required=True)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--large-output', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path(__file__).with_name('results')/'e3_baselines_current')
    a = p.parse_args()
    sys.path.insert(0, str(a.runtime))
    import igraph as ig
    import leidenalg
    a.output.mkdir(parents=True, exist_ok=True)
    a.large_output.mkdir(parents=True, exist_ok=True)
    tree = json.loads((a.snapshot/'repository_tree.json').read_text(encoding='utf8'))
    rel = 'E3/graph_baselines/E3_Graph_Baselines_Nguoi4_Colab.ipynb'
    source = a.snapshot/rel
    blob = source.read_bytes()
    expected = next(t['sha'] for t in tree['tree'] if t['path'] == rel)
    assert hashlib.sha1(f'blob {len(blob)}\0'.encode()+blob).hexdigest() == expected
    notebook = json.loads(blob)
    cells = {i: ''.join(c.get('source', [])) for i,c in enumerate(notebook['cells'])}
    reference = pd.read_csv(a.processed/'honduras_account_splits.csv', dtype={'user_id':str})
    manifest = pd.read_csv(a.processed/'honduras_account_manifest.csv', dtype={'user_id':str})
    assert manifest.user_id.is_unique and reference.user_id.is_unique
    assert set(manifest.user_id) == set(reference.user_id)
    labels = manifest.set_index('user_id').label.astype(int).to_dict()
    splits = reference.set_index('user_id').split.to_dict()
    assert Counter(labels.values()) == {0:1866, 1:222819}
    assert all(labels[r.user_id] == int(r.label) for r in reference.itertuples())
    files = [a.processed/'honduras_bad_normalized.jsonl', a.processed/'honduras_good_normalized.jsonl']
    conflicts = set()
    for name in ['group_b_metadata_conflicts.csv', 'group_c_text_conflicts.csv']:
        conflicts.update(pd.read_csv(a.processed/'quality_audit'/name,dtype=str).tweet_id.dropna())
    ns = dict(Counter=Counter,defaultdict=defaultdict,datetime=datetime,timezone=timezone,
              re=re,unicodedata=unicodedata,pd=pd,json=json,hashlib=hashlib,FILES=files,labels=labels,
              conflicts=conflicts,WINDOW_SECONDS=3600,ALLOW_URL_TEXT_FALLBACK=True,
              urlparse=urlparse,urlunparse=urlunparse,parse_qsl=parse_qsl,urlencode=urlencode)
    # Use owner's definitions verbatim; preserve URL normalization constants.
    for i in [3,13,14]:
        nodes = []
        for n in ast.parse(cells[i]).body:
            if isinstance(n,ast.FunctionDef): nodes.append(n)
            elif i == 13 and isinstance(n,ast.Assign): nodes.append(n)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),f'owner_notebook_cell_{i}','exec'),ns)
    ids = sorted(labels)
    idx = {u:i for i,u in enumerate(ids)}
    truth = np.array([labels[u] for u in ids])
    split_array = np.array([splits[u] for u in ids])
    rows, quality_rows, memberships, graph_claim_checks = [], [], {}, []
    reported = pd.read_csv(a.snapshot/'E3/graph_baselines/e3_baseline_comparison.csv').set_index('graph')
    inputs = {str(f):digest(f) for f in files+[a.processed/'honduras_account_manifest.csv',a.processed/'honduras_account_splits.csv']}

    def analyze(name, edges, weights):
        print(f'{name}: Leiden on {len(edges):,} edges',flush=True)
        # Canonical lexicographic endpoints/order match owner's sorted account IDs.
        g = ig.Graph(n=len(ids), edges=edges.tolist(), directed=False)
        g.es['weight'] = weights.tolist()
        degree = np.array(g.degree())
        active_indices = np.flatnonzero(degree)
        active = g.induced_subgraph(active_indices.tolist())
        part = leidenalg.find_partition(active,leidenalg.RBConfigurationVertexPartition,
                weights='weight',resolution_parameter=1.0,n_iterations=-1,seed=42)
        # Community numbering is descriptive; ARI below is invariant to renumbering.
        groups = sorted(([int(active_indices[v]) for v in group] for group in part),key=lambda c:(-len(c),min(c)))
        membership = np.full(len(ids),-1,dtype=int)
        for cid,group in enumerate(groups): membership[group] = cid
        memberships[name] = membership
        sizes = np.array([len(c) for c in groups])
        components = g.connected_components()
        row = dict(graph=name,n_nodes=len(ids),n_active_nodes=int(len(active_indices)),n_edges=len(edges),
                   n_isolates=int((degree==0).sum()),density=2*len(edges)/(len(ids)*(len(ids)-1)),
                   n_components=len(components),largest_component_nodes=max(components.sizes()),
                   n_communities=len(groups),largest_community_nodes=int(sizes.max()),
                   median_community_size=float(np.median(sizes)),modularity=float(active.modularity(part.membership,weights='weight',resolution=1.0)),
                   scope='full_corpus_descriptive_transductive',algorithm='Leiden',seed=42,resolution=1.0)
        row['isolated_node_rate'] = row['n_isolates']/len(ids)
        rows.append(row)
        folder = a.large_output/name;folder.mkdir(exist_ok=True)
        pd.DataFrame({'source':[ids[i] for i in edges[:,0]],'target':[ids[i] for i in edges[:,1]],'weight':weights}).to_csv(folder/'edges.csv',index=False)
        pd.DataFrame({'account_id':ids,'label':truth,'split':split_array,'degree':degree,'community':membership}).to_csv(folder/'nodes.csv',index=False)
        summaries = []
        for cid, group in enumerate(groups):
            group = np.asarray(group); bad = int((truth[group]==0).sum())
            test = group[split_array[group]=='test']; testbad = int((truth[test]==0).sum())
            summaries.append(dict(graph=name,community=cid,n_nodes=len(group),bad_accounts=bad,
                bad_ratio=bad/len(group),bad_coverage=bad/1866,bad_enrichment=(bad/len(group))/(1866/len(ids)),
                test_nodes=len(test),test_bad_accounts=testbad,
                test_bad_ratio=testbad/len(test) if len(test) else None,
                test_bad_coverage=testbad/373,test_bad_enrichment=(testbad/len(test))/(373/44938) if len(test) else None))
        pd.DataFrame(summaries).to_csv(a.output/f'{name}_community_bad_metrics.csv',index=False)
        # Test label metrics describe Test members embedded in a full-corpus graph;
        # they do not turn this into a Test-only graph or validate campaigns.
        for scope,mask in [('full_corpus',np.ones(len(ids),dtype=bool)),('test_members_in_full_graph',split_array=='test')]:
            badmask = mask & (truth==0);goodmask=mask & (truth==1)
            quality_rows.append(dict(graph=name,scope=scope,n_accounts=int(mask.sum()),
                n_bad=int(badmask.sum()),bad_active=int((badmask & (degree>0)).sum()),
                good_active=int((goodmask & (degree>0)).sum()),
                bad_active_coverage=float((badmask & (degree>0)).sum()/badmask.sum())))
        for metric in reported.columns:
            if metric in row:
                claim = float(reported.loc[name,metric]);value = float(row[metric])
                graph_claim_checks.append(dict(graph=name,metric=metric,reported=claim,recomputed=value,
                    matches=bool(np.isclose(claim,value,atol=1e-9,rtol=1e-10))))
        print(json.dumps(row),flush=True)
        del g,active,part,components,groups;gc.collect()

    def event_edges(events):
        pairs = Counter();updates=0;popular=0
        for event, users in sorted(events.items()):
            if len(users)>100: popular+=1;continue
            if len(users)<2: continue
            updates+=len(users)*(len(users)-1)//2
            assert updates <= 20_000_000
            pairs.update(combinations(sorted(users),2))
        selected=sorted((u,v,w) for (u,v),w in pairs.items() if w>=2)
        e=np.array([(idx[u],idx[v]) for u,v,w in selected],dtype=np.int32)
        w=np.array([w for u,v,w in selected],dtype=float)
        return e,w,updates,popular

    # Owner's exact streaming Hash cell with only DATA/OUT paths supplied locally.
    ns.update(DATA=a.processed,OUT=a.large_output,display=lambda *x:None)
    print('Rebuilding Hash events from normalized tweets',flush=True)
    exec(compile(cells[3],'owner_notebook_cell_3','exec'),ns)
    e,w,updates,popular=event_edges(ns['events'])
    hash_e,hash_w=e,w
    quality = {'hash':dict(ns['quality'])|{'pair_updates':updates,'dropped_popular_events':popular}}
    for key in ['events','seen','observed_accounts']: ns.pop(key,None)
    gc.collect()
    analyze('hash',e,w)
    print('Rebuilding URL events from normalized tweets',flush=True)
    events,q,source_rows,url_sources,_=ns['build_signal_events']('url')
    e,w,updates,popular=event_edges(events)
    quality['url']=dict(q)|{'url_sources':dict(url_sources),'pair_updates':updates,'dropped_popular_events':popular}
    del events;gc.collect()
    analyze('url',e,w)
    print('Auditing structured retweet IDs',flush=True)
    events,q,_,_,sources=ns['build_signal_events']('cort')
    quality['cort']=dict(q)|{'retweet_sources':dict(sources),'CORT_APPROVED':False,'n_events':len(events)}
    del events;gc.collect()
    # Equal coefficient of max-normalized Hash/URL weights, union with score>0.
    hash_max=float(hash_w.max());url_max=float(w.max())
    combined={tuple(pair):.5*weight/hash_max for pair,weight in zip(hash_e,hash_w)}
    overlap=0
    for pair,weight in zip(e,w):
        key=tuple(pair);overlap+=key in combined
        combined[key]=combined.get(key,0)+.5*weight/url_max
    ordered=sorted(combined)
    ce=np.array(ordered,dtype=np.int32);cw=np.array([combined[k] for k in ordered])
    quality['combined']={'hash_max_weight':float(hash_w.max()),'url_max_weight':float(w.max()),
                         'overlapping_edges':overlap,'combined_edges':len(ce),'coefficient':.5,'combined_min_score':0}
    del combined,ordered,e,w,hash_e,hash_w;gc.collect()
    analyze('combined',ce,cw)
    aris=[]
    for left,right in [('hash','url'),('hash','combined'),('url','combined')]:
        for scope,mask in [('all_nonisolated_in_both',(memberships[left]>=0)&(memberships[right]>=0)),
                           ('all_accounts_including_isolate_code',np.ones(len(ids),dtype=bool))]:
            aris.append(dict(left=left,right=right,scope=scope,n_accounts=int(mask.sum()),
                              ari=float(adjusted_rand_score(memberships[left][mask],memberships[right][mask]))))
    pd.DataFrame(rows).to_csv(a.output/'e3_recomputed_baselines.csv',index=False)
    pd.DataFrame(quality_rows).to_csv(a.output/'e3_bad_coverage.csv',index=False)
    pd.DataFrame(graph_claim_checks).to_csv(a.output/'e3_reported_vs_recomputed.csv',index=False)
    pd.DataFrame(aris).to_csv(a.output/'e3_signal_ari.csv',index=False)
    audit=dict(status='rebuilt_from_tweets',snapshot_commit=tree['commit'],checked_at_utc=datetime.now(timezone.utc).isoformat(),
        source_notebook_sha256=digest(source),input_sha256=inputs,quality=quality,graph_metrics=rows,
        reported_metric_checks=graph_claim_checks,ari=aris,large_output=str(a.large_output.resolve()),
        environment={'python':platform.python_version(),'igraph':ig.__version__,'leidenalg':metadata.version('leidenalg'),'numpy':np.__version__,'pandas':pd.__version__},
        adaptations=['Reuse original notebook event parsing/helper functions and Hash streaming cell. No edge/threshold tuning.',
                     'Use compact igraph graph instead of duplicate NetworkX objects for RAM; canonical vertex/edge ordering and Leiden settings retained.',
                     'Skip Colab Drive mount, plotting and manual evidence review; no human-review completion claimed.',
                     'Compare numeric summaries; original per-account assignments/edges not delivered in Git, so exact partition identity with owner is unverified.',
                     'ARI involving -1 groups isolates artificially; primary comparison restricts to nodes active in both graphs.'],
        limitations=['Full-corpus protocol explicitly declared by new handoff; descriptive/transductive, not Test-only campaign evaluation.',
                     'Community Bad metrics are descriptive label association, not campaign ground truth.',
                     'Proposed Scratch/Transferred graphs are not produced by this baseline audit.'])
    (a.output/'e3_baseline_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print('DONE',a.output,flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf8')
    main()
