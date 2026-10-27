"""Audit delivered E3 representation-enhanced graphs; optional slow partition reruns.

Read-only inputs. Never trains models, rebuilds ANN, edits owner files or runs Git
mutations. All outputs go outside the repository. Run --self-test first.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata as metadata
import json
import logging
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def available(path):
    require(path.is_file(),f'Missing input: {path}')
    with path.open('rb') as f: head=f.read(150)
    require(not head.startswith(b'version https://git-lfs.github.com/spec/'),
            f'LFS pointer instead of real data: {path}. Hydrate the file before running this audit.')


def edges(path, index):
    available(path);logging.info('Read edges: %s',path)
    df=pd.read_csv(path,dtype={'source':str,'target':str},float_precision='round_trip')
    require(set(['source','target','weight'])<=set(df),'Edge schema incomplete')
    u=index.get_indexer(df.source);v=index.get_indexer(df.target)
    w=df.weight.to_numpy(dtype=float)
    require((u>=0).all() and (v>=0).all(),'Unknown edge endpoint')
    require((u<v).all(),'Self-loop or noncanonical edge endpoints')
    require(np.isfinite(w).all() and (w>0).all(),'Nonpositive/nonfinite weights')
    key=u.astype(np.int64)*len(index)+v
    require(len(np.unique(key))==len(key),'Duplicate undirected edges')
    result=sparse.coo_matrix((w,(u,v)),shape=(len(index),len(index))).tocsr()
    del df,u,v,w,key;gc.collect()
    return result


def properties(matrix, membership):
    co=matrix.tocoo();n=matrix.shape[0]
    degree=np.bincount(co.row,minlength=n)+np.bincount(co.col,minlength=n)
    strength=np.bincount(co.row,weights=co.data,minlength=n)+np.bincount(co.col,weights=co.data,minlength=n)
    require(((degree==0)==(membership==-1)).all(),'Isolate/community=-1 policy mismatch')
    require((membership[degree>0]>=0).all(),'Active node without a community')
    nc,cc=connected_components(matrix,directed=False)
    active=degree>0
    sizes=np.bincount(membership[active])
    sizes=sizes[sizes>0]
    total=float(co.data.sum())
    if total:
        internal=float(co.data[membership[co.row]==membership[co.col]].sum())
        masses=np.bincount(membership[active],weights=strength[active])
        q=internal/total-float(np.square(masses/(2*total)).sum())
    else:q=None
    row=dict(n_nodes=n,n_edges=matrix.nnz,n_active_nodes=int(active.sum()),n_isolates=int((~active).sum()),
        isolated_node_rate=float((~active).mean()),density=2*matrix.nnz/(n*(n-1)) if n>1 else 0,
        n_components=int(nc),largest_component_nodes=int(np.bincount(cc).max()) if n else 0,
        n_communities=len(sizes),largest_community_nodes=int(sizes.max()) if len(sizes) else 0,
        median_community_size=float(np.median(sizes)) if len(sizes) else 0,modularity=q)
    return row,degree,strength


def metric_checks(measured, claims, method, checks):
    for key,value in measured.items():
        if key not in claims or pd.isna(claims[key]):continue
        expected=float(claims[key]);ok=bool(np.isclose(value,expected,rtol=1e-9,atol=1e-8))
        checks.append(dict(method=method,metric=key,recomputed=value,reported=expected,matches=ok))


def self_test():
    b=sparse.coo_matrix(([2.,1.,3.],([0,0,3],[1,2,4])),shape=(6,6)).tocsr()
    member=np.array([0,0,0,1,1,-1])
    row,degree,strength=properties(b,member)
    require(row['n_components']==3 and row['n_isolates']==1,'Components test')
    require(np.isclose(row['modularity'],.5),'Weighted modularity test')
    require(degree.sum()==6 and np.isclose(strength.sum(),12),'Degree/strength test')
    sim=sparse.coo_matrix(([.9,.95],([0,1],[2,3])),shape=b.shape).tocsr()
    mix=.5*b/b.sum()+.5*sim/sim.sum()
    require(np.isclose(mix.sum(),1) and mix.nnz==4,'Normalization/union test')
    require(adjusted_rand_score([0,0,1,1],[5,5,7,7])==1,'ARI renumbering test')
    try:properties(b,np.zeros(6,dtype=int))
    except ValueError:pass
    else:raise AssertionError('Isolate validation must reject grouped isolates')
    print('PASS: weighted Q, components/isolates, degree/strength, sum-normalized union, ARI renumbering and invalid isolate policy')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--processed',type=Path,required=True,help='Directory containing processed Honduras manifests')
    p.add_argument('--baseline-edges',type=Path,required=True,help='Locally reconstructed Combined baseline edge list')
    p.add_argument('--output',type=Path,required=True,help='Audit output directory outside the repository')
    p.add_argument('--runtime',type=Path,help='Python package directory with igraph/leidenalg for optional reruns')
    p.add_argument('--rerun-leiden',action='store_true',help='Slow: rerun delivered three graphs using same Leiden settings')
    p.add_argument('--edge-count-control',action='store_true',help='Slow: top-M control for baseline and both enhanced graphs')
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:self_test();return 0
    if a.rerun_leiden or a.edge_count_control:
        require(a.runtime is not None and a.runtime.is_dir(),
                '--runtime must point to a package directory containing igraph and leidenalg for optional reruns')
    a.output.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler(a.output/'run.log',encoding='utf8')])
    status=dict(status='running',started_at_utc=datetime.now(timezone.utc).isoformat(),stage='preflight')
    def checkpoint(stage):
        status['stage']=stage;(a.output/'status.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf8')
        logging.info('STAGE: %s',stage)
    try:
        b=a.repo/'E3/proposed-graph';r=b/'results'
        commit=subprocess.run(['git','rev-parse','HEAD'],cwd=a.repo,capture_output=True,text=True,timeout=15)
        status['repository_commit']=commit.stdout.strip() if commit.returncode==0 else 'unavailable'
    require(a.output.resolve()!=r.resolve() and not a.output.resolve().is_relative_to(b.resolve()),'Outputs must be outside owner input folder')
    require(not a.output.resolve().is_relative_to(a.repo.resolve()),'Outputs must be outside the repository')
        ref=pd.read_csv(a.processed/'honduras_account_splits.csv',dtype={'user_id':str}).set_index('user_id')
        require(ref.index.is_unique and len(ref)==224685,'Wrong/duplicate account reference')
        ids=pd.read_csv(b/'embedding_account_ids.csv',dtype={'account_id':str})
        require(ids.account_id.is_unique and ids.account_id.tolist()==sorted(ids.account_id),'Embedding ID order/duplicates')
        index=pd.Index(ids.account_id);require(set(index)==set(ref.index),'Embedding IDs do not match split manifest')
        ref=ref.loc[index]
        require(ref.label.eq(0).sum()==1866 and (ref.split=='test').sum()==44938,'Wrong labels/split version')
        valid=np.load(b/'input_has_tfidf_signal.npy',allow_pickle=False)
        require(valid.dtype==bool and valid.shape==(len(index),),'Invalid TF-IDF signal mask')
        require(np.array_equal(valid,ids.has_tfidf_signal.astype(str).str.lower().eq('true')),'Mask and ID table mismatch')
        chosen=json.loads((b/'parameter_selection/chosen_config.json').read_text())
        run=json.loads((r/'run_config.json').read_text())
        require(not chosen['selection_uses_labels'] and chosen['scope']=='full_corpus_descriptive_transductive','Unexpected selection/scope')
        for key in ['lambda','top_k','cosine_threshold']:require(chosen[key]==run['config'][key],f'Config mismatch: {key}')
        comparison=pd.read_csv(r/'e3_proposed_comparison.csv').set_index('method')
        methods=['baseline_delivered','baseline_plus_scratch','baseline_plus_transfer']
        nodes={};checks=[];warnings=[];metrics=[];community_rows=[];coverage=[];provenance={}
        for method in methods:
            frame=pd.read_csv(r/method/'nodes.csv',dtype={'account_id':str},float_precision='round_trip')
            require(frame.account_id.tolist()==index.tolist(),f'{method}: IDs/order differ')
            require(np.array_equal(frame.label.to_numpy(),ref.label.to_numpy()),f'{method}: labels differ')
            require(np.array_equal(frame.split.to_numpy(),ref.split.to_numpy()),f'{method}: splits differ')
            require(np.isfinite(frame[['degree','strength','community']]).all().all(),f'{method}: nonfinite node data')
            require(np.equal(frame.community,np.floor(frame.community)).all(),f'{method}: noninteger communities')
            require(frame.community.between(-1,len(index)-1).all(),f'{method}: invalid community index')
            nodes[method]=frame
        checkpoint('validate_selection_table')
        pilot=pd.read_csv(b/'parameter_selection/validation_grid.csv')
        rule=json.loads((b/'parameter_selection/selection_rule.json').read_text())
        expected={(m,t,k) for m in ['scratch','transfer'] for t in rule['threshold_candidates'] for k in rule['k_candidates']}
        actual=set(zip(pilot.method,pilot.threshold,pilot.top_k))
        require(actual==expected and len(pilot)==len(expected),'Incomplete/duplicate pilot grid')
        require(not rule['labels_used'] and not rule['test_metrics_used'],'Pilot selection uses labels/Test')
        val_count=int(((ref.split=='validation').to_numpy()&valid).sum())
        require(pilot.n_nodes.eq(val_count).all(),'Pilot valid Validation node count mismatch')
        passing=[]
        for t in rule['threshold_candidates']:
            subset=pilot[(pilot.threshold==t)&(pilot.top_k==max(rule['k_candidates']))]
            if (subset.active_coverage>=rule['minimum_valid_node_coverage']).all():passing.append(t)
        require(bool(passing),'No passing pilot threshold')
        threshold=max(passing);selected_k=None
        for k in sorted(rule['k_candidates']):
            candidate=pilot[(pilot.threshold==threshold)&(pilot.top_k==k)].set_index('method')
            reference=pilot[(pilot.threshold==threshold)&(pilot.top_k==max(rule['k_candidates']))].set_index('method')
            if (candidate.active_coverage>=rule['minimum_valid_node_coverage']).all() and (candidate.lcc_fraction>=rule['minimum_lcc_retention_vs_k20']*reference.lcc_fraction).all():selected_k=k;break
        require(threshold==chosen['cosine_threshold'] and selected_k==chosen['top_k'],'Chosen parameters inconsistent with pilot rule/table')
        warnings.append('Pilot table selection recomputed; exact neighbor search for the 18 pilot configurations was not rerun.')
        warnings.append('Pilot followed earlier full-corpus exploration: not preregistration before the entire study.')
        checkpoint('read_baseline')
        baseline=edges(a.baseline_edges,index)
        raw_baseline_sum=float(baseline.sum())
        base_member=nodes[methods[0]].community.to_numpy(dtype=int)
        mask=nodes[methods[0]].degree.to_numpy()>0
        # Nodes hold small vectors; full edge matrices are processed sequentially.
        for method in methods:
            checkpoint('audit_'+method)
            if method==methods[0]:matrix=baseline
            else:
                key='scratch' if method.endswith('scratch') else 'transfer'
                cfg=json.loads((r/method/'config.json').read_text())
                require(cfg['algorithm']=='leiden' and cfg['partition_type']=='RBConfigurationVertexPartition','Wrong Leiden objective')
                require(cfg['seed']==42 and cfg['resolution']==1 and cfg['n_iterations']==-1,'Changed Leiden settings')
                for k in ['lambda','top_k','cosine_threshold']:require(cfg[k]==chosen[k],f'{method} config differs: {k}')
                sim=edges(r/f'similarity_{key}'/'edges.csv',index)
                co=sim.tocoo();require(valid[co.row].all() and valid[co.col].all(),'Bias-only account used in similarity edges')
                require((co.data>chosen['cosine_threshold']).all() and (co.data<=1+1e-6).all(),'Cosine outside selected threshold/range')
                # Independently score a fixed sample of delivered edges against the embeddings.
                z=np.load(b/f'{key}_embeddings.npy',mmap_mode='r',allow_pickle=False)
                require(z.shape==(len(index),128),f'{key} embedding shape')
                for start in range(0,len(index),4096):require(np.isfinite(z[start:start+4096]).all(),'Nonfinite embeddings')
                sample=np.random.default_rng(42).choice(sim.nnz,min(256,sim.nnz),replace=False)
                x=np.asarray(z[co.row[sample]],dtype=float);y=np.asarray(z[co.col[sample]],dtype=float)
                cosine=np.einsum('ij,ij->i',x,y)/(np.linalg.norm(x,axis=1)*np.linalg.norm(y,axis=1))
                err=float(np.max(np.abs(np.clip(cosine,-1,1)-co.data[sample])))
                require(err<=1e-5,f'{key} delivered similarity weights disagree with embeddings: {err}')
                del z,x,y,co
                matrix=edges(r/method/'edges.csv',index)
                expected=(1-chosen['lambda'])*baseline/raw_baseline_sum+chosen['lambda']*sim/float(sim.sum())
                diff=matrix-expected;maxdiff=float(np.abs(diff.data).max()) if diff.nnz else 0
                require(matrix.nnz==expected.nnz and maxdiff<=1e-12,f'{method}: enhanced graph formula/topology mismatch ({maxdiff})')
                overlap=int(baseline.multiply(sim>0).nnz)
                structural=dict(n_edges_added=sim.nnz-overlap,n_edges_removed=0,similarity_baseline_overlap=overlap,
                    active_partition_ari_vs_baseline=adjusted_rand_score(base_member[mask],nodes[method].community.to_numpy()[mask]))
                metric_checks(structural,comparison.loc[method],method,checks)
                provenance[key]=dict(similarity_sample_size=len(sample),max_cosine_error=err,enhanced_max_weight_error=maxdiff,
                    n_similarity_edges=sim.nnz,baseline_weight_mass=1-chosen['lambda'],similarity_weight_mass=chosen['lambda'])
                del sim,expected,diff;gc.collect()
            frame=nodes[method];member=frame.community.to_numpy(dtype=int)
            row,degree,strength=properties(matrix,member)
            require(np.array_equal(degree,frame.degree.to_numpy()),f'{method}: node degrees do not match edges')
            require(np.allclose(strength,frame.strength,atol=1e-10,rtol=1e-8),f'{method}: node strength does not match edges')
            metric_checks(row,comparison.loc[method],method,checks)
            metric_checks(row,pd.read_csv(r/method/'metrics.csv').iloc[0],method+'_metrics_file',checks)
            row.update(method=method);metrics.append(row)
            for scope,selection in [('full_corpus',np.ones(len(index),dtype=bool)),('test_members_in_full_graph',ref.split.eq('test').to_numpy())]:
                bad=selection&ref.label.eq(0).to_numpy();good=selection&ref.label.eq(1).to_numpy()
                coverage.append(dict(method=method,scope=scope,n_accounts=int(selection.sum()),n_bad=int(bad.sum()),
                    bad_active=int((bad&(degree>0)).sum()),good_active=int((good&(degree>0)).sum()),
                    bad_active_coverage=float((bad&(degree>0)).sum()/bad.sum())))
            for cid in sorted(set(member)-{-1}):
                selected=member==cid;bad=int((ref.label.eq(0).to_numpy()&selected).sum());size=int(selected.sum())
                t=selected&ref.split.eq('test').to_numpy();tb=int((ref.label.eq(0).to_numpy()&t).sum())
                community_rows.append(dict(method=method,community=int(cid),n_nodes=size,bad_accounts=bad,bad_ratio=bad/size,
                    bad_coverage=bad/1866,bad_enrichment=(bad/size)/(1866/len(index)),test_nodes=int(t.sum()),test_bad_accounts=tb,
                    test_bad_coverage=tb/373))
            pd.DataFrame(checks).to_csv(a.output/'metric_checks.csv',index=False)
            pd.DataFrame(metrics).to_csv(a.output/'verified_graph_metrics.csv',index=False)
            pd.DataFrame(coverage).to_csv(a.output/'bad_coverage.csv',index=False)
            pd.DataFrame(community_rows).to_csv(a.output/'community_bad_metrics.csv',index=False)
            logging.info('%s metrics audited; delivered Q=%s',method,row['modularity'])
            if method!=methods[0]:del matrix
            gc.collect()
        mismatches=[v for v in checks if not v['matches']]
        # Provenance hashes are done by the user-run job, not claimed from static review.
        checkpoint('hash_sources')
        for f in [b/'E3_Proposed_Graphs_Nguoi5.ipynb',b/'scratch_embeddings.npy',b/'transfer_embeddings.npy',
                  b/'embedding_account_ids.csv',a.baseline_edges,*r.glob('*/edges.csv'),*r.glob('*/nodes.csv'),
                  *r.glob('*/config.json'),r/'run_config.json',* (b/'parameter_selection').glob('*')]:
            if f.is_file():provenance[str(f)]=dict(bytes=f.stat().st_size,sha256=sha(f))
        # Compare E2 checkpoint hashes recorded by the graph run with local files.
        model_hash_checks=[]
        for remote,expected in run.get('input_sha256',{}).items():
            if remote.endswith('.pt'):
                path=a.repo/'E2/artifacts'/Path(remote).name
                if path.exists():
                    actual=sha(path);model_hash_checks.append(dict(file=str(path),expected=expected,actual=actual,matches=actual==expected))
        if any(not x['matches'] for x in model_hash_checks):warnings.append('E2 checkpoint hashes differ from graph-run provenance; inspect before associating graphs with the current E2 run.')
        # Optional slow stages: fixed delivered topology, no ANN regeneration.
        partitions=[]
        if a.rerun_leiden or a.edge_count_control:
            checkpoint('prepare_optional_leiden')
            sys.path.insert(0,str(a.runtime))
            import igraph as ig
            import leidenalg
            m=min(int(comparison.loc[name,'n_edges']) for name in methods)
            for method in methods:
                matrix=baseline if method==methods[0] else edges(r/method/'edges.csv',index)
                co=matrix.tocoo();order=np.lexsort((co.col,co.row));u=co.row[order];v=co.col[order];w=co.data[order]
                for mode,select in [('rerun',a.rerun_leiden),('equal_edge_count',a.edge_count_control)]:
                    if not select:continue
                    checkpoint(mode+'_'+method)
                    if mode=='equal_edge_count':
                        picked=np.lexsort((v,u,-w))[:m];picked=picked[np.lexsort((v[picked],u[picked]))]
                        aa,bb,ww=u[picked],v[picked],w[picked]
                    else:aa,bb,ww=u,v,w
                    active=np.unique(np.concatenate([aa,bb]));mapping=np.full(len(index),-1);mapping[active]=np.arange(len(active))
                    graph=ig.Graph(n=len(active),edges=np.column_stack([mapping[aa],mapping[bb]]),directed=False)
                    graph.es['weight']=ww.tolist()
                    part=leidenalg.find_partition(graph,leidenalg.RBConfigurationVertexPartition,weights='weight',resolution_parameter=1,n_iterations=-1,seed=42)
                    membership=np.full(len(index),-1,dtype=int)
                    for cid,group in enumerate(part):membership[active[np.asarray(group,dtype=int)]]=cid
                    output=sparse.coo_matrix((ww,(aa,bb)),shape=baseline.shape).tocsr()
                    row,degree,strength=properties(output,membership)
                    shared=mask&(degree>0)
                    row.update(method=method,mode=mode,ari_vs_delivered_same_method=float(adjusted_rand_score(nodes[method].community.to_numpy()[degree>0],membership[degree>0])),
                        ari_vs_delivered_baseline_on_shared_active=float(adjusted_rand_score(base_member[shared],membership[shared])),
                        bad_active_coverage=float(((ref.label.to_numpy()==0)&(degree>0)).sum()/1866))
                    partitions.append(row)
                    pd.DataFrame(partitions).to_csv(a.output/'leiden_rerun_and_controls.csv',index=False)
                    pd.DataFrame({'account_id':index,'community':membership,'degree':degree,'strength':strength}).to_csv(a.output/f'{mode}_{method}_nodes.csv',index=False)
                    del graph,part,output;gc.collect()
                if method!=methods[0]:del matrix
                gc.collect()
        warnings.extend(['No E2 retraining or full embedding generation replayed.',
            'Embedding sample checks validate delivered cosine edges, not ANN exactness/recall over all accounts.',
            'Full corpus includes training accounts: descriptive/transductive, not held-out campaign detection.',
            'Baseline edges here are a prior local reconstruction, not the missing original baseline edge export.',
            'Lambda sensitivity not run; do not call it completed based on the threshold/k pilot.'])
        status.update(status='completed_with_metric_mismatches' if mismatches else 'completed',stage='done',
            completed_at_utc=datetime.now(timezone.utc).isoformat(),metric_mismatches=len(mismatches))
        audit=dict(**status,graph_metrics=metrics,metric_checks=checks,source_provenance=provenance,
            e2_checkpoint_checks=model_hash_checks,optional_partition_runs=partitions,warnings=warnings,
            selection=dict(threshold=threshold,top_k=selected_k,lambda_=chosen['lambda'],validation_valid_accounts=val_count),
            environment=dict(python=platform.python_version(),packages={name:metadata.version(name) for name in ['numpy','pandas','scipy','scikit-learn']}))
        (a.output/'audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
        report=['# Kiểm định output E3 proposed','',f"Trạng thái: {status['status']}. Đối chiếu metric lệch: {len(mismatches)}.",'',
            'Đã tính lại graph metrics, weighted modularity của partition bàn giao, degrees/strength, Bad metrics, ARI và công thức trộn graph. Không tự huấn luyện lại model hoặc dựng lại ANN.','',
            '## Điểm cần đọc','',*[f'- {w}' for w in warnings],'',
            '## Output','', '- verified_graph_metrics.csv / metric_checks.csv: số tính lại và đối chiếu.',
            '- community_bad_metrics.csv / bad_coverage.csv: phân bố Bad, tách scope full corpus và Test members.',
            '- leiden_rerun_and_controls.csv: chỉ có khi bật tùy chọn chạy chậm.',
            '- audit.json: hash/config/môi trường; run.log và status.json: tiến độ/lỗi.','',
            'Nếu có metric mismatch hoặc checkpoint provenance mismatch, giữ nguyên evidence để review; không tự đổi số nguồn.']
        (a.output/'REVIEW_RESULTS.md').write_text('\n'.join(report)+'\n',encoding='utf8')
        checkpoint('done')
        return 2 if mismatches else 0
    except Exception as e:
        status.update(status='failed',error=str(e),failed_at_utc=datetime.now(timezone.utc).isoformat())
        (a.output/'status.json').write_text(json.dumps(status,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
        logging.exception('Audit stopped. Inputs and completed output files were preserved.')
        return 1


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf8')
    sys.exit(main())
