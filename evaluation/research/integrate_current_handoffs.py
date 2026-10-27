"""Integrate verified current E2/E3 audits, preserving earlier snapshots as history."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_research_evaluation import classification_scores, markdown_table as _markdown_table, probability_values, sha256


def markdown_table(rows, columns):
    return _markdown_table(rows.to_dict('records') if isinstance(rows,pd.DataFrame) else rows, columns)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--dummy-report-snapshot',type=Path,required=True)
    p.add_argument('--processed',type=Path,required=True)
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--results',type=Path,default=Path(__file__).with_name('results'))
    a=p.parse_args();r=a.results
    e2path=r/'e2_handoff_current/e2_handoff_audit.json'
    e3path=r/'e3_baselines_current/e3_baseline_audit.json'
    e2=json.loads(e2path.read_text(encoding='utf8'));e3=json.loads(e3path.read_text(encoding='utf8'))
    tree=json.loads((a.snapshot/'repository_tree.json').read_text(encoding='utf8'))
    assert e2['snapshot_commit']==e3['snapshot_commit']==tree['commit']
    assert e2['neural_inference_passed']
    for rel,info in e2['artifact_evidence'].items():
        if (a.snapshot/rel).is_file(): assert sha256(a.snapshot/rel)==info['sha256']
    assert sha256(a.snapshot/'E3/graph_baselines/E3_Graph_Baselines_Nguoi4_Colab.ipynb')==e3['source_notebook_sha256']
    for name,h in e3['input_sha256'].items(): assert sha256(Path(name))==h
    history=r/'history_before_20261027';history.mkdir(exist_ok=True)
    # One-time archive: subsequent integration never overwrites historical results.
    archive_names=['E1_E2_E3_COMPARISON.md','classification_comparison.csv','graph_comparison.csv',
                   'community_metrics.csv','activity_analysis.csv','handoff_status.csv','ablation_results.csv',
                   'e2_scratch_transfer_comparison.png']
    for name in archive_names:
        if (r/name).exists() and not (history/name).exists(): (history/name).write_bytes((r/name).read_bytes())
    old=pd.read_csv(history/'classification_comparison.csv')
    base=old[(old.experiment=='E1')|((old.experiment=='E2')&old.method.isin(['DummyClassifier','LogisticRegression']))].copy()
    # Verify baseline sources are unchanged in the current tree before reuse.
    baseline_paths=['Fox8 baseline bot detection/evaluation/e1_predictions.csv','E2/evaluation/e2_predictions.csv']
    manifest=json.loads((Path(__file__).with_name('research_manifest.json')).read_text(encoding='utf8'))
    baseline_snapshot=a.snapshot.parent/manifest['snapshot_commit']
    baseline_tree=json.loads((baseline_snapshot/'repository_tree.json').read_text(encoding='utf8'))
    for rel in baseline_paths:
        expected=next(t['sha'] for t in tree['tree'] if t['path']==rel)
        previous=next(t['sha'] for t in baseline_tree['tree'] if t['path']==rel)
        assert expected==previous, f'Baseline changed; recompute before reuse: {rel}'
        current=a.snapshot/rel
        if current.is_file():
            b=current.read_bytes()
            assert hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest()==expected
    ref=pd.read_csv(a.processed/'honduras_account_splits.csv',dtype={'user_id':str})
    test=ref[ref.split=='test'];assert len(test)==44938 and (test.label==0).sum()==373
    dummy=classification_scores(test.label,np.ones(len(test),dtype=int),np.zeros(len(test)),'0','1')
    prior=base[(base.method=='DummyClassifier')&(base.split=='test')].iloc[0]
    for key in ['positive_f1','macro_f1','average_precision']: assert np.isclose(prior[key],dummy[key],rtol=0,atol=1e-12)
    report_path=a.dummy_report_snapshot/'E2/e2_results_report.md'
    branch_tree=json.loads((a.dummy_report_snapshot/'repository_tree.json').read_text(encoding='utf8'))
    content=report_path.read_bytes()
    assert hashlib.sha1(f'blob {len(content)}\0'.encode()+content).hexdigest()==next(t['sha'] for t in branch_tree['tree'] if t['path']=='E2/e2_results_report.md')
    line=next(s for s in content.decode('utf8').splitlines() if s.startswith('| **Dummy Classifier'))
    numbers=[float(s.strip()) for s in line.split('|')[2:7]]
    keys=['positive_precision','positive_recall','positive_f1','macro_f1','average_precision']
    dummychecks=[dict(metric=k,reported=v,recomputed=dummy[k],matches=bool(np.isclose(v,dummy[k],atol=5e-5,rtol=0))) for k,v in zip(keys,numbers)]
    pd.DataFrame(dummychecks).to_csv(r/'e2_handoff_current/dummy_report_audit.csv',index=False)
    extra=pd.DataFrame(e2['classification_results']);extra['snapshot_commit']=tree['commit']
    extra.loc[extra.method!='LR (No tweet_count)','status']='checkpoint_inference_verified_report_mismatch_protocol_pending'
    extra.loc[extra.method=='LR (No tweet_count)','source']='e2_handoff_current/lr_ablation_predictions.csv'
    for rel in baseline_paths:
        matching=base.source.eq(rel);base.loc[matching,'snapshot_commit']=tree['commit']
    classification=pd.concat([base,extra],ignore_index=True)
    classification.to_csv(r/'classification_comparison.csv',index=False)
    graph=pd.read_csv(r/'e3_baselines_current/e3_recomputed_baselines.csv')
    graph['snapshot_commit']=tree['commit'];graph['status']='rebuilt_from_tweets_summary_compared_partition_identity_unverified'
    graph.to_csv(r/'graph_comparison.csv',index=False)
    communities=pd.concat([pd.read_csv(r/f'e3_baselines_current/{name}_community_bad_metrics.csv') for name in ['hash','url','combined']],ignore_index=True)
    communities.to_csv(r/'community_metrics.csv',index=False)
    # Refresh the activity analysis for current checkpoints rather than old labels.
    feature=pd.read_csv(a.features,dtype={'account_id':str}).set_index('account_id')
    preds=pd.read_csv(a.snapshot/'E2/artifacts/e2_predictions.csv',dtype=str,keep_default_na=False)
    counts=preds.account_id.map(feature.tweet_count)
    assert counts.notna().all()
    bins=pd.cut(counts,[-1,0,1,5,30,np.inf],labels=['0','1','2-5','6-30','31+'])
    activity=[]
    for method in ['scratch','transfer']:
        for label in bins.cat.categories:
            s=preds[bins==label]
            if len(s): activity.append(dict(experiment='E2',method=method,activity_bin=label,**classification_scores(s.true_label,s['pred_'+method],probability_values(s['prob_bad_'+method]),'0','1')))
    pd.DataFrame(activity).to_csv(r/'e2_handoff_current/activity_analysis.csv',index=False)
    previous_activity=pd.read_csv(history/'activity_analysis.csv')
    previous_activity=previous_activity[(previous_activity.experiment=='E2') & previous_activity.run_id.isin(['e2_majority_class','e2_logistic_regression'])]
    activity_frame=pd.DataFrame(activity)
    activity_frame['run_id']='e2_'+activity_frame.method+'_received'
    activity_frame['scope']='test'
    activity_frame['analysis']='classification_current_checkpoint'
    graph_activity=[]
    for name in ['hash','url','combined']:
        nodes=pd.read_csv(Path(e3['large_output'])/name/'nodes.csv',dtype={'account_id':str})
        counts=nodes.account_id.map(feature.tweet_count);assert counts.notna().all()
        node_bins=pd.cut(counts,[-1,0,1,5,30,np.inf],labels=['0','1','2-5','6-30','31+'])
        for label in node_bins.cat.categories:
            sub=nodes[node_bins==label]
            if not len(sub):continue
            bad=sub.label.eq(0);good=sub.label.eq(1)
            graph_activity.append(dict(experiment='E3',run_id='e3_'+name+'_leiden',scope='full_corpus',activity_bin=label,
                analysis='active_node_coverage',n_accounts=len(sub),n_positive=int(bad.sum()),n_negative=int(good.sum()),
                n_active=int(sub.degree.gt(0).sum()),bad_active_coverage=float((bad&sub.degree.gt(0)).sum()/bad.sum()) if bad.sum() else None,
                good_active_coverage=float((good&sub.degree.gt(0)).sum()/good.sum()) if good.sum() else None))
    pd.concat([previous_activity,activity_frame,pd.DataFrame(graph_activity)],ignore_index=True).to_csv(r/'activity_analysis.csv',index=False)
    edge_control_path=r/'e3_baselines_current/e3_equal_edge_count_audit.json'
    edge_control=json.loads(edge_control_path.read_text(encoding='utf8')) if edge_control_path.exists() else None
    if edge_control:
        assert edge_control['snapshot_commit']==tree['commit']
        for path,h in edge_control['input_sha256'].items(): assert sha256(Path(path))==h
    ablations=[
        dict(analysis='E2 Scratch vs Transfer',status='current_checkpoints_verified_report_stale_protocol_pending',result='e2_handoff_current/scratch_transfer_deltas.csv'),
        dict(analysis='E2 with/without tweet_count',status='reproduced_for_LR',result='e2_handoff_current/e2_verified_metrics.csv'),
        dict(analysis='E3 Hash vs URL vs Combined',status='rebuilt_from_tweets_full_corpus',result='e3_baselines_current/e3_recomputed_baselines.csv'),
        dict(analysis='E3 CoRT availability',status='audited_conditional_signal_not_used',result='e3_baselines_current/e3_baseline_audit.json'),
        dict(analysis='E3 baseline vs +Scratch vs +Transferred',status='pending_proposed_graph_artifacts',result=''),
        dict(analysis='E3 equal-edge-count control',status='pending_paired_proposed_graphs',result=''),
        dict(analysis='E3 signal equal-edge-count control',status='completed' if edge_control else 'pending',result='e3_baselines_current/e3_equal_edge_count_signals.csv' if edge_control else ''),
        dict(analysis='E3 lambda/top-k/threshold/window',status='pending_approved_grid_and_runs',result=''),
        dict(analysis='E3 signal partition ARI',status='descriptive_calculated_explicit_active_intersection',result='e3_baselines_current/e3_signal_ari.csv'),
        dict(analysis='E2 activity analysis',status='refreshed_current_predictions',result='e2_handoff_current/activity_analysis.csv')]
    pd.DataFrame(ablations).to_csv(r/'ablation_results.csv',index=False)
    handoffs=[dict(experiment='E1',method='Encoder and full Test inference',status='verified_unchanged'),
        dict(experiment='E2',method='Dummy',status='baseline_verified_new_report_MacroF1_AP_incorrect_branch_only'),
        dict(experiment='E2',method='Scratch / Transfer',status='current_checkpoints_verified_report_and_protocol_need_revision'),
        dict(experiment='E3',method='Hash / URL / Combined Leiden',status='received_and_rebuilt_original_edges_nodes_not_delivered_in_Git'),
        dict(experiment='E3',method='CoRT',status='conditional_exclusion_audited'),
        dict(experiment='E3',method='Combined + Scratch / Transferred',status='not_found_in_audited_snapshot')]
    pd.DataFrame(handoffs).to_csv(r/'handoff_status.csv',index=False)
    neural=extra[extra.method!='LR (No tweet_count)']
    fig,ax=plt.subplots(figsize=(8.5,4.4));x=np.arange(5)
    for i,row in enumerate(neural.to_dict('records')):
        ax.bar(x+(i-.5)*.34,[row[k] for k in keys],width=.34,label=row['method'])
    ax.set_xticks(x,['Bad precision','Bad recall','Bad F1','Macro F1','AP']);ax.set_ylim(0,1.12)
    ax.set_title('Current E2 checkpoints | Test n=44,938');ax.legend(loc='upper center',fontsize=8)
    ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(r/'e2_scratch_transfer_comparison.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(11,3.6))
    for ax,key,title in zip(axs,['n_edges','n_isolates','modularity'],['Edges','Isolates','Weighted modularity']):
        ax.bar(graph.graph,graph[key],color=['#555555','#aaaaaa','#222222']);ax.set_title(title);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('E3 signal baselines | Full corpus | Leiden');fig.tight_layout();fig.savefig(r/'e3_signal_baselines.png',dpi=180);plt.close(fig)
    checks=pd.DataFrame(e3['reported_metric_checks']);mismatches=checks[~checks.matches]
    coverage=pd.read_csv(r/'e3_baselines_current/e3_bad_coverage.csv')
    q=e3['quality']['cort'];cort_available=q.get('retained_valid_rows',0)-q.get('rows_without_signal',0)
    report=[
        '# Tổng hợp E1 / E2 / E3 đã kiểm định', '',
        f"Nguồn hiện hành [{tree['commit'][:12]}](https://github.com/luuquanghuy-ui/SNA-GROUP3/tree/{tree['commit']}). Thời điểm kiểm định {datetime.now(timezone.utc).isoformat()}. Các đợt trước nằm trong `history_before_20261027/` và `e2_handoff/`; không dùng delta cũ cho checkpoint hiện tại.", '',
        '## 1. Phạm vi và mức kiểm tra', '',
        'E1 giữ nguyên artifact đã kiểm định, replay inference đủ 456 Test account. E2 tính lại metrics và replay hai checkpoint mới trên đủ 44.938 Test account, đồng thời tái huấn luyện hai LR bằng đúng hàm tác giả. E3 dựng lại Hash/URL/Combined từ normalized tweets, cùng rule events/weights và Leiden; dùng igraph gọn RAM thay các bản sao NetworkX. Không chạy lại neural training, không sửa notebook/model nguồn và không nhận đã làm manual review.', '',
        'Tất cả ID giữ dạng chuỗi. E1 Bot=1; E2 Bad=0. Bad/Good không được đồng nhất Bot/Human. Mỗi thí nghiệm giữ tập đánh giá riêng: các model E1 dùng chung Test Fox8, các model E2 dùng chung Test Honduras; E1 và E2 không dùng chung một dataset.', '',
        '## 2. E1 baseline và encoder', '',
        markdown_table(classification[(classification.experiment=='E1')],['method','n_accounts','positive_precision','positive_recall','positive_f1','macro_f1','average_precision','tp','fp','fn','tn']), '',
        'LR/MLP cùng Bot-F1=1, nên chưa có cải thiện F1 so với LR. RF có một FP. Cả 1.140 Bot thuộc Fox8, Human từ nguồn khác: source-label confounding chưa được loại bỏ khi bỏ cột dataset. Không coi điểm Test cao là bằng chứng tổng quát ngoài nguồn.', '',
        'Encoder: CSV 224.685 ID × 128 chiều đã kiểm tra schema/ID/label/finite, tái tạo 128 vector mẫu khớp checkpoint E1. Ba model E1 replay nhãn khớp toàn Test. Không tái sinh toàn CSV, không retrain E1 trong lần này. Xem `encoder_delivery_audit.json` và `e1_checkpoint_inference_audit.json`.', '',
        '## 3. E2 kết quả từ predictions hiện hành', '',
        markdown_table(classification[(classification.experiment=='E2')&(classification.split=='test')],['method','n_accounts','positive_precision','positive_recall','positive_f1','macro_f1','average_precision','tp','fp','fn','tn']), '',
        'Test giữ 373 Bad và 44.565 Good. Hai neural models dùng văn bản tối đa 30 tweet/account, cap 5.000 ký tự, E1 TF-IDF 10.000 chiều; CSV embedding E1 128 chiều gửi riêng không phải đầu vào của notebook E2. Scratch không dùng trọng số encoder Fox8 nhưng vẫn dùng TF-IDF Fox8 đã lưu: cần mô tả rõ đối chứng này.', '',
        '### 3.1. Transfer trừ Scratch của đúng run mới', '',
        markdown_table(e2['deltas'],['metric','scratch','transfer','transfer_minus_scratch','difference_percentage_points']), '',
        'Run mới cho Transfer tăng Precision nhưng giảm Recall, Bad-F1, Macro-F1 và AP. Chưa có bằng chứng transfer cải thiện phân loại trong run này. Không suy luận từ run cũ hoặc gộp nhiều metric thành kết luận tốt hơn. Chưa có multi-seed confidence intervals.', '',
        '### 3.2. Checkpoint và báo cáo không đồng bộ', '',
        markdown_table([dict(method=k,**v) for k,v in e2['neural_inference'].items()],['method','n_accounts','prediction_agreement_fraction','prediction_mismatches','max_absolute_probability_difference']), '',
        'Cả hai checkpoint replay khớp toàn bộ nhãn predictions; bảng E2_RESULTS_TABLE.csv/report vẫn ghi số của run cũ. Chi tiết từng chênh lệch trong `e2_handoff_current/e2_handoff_audit.json`, mục neural_metric_checks. Sai khác xác suất CPU/GPU cỡ 1e-7; vectorizer được lưu sklearn 1.6.1, runtime kiểm định 1.8.0. Đây không phải bằng chứng runtime huấn luyện giống hệt.', '',
        '### 3.3. Dummy mới thêm trên nhánh', '',
        f"Báo cáo Dummy ở nhánh `{branch_tree['commit']}`; chưa thấy thay đổi đó trên main đã kiểm tra. Chỉ có cập nhật Markdown, không có prediction/model Dummy mới kèm theo. Với most_frequent toàn Good và score hằng, kết quả bằng baseline Dummy đã có: Bad P/R/F1=0, Macro-F1≈0,497916, AP=373/44.938≈0,008300.", '',
        markdown_table(dummychecks,['metric','reported','recomputed','matches']), '',
        'AP là average precision theo thứ hạng điểm; không mặc định là diện tích PR bằng quy tắc hình thang. Xem `e2_handoff_current/dummy_report_audit.csv`.', '',
        '### 3.4. Ablation tweet_count và activity', '',
        markdown_table(e2['lr_ablation_reproduction'],['method','matches_reported_metrics','positive_precision','positive_recall','positive_f1','macro_f1','average_precision']), '',
        'LR bỏ tweet_count làm Bad-F1 giảm khoảng 0,004475 và AP giảm khoảng 0,102344; không kết luận model chỉ học shortcut tweet_count. Bảng activity của hai checkpoint mới được tính lại theo bins cố định 0,1,2–5,6–30,31+ trong `e2_handoff_current/activity_analysis.csv`. Chưa có neural tweet-budget ablation.', '',
        '### 3.5. Protocol E2 còn cần xử lý', '',
        markdown_table(e2['protocol_issues'],['severity','item','finding','action']), '',
        '## 4. E3 Hash / URL / Combined + Leiden', '',
        'Bản bàn giao Như công bố nhóm chuyển sang full Honduras corpus. Báo cáo này ghi nhận scope đó, không tự ép Test-only. So với proposal trước, shared-event counts và max normalization thay TF-IDF/cosine; phương pháp hiện hành đã được chốt theo xác nhận của người dùng ngày 27/10/2026; báo cáo phân biệt thiết kế cũ với triển khai hiện tại. Full-corpus graph mang tính mô tả/transductive, không là held-out campaign detection.', '',
        'Giữ toàn bộ 224.685 node; cửa sổ UTC 1 giờ; event 2–100 account; single-signal weight là số event chung và giữ weight≥2. URL ưu tiên structured fields rồi text fallback, có chuẩn hóa. Combined là trung bình hai weight đã chia max từng signal, giữ union với score>0. Leiden RBConfiguration, resolution=1, seed=42, iterations=-1; isolate=-1.', '',
        markdown_table(graph,['graph','n_nodes','n_edges','density','n_isolates','largest_component_nodes','n_communities','largest_community_nodes','modularity']), '',
        f"Đã dựng lại từ tweet; {len(checks)-len(mismatches)}/{len(checks)} số tóm tắt đối chiếu khớp trong tolerance đã công bố. Bảng đầy đủ `e3_baselines_current/e3_reported_vs_recomputed.csv`. Có thể khác phân hoạch/cộng đồng nếu môi trường khác; không đánh dấu khớp partition chỉ vì số cộng đồng giống.", '',
        markdown_table(mismatches,['graph','metric','reported','recomputed','matches']) if len(mismatches) else 'Không có metric tóm tắt lệch trong các cột đã bàn giao.', '',
        f"CoRT audit {q.get('retained_valid_rows',0):,} dòng hợp lệ, có structured retweet target {cort_available:,} dòng. CORT_APPROVED=False; không suy retweet ID từ RT @, không tự bật CoRT. Xem quality.cort trong audit.", '',
        '### 4.1. Bad Ratio / Coverage / Enrichment', '',
        markdown_table(coverage,['graph','scope','n_accounts','n_bad','bad_active','good_active','bad_active_coverage']), '',
        'Mỗi cộng đồng có Ratio=Bad/n_nodes, Coverage=Bad/1.866, Enrichment=Ratio/(1.866/224.685). Cột test_* chỉ mô tả thành viên Test nằm trong full graph, dùng mẫu số 373/44.938; không đổi node set thành Test-only. File `community_metrics.csv` chứa toàn bộ cộng đồng và mẫu số rõ ràng. Không cộng các coverage giữa các graph như thể tập độc lập.', '',
        '### 4.2. So partition giữa các tín hiệu', '',
        markdown_table(e3['ari'],['left','right','scope','n_accounts','ari']), '',
        'ARI chính giới hạn node có cạnh ở cả hai graph. Bản all-account dùng chung code -1 có thể bị isolates chi phối, chỉ đưa để cảnh báo. Đây là ARI giữa partition tín hiệu, chưa phải ARI Baseline vs Proposed và không là campaign accuracy.', '',
        '### 4.3. Giới hạn bàn giao E3', '',
        'Git mới có bảng metrics/config/community sizes, chưa có edges và per-account nodes của các baseline để đối chiếu trực tiếp partition với tác giả. Graph/assignments tự tái dựng được lưu ngoài repo và hash trong audit; không giả làm file gốc Như bàn giao. Chưa có Combined+Scratch/+Transferred trên snapshot đã kiểm tra. Louvain cũ chỉ giữ tham chiếu trong lịch sử.', '',
        '### 4.4. Đối chứng tín hiệu cùng số cạnh', '',
        f"Đã giữ top-M weight với M={edge_control['M']:,}, bằng số cạnh nhỏ nhất giữa ba baseline, hòa theo endpoint ID dạng chuỗi. Giữ toàn bộ node, original weights và cùng Leiden settings; không dùng nhãn chọn M. Đây là phép ablation của người kiểm định, không phải run tác giả bàn giao." if edge_control else 'Chưa thực hiện đối chứng cùng số cạnh.', '',
        markdown_table(edge_control['metrics'],['graph','n_edges','n_active_nodes','n_isolates','largest_component_nodes','n_communities','modularity','bad_active_coverage']) if edge_control else '', '',
        'Khi giữ cùng số cạnh, URL có coverage Bad lớn hơn Hash/Combined bị cắt xuống top-M. Đây là kết quả mô tả ở quy tắc trimming đã công bố, không là bằng chứng URL phát hiện campaign tốt hơn. So sánh Baseline/+Scratch/+Transferred cùng số cạnh vẫn chờ graph proposed. File activity_analysis.csv đã cập nhật cả E2 checkpoint mới và coverage của ba graph theo mức hoạt động.', '',
        '## 5. Tổng hợp theo câu hỏi nghiên cứu', '',
        '- RQ1: encoder E1 có artifact hoạt động, nhưng cùng F1 với LR và còn source-label confounding; chưa chứng minh ưu thế tổng quát.',
        '- RQ2: tính lại đúng hai checkpoint E2 mới cho thấy Transfer Bad-F1 thấp hơn Scratch; còn vấn đề objective, seed, validation và báo cáo không đồng bộ. Chỉ kết luận ở phạm vi run đã kiểm tra.',
        '- RQ3: Hash/URL/Combined Leiden đã có kiểm định và phân bố Bad mô tả. Chưa có graph bổ sung Scratch/Transferred, nên chưa trả lời lợi ích transfer cho graph.', '',
        '## 6. Checklist còn lại', '',
        markdown_table(ablations,['analysis','status','result']), '',
        'Cần đồng bộ báo cáo E2 với checkpoint đang dùng; giải trình/sửa class weighting, seed và checkpoint selection; nhận proposed graphs và config/checkpoint provenance; làm equal-edge-count control và sensitivity theo grid đã chốt; tích hợp kết luận sau khi có đủ bằng chứng. Không coi file kế hoạch ablation là kết quả đã chạy.', '',
        '## 7. Tái lập và output', '',
        'Lệnh đầy đủ trong `evaluation/research/README.md`. Thứ tự: kiểm định E1/baseline đã có → review_e2_handoff.py với commit hiện hành và output e2_handoff_current → review_e3_baselines.py → control_e3_signal_edge_counts.py → integrate_current_handoffs.py. Tách input snapshots, runtime và graph lớn ra khỏi repo. Audit có SHA256, commit, môi trường, nguồn class/helper và các adaptation. Các file trong history là bằng chứng trước đó, không là kết quả hiện hành.', '',
        'Không add/commit/push. Checklist cá nhân tiếp tục được Git ignore. Chưa đánh dấu toàn bộ nhiệm vụ Người 6 hoàn tất.', '']
    (r/'E1_E2_E3_COMPARISON.md').write_text('\n'.join(report),encoding='utf8')
    evidence=dict(snapshot_commit=tree['commit'],dummy_report_commit=branch_tree['commit'],
        e2_audit_sha256=sha256(e2path),e3_audit_sha256=sha256(e3path),dummy_report_sha256=sha256(report_path),
        dummy_checks=dummychecks,graph_metric_mismatches=mismatches.to_dict('records'),
        status='current_handoffs_integrated_remaining_proposed_graphs_and_protocol_issues',
        regenerated_graph_files={str(f):sha256(f) for f in Path(e3['large_output']).glob('*/*.csv')},
        outputs={name:sha256(r/name) for name in ['classification_comparison.csv','graph_comparison.csv','community_metrics.csv','E1_E2_E3_COMPARISON.md']})
    (r/'current_handoff_integration.json').write_text(json.dumps(evidence,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print(json.dumps({'status':evidence['status'],'classification_rows':len(classification),'graphs':len(graph),'communities':len(communities),'dummy_mismatches':sum(not c['matches'] for c in dummychecks),'graph_metric_mismatches':len(mismatches)}))


if __name__=='__main__': main()
