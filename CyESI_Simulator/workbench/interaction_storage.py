"""Separate two-dimensional study records; original CFD archives stay read-only."""
from pathlib import Path
import hashlib
import json
import numpy as np
from .compute import digest
from ..scene3d.data import _clean
from ..particle.interactions import InteractionResult


def source_signature(session):
    data=session.data
    h=hashlib.sha256()
    for key in sorted(data.fields):
        array=np.ascontiguousarray(data.fields[key]);h.update(key.encode());h.update(str(array.shape).encode());h.update(array.tobytes())
    return dict(config=digest(data.config.to_dict()),fields_sha256=h.hexdigest(),duration_s=session.duration_s,startup=session.startup)


def save_study(session,directory):
    if not session.has_interactions():raise ValueError('No current axisymmetric interaction study to save')
    path=Path(directory);path.mkdir(parents=True,exist_ok=False)
    result=session.interaction_result;meta,arrays=result.to_storage();model=session.process();ids=result.event_ids
    arrays.update(source_injection_s=model.records['injection_s'][ids],source_template_id=model.records['template_id'][ids])
    np.savez_compressed(path/'interactions.npz',**arrays)
    meta.update(format='cyesi-axisymmetric-interactions',version=1,source=session.data.source,
        source_signature=source_signature(session),archive_sha256=hashlib.sha256((path/'interactions.npz').read_bytes()).hexdigest(),
        conditional_acquisition=session.interaction_acquisition())
    (path/'interactions.json').write_text(json.dumps(_clean(meta),ensure_ascii=False,allow_nan=False,indent=2),encoding='utf-8')
    return path


def load_study(session,directory):
    if session.data is None or session.pending:raise ValueError('Load the solved two-dimensional source first')
    path=Path(directory);meta=json.loads((path/'interactions.json').read_text(encoding='utf-8'))
    if meta.get('format')!='cyesi-axisymmetric-interactions' or meta.get('version')!=1:raise ValueError('Not a supported two-dimensional study')
    if meta.get('source_signature')!=source_signature(session):raise ValueError('Source fields, concentration or continuous observation settings differ; study cannot be reused')
    if hashlib.sha256((path/'interactions.npz').read_bytes()).hexdigest()!=meta['archive_sha256']:raise ValueError('Interaction archive integrity check failed')
    with np.load(path/'interactions.npz',allow_pickle=False) as a:arrays={k:a[k].copy() for k in a.files}
    result=InteractionResult.from_storage(meta,arrays);model=session.process();ids=result.event_ids
    if (np.any(ids<0) or np.any(ids>=len(model.records['event_id'])) or len(np.unique(ids))!=len(ids)
            or result.positions_m.shape!=(len(result.time_s),len(ids),3)
            or result.states.shape!=(len(result.time_s),len(ids)) or np.any(np.diff(result.time_s)<=0)
            or not np.isin(result.states,range(7)).all()):raise ValueError('Invalid study identity, time or trajectory arrays')
    for name in ('injection_s','template_id'):
        if not np.array_equal(model.records[name][ids],arrays['source_'+name]):raise ValueError('Continuous feed identity differs')
    from ..particle.axisymmetric_interactions import AxisymmetricScene
    scene=AxisymmetricScene(session.data,model)
    session.accept_interactions(scene,result,session.interaction_source_key())
    return result
