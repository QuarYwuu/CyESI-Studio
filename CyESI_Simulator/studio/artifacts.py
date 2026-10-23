"""Read-only catalog of old and new scientific study artifacts."""
from hashlib import sha256
from pathlib import Path
import json


def study_catalog(service):
    outputs=Path(__file__).resolve().parents[2]/'outputs'
    roots=[]
    for parent in (outputs/'size_studies',outputs/'length_studies',service.studio_root/'studies'):
        if parent.exists():roots.extend(x for x in parent.iterdir() if x.is_dir())
    rows=[]
    for root in roots:
        files=sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in ('.png','.pdf','.html','.json','.csv'))
        if not files:continue
        identifier=sha256(str(root.resolve()).encode()).hexdigest()[:16]
        kind='size' if (root/'size_summary.json').exists() else 'length' if (root/'length_study_report.html').exists() else 'scan'
        rows.append(dict(id=identifier,name=root.name,kind=kind,path=str(root.resolve()),
            modified=max(p.stat().st_mtime for p in files),
            files=[dict(name=p.name,url=f'/research/{identifier}/{p.name}',type=p.suffix[1:]) for p in files]))
    rows.sort(key=lambda x:x['modified'],reverse=True)
    service.study_records={r['id']:r for r in rows}
    return rows


def archived_study(service,kind):
    record=next((row for row in study_catalog(service) if row['kind']==kind),None)
    if record is None:return None
    root=Path(record['path'])
    summary=next((p for p in (root/'studio_study.json',root/'size_summary.json',root/'length_study.json') if p.is_file()),None)
    payload={}
    if summary is not None:
        try:payload=json.loads(summary.read_text(encoding='utf-8'))
        except (ValueError,OSError):pass
    return dict(payload,kind=kind,computed=True,archived=True,record=record,figures=[p for p in record['files'] if p['type']=='png'],
        notes=[' Saved parameter study ;  Parameters and provenance in the report ,  Study-specific result .'])
