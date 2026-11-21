"""Declarative model loading; deliberately independent of quantum packages."""
from dataclasses import asdict
import json
import hashlib
from copy import deepcopy
from pathlib import Path

from config import (EnergyGroups, Geometry3D, MaterialLayer, MaterialParams,
                    Materials, Source, TransportConfig, validate_config)

PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_path(path):
    p = Path(path)
    return p.resolve() if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def config_dict(cfg):
    return asdict(cfg)


def model_from_dict(data):
    data = dict(data)
    data.pop('metadata', None)
    if 'steps' in data:
        steps = data.pop('steps')
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ValueError('steps must be a positive integer')
    fields = {'geom', 'groups', 'mats', 'source', 'model', 'layers', 'h'}
    if set(data) - fields:
        raise ValueError(f'Unknown configuration fields: {sorted(set(data)-fields)}')
    groups = dict(data['groups'])
    groups['boundaries'] = tuple(groups['boundaries'])
    mats = Materials(**{name: MaterialParams(**{key: tuple(value) for key, value in values.items()})
                        for name, values in data['mats'].items()})
    cfg = TransportConfig(geom=Geometry3D(**data['geom']), groups=EnergyGroups(**groups),
        mats=mats, source=Source(**data['source']), model=data.get('model', 'legacy_duct'),
        layers=tuple(MaterialLayer(**layer) for layer in data.get('layers', [])), h=data.get('h', 1.0))
    validate_config(cfg)
    return cfg


def load_model(path):
    return model_from_dict(load_document(path))


def document_info(value, *, project_root=PROJECT_ROOT):
    """Named preset or optional external JSON, without writing/copying inputs."""
    from presets import get_preset, preset_name
    if isinstance(value,dict):
        doc=deepcopy(value);locator=None;origin='inline';file_hash=None
    else:
        text=str(value); p=Path(text)
        p=p.resolve() if p.is_absolute() else (Path(project_root)/p).resolve()
        if p.is_file():
            raw=p.read_bytes();doc=json.loads(raw.decode('utf-8-sig'))
            locator=str(p);origin='external_json';file_hash=hashlib.sha256(raw).hexdigest()
        else:
            locator=preset_name(text);doc=get_preset(locator);origin='preset';file_hash=None
    if isinstance(doc,dict) and 'preset' in doc:
        if set(doc)!={'preset'}:raise ValueError('A preset alias must not contain duplicate configuration fields')
        locator=preset_name(doc['preset']);doc=get_preset(locator);origin='preset_alias'
    if not isinstance(doc,dict):raise ValueError('Model/suite JSON must be an object')
    canonical=json.dumps(doc,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
    return dict(locator=locator,document=doc,sha256=hashlib.sha256(canonical).hexdigest(),
                origin=origin,file_sha256=file_hash)


def load_document(value, *, project_root=PROJECT_ROOT):
    return document_info(value,project_root=project_root)['document']
