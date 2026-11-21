"""Independent hand-solvable cases catch order, branching and layer errors."""
import copy
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def fixture_cfg():
    return dict(model='layered_slab', h=1.0,
        geom=dict(nx=5, ny=3, nz=3, slab_x0=1, slab_x1=2,
                  duct_y0=1, duct_y1=1, duct_z0=1, duct_z1=1,
                  detector_x=3, roi_y0=1, roi_y1=1, roi_z0=1, roi_z1=1),
        groups=dict(n_groups=1, boundaries=[1,0]),
        source=dict(x0=1,y0=1,z0=1,dir0=0,g0=0),
        layers=[dict(x0=1,x1=1,material='tungsten'),dict(x0=2,x1=2,material='air')],
        mats={name:dict(p_int=[pi],p_abs=[1],q_down=[0])
              for name,pi in [('concrete',0),('tungsten',.2),('air',.3)]})


def solver():
    assert importlib.util.find_spec('reference_solver') is not None, 'independent reference module is required'
    from reference_solver import reference
    return reference


def test_absorption_order_and_next_kernel_detection():
    result=solver()(fixture_cfg(),3,retain_joint=True)
    assert result['rows'][1]['probabilities'] == pytest.approx([.56,0,.44,0])
    assert result['rows'][2]['probabilities'] == pytest.approx([0,.56,.44,0])
    assert result['joint'][(1,3,1,1,0,0)] == pytest.approx(.56)
    assert result['diagnostics']['earliest_detection_step'] == 3
    assert result['diagnostics']['p_D_uncollided'] == pytest.approx(.56)
    swapped=fixture_cfg()
    swapped['layers'][0]['material']='air'
    swapped['layers'][1]['material']='tungsten'
    assert solver()(swapped,3)['rows'][-1]['probabilities'][1] == pytest.approx(.56)


def test_scattered_detection_leakage_and_collision_moment():
    cfg=fixture_cfg();cfg['model']='uniform_slab';cfg['source']['x0']=2
    cfg['geom']['slab_x0']=2
    cfg['layers']=[dict(x0=2,x1=2,material='tungsten')]
    cfg['mats']['tungsten']=dict(p_int=[.6],p_abs=[.25],q_down=[0])
    result=solver()(cfg,3)
    assert result['rows'][-1]['probabilities'] == pytest.approx([0,.475,.15,.375])
    d=result['diagnostics']
    assert d['p_D_uncollided'] == pytest.approx(.4)
    assert d['p_D_scattered'] == pytest.approx(.075)
    assert d['detected_mean_scatter_count'] == pytest.approx(.075/.475)
    assert d['leakage']['side'] == pytest.approx(.3)
    assert d['leakage']['front'] == pytest.approx(.075)


def test_boundary_priority_and_zero_step():
    cfg=fixture_cfg();cfg['geom']['detector_x']=4
    r=solver()(cfg,4)
    assert r['rows'][-1]['probabilities'] == pytest.approx([0,0,.44,.56])
    assert r['diagnostics']['earliest_detection_step'] is None
    z=solver()(cfg,0,retain_joint=True)
    assert z['joint'] == {(0,1,1,1,0,0):1.0}
    with pytest.raises(ValueError):solver()(cfg,-1)


def test_reference_does_not_require_quantum_or_production_geometry():
    import subprocess
    code="import sys; from reference_solver import reference; assert 'geometry' not in sys.modules; assert 'qiskit' not in sys.modules; assert 'unitary_circuit' not in sys.modules"
    assert subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=False).returncode == 0


def test_screen_writes_finite_case_identity_and_refuses_overwrite(tmp_path):
    assert importlib.util.find_spec('screening') is not None, 'screening module is required'
    from screening import run_screen
    p=tmp_path/'case.json';p.write_text(json.dumps(fixture_cfg()),encoding='utf-8')
    suite=tmp_path/'suite.json';suite.write_text(json.dumps(dict(version=1,
        candidates=[dict(id='analytic',config=str(p),role='negative_control',reason='hand computed')],
        max_steps=8,tail_relative_tolerance=.01)),encoding='utf-8')
    out=tmp_path/'out';out.mkdir();(out/'run.log').write_text('main log')
    result=run_screen(suite,out,cpu_budget=1)
    assert result['candidates'][0]['p_D'] == pytest.approx(.56)
    saved=json.loads((out/'classical'/'analytic.json').read_text(encoding='utf-8'))
    assert saved['identity']['config_sha256']
    assert saved['rows'][-1]['p_S'] == 0
    with pytest.raises(FileExistsError):run_screen(suite,out,cpu_budget=1)


@pytest.mark.parametrize('budget',[1.5,True])
def test_screen_rejects_noninteger_cpu_budget(tmp_path,budget):
    from screening import run_screen
    p=tmp_path/'case.json';p.write_text(json.dumps(fixture_cfg()),encoding='utf-8')
    suite=tmp_path/'suite.json';suite.write_text(json.dumps(dict(candidates=[dict(id='case',config=str(p))],max_steps=8)),encoding='utf-8')
    with pytest.raises(ValueError):run_screen(suite,tmp_path/'out',cpu_budget=budget)


@pytest.mark.parametrize('steps',[2.9,True,0,-1])
def test_screen_rejects_invalid_max_steps(tmp_path,steps):
    from screening import run_screen
    p=tmp_path/'case.json';p.write_text(json.dumps(fixture_cfg()),encoding='utf-8')
    suite=tmp_path/'suite.json';suite.write_text(json.dumps(dict(candidates=[dict(id='case',config=str(p))],max_steps=steps)),encoding='utf-8')
    with pytest.raises(ValueError):run_screen(suite,tmp_path/'out',cpu_budget=1)


def test_screen_rejects_unrepresentable_observation_window(tmp_path):
    from screening import run_screen
    coarse=fixture_cfg();fine=copy.deepcopy(coarse);fine['h']=.7
    paths=[]
    for name,cfg in [('coarse',coarse),('fine',fine)]:
        p=tmp_path/(name+'.json');p.write_text(json.dumps(cfg),encoding='utf-8');paths.append(str(p))
    suite=tmp_path/'suite.json';suite.write_text(json.dumps(dict(candidates=[
        dict(id='coarse',config=paths[0]),dict(id='fine',config=paths[1],match_observation_to='coarse')],max_steps=8)),encoding='utf-8')
    with pytest.raises(ValueError):run_screen(suite,tmp_path/'out',cpu_budget=1)
