"""Hand-derived transport and decimal checks for compact shielding."""
from decimal import Decimal, localcontext
import importlib.util
import pytest


def api():
    assert importlib.util.find_spec('compact_classical') is not None
    import compact_classical
    return compact_classical


def test_refinement_preserves_physical_problem():
    c=api().build_compact(8,23,.003,4)
    assert c['h']==.125
    assert (c['geom']['nx'],c['geom']['ny'],c['geom']['nz'])==(65,33,33)
    assert (c['layers'][0]['x0'],c['layers'][0]['x1'])==(16,47)
    assert c['geom']['detector_x']*.125==7
    assert c['source']['x0']*.125==1
    assert c['geom']['roi_y0']*.125==1.5
    assert c['geom']['roi_y1']*.125==2.5


def test_pure_absorption_decimal_from_physics_and_fixed_coefficients():
    mod=api();c=mod.build_compact(4,24,0)
    r=mod.decimal_reference(c,9,digits=80,coefficient_mode='physical')
    with localcontext() as ctx:
        ctx.prec=80
        expected=Decimal(-24).exp()
        assert abs(Decimal(r['p_D'])/expected-1)<Decimal('1e-70')
    assert r['p_S']=='0'
    fixed=mod.decimal_reference(c,9,digits=80,coefficient_mode='fixed_float')
    assert Decimal(fixed['p_D'])!=Decimal(r['p_D'])


def test_decimal_nontrivial_scatter_and_terminal_priority():
    mod=api();c=mod.build_compact(4,14,.003)
    c['geom'].update(nx=5,ny=3,nz=3,slab_x0=2,slab_x1=2,detector_x=3,roi_y0=1,roi_y1=1,roi_z0=1,roi_z1=1)
    c['source'].update(x0=2,y0=1,z0=1)
    c['layers']=[dict(x0=2,x1=2,material='tungsten')]
    c['mats']['tungsten'].update(p_int=[.5],p_abs=[.5])
    r=mod.decimal_reference(c,3,digits=80,coefficient_mode='fixed_float')
    with localcontext() as ctx:
        ctx.prec=80
        assert abs(Decimal(r['p_D'])-Decimal(13)/24)<Decimal('1e-70')
        assert abs(Decimal(r['p_B'])-Decimal(5)/24)<Decimal('1e-70')
    c['geom']['detector_x']=4
    assert Decimal(mod.decimal_reference(c,4,digits=50,coefficient_mode='fixed_float')['p_D'])==0


def test_decimal_does_not_lose_uncolllided_branch_when_float_p_rounds_one():
    mod=api();c=mod.build_compact(4,200,0)
    assert c['mats']['tungsten']['p_int']==[1.0]
    assert Decimal(mod.decimal_reference(c,9,digits=80)['p_D'])>0
    assert Decimal(mod.decimal_reference(c,9,digits=80,coefficient_mode='fixed_float')['p_D'])==0


@pytest.mark.parametrize('steps',[-1,True,2.5])
def test_decimal_rejects_invalid_steps(steps):
    with pytest.raises(ValueError):api().decimal_reference({},steps)
