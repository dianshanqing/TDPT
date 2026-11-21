"""Truth-independent fixed-depth binomial likelihood with disconnected sets.

All singularity intervals in theta are searched. On each interval the negative
log likelihood is convex, so derivative bisection finds its unique minimum.
The asymptotic chi-square set is diagnostic; finite-sample coverage is unproved.
"""
from __future__ import annotations
import math
import numbers
import time
import numpy as np
from scipy.stats import chi2


def integer(value, name):
    if isinstance(value, bool): raise ValueError(f'{name} must be an integer')
    if isinstance(value, numbers.Integral): return int(value)
    if isinstance(value,str) and value.strip().lstrip('+-').isdigit(): return int(value)
    if isinstance(value,numbers.Real) and math.isfinite(value) and float(value).is_integer() and abs(value)<=2**53:
        return int(value)
    raise ValueError(f'{name} must be an exact integer')


def validate_rows(rows):
    result=[]
    for row in rows:
        r={key:integer(row[key],key) for key in ('m','shots','success_count')}
        if r['m']<0 or r['shots']<=0 or not 0<=r['success_count']<=r['shots']:
            raise ValueError('require m>=0, shots>0 and 0<=success_count<=shots')
        result.append(r)
    if not result: raise ValueError('no integer count rows supplied')
    return result


def _functions(rows):
    mm=np.asarray([2*r['m']+1 for r in rows],dtype=np.int64)
    hh=np.asarray([r['success_count'] for r in rows],dtype=float)
    ff=np.asarray([r['shots']-r['success_count'] for r in rows],dtype=float)
    def nll(theta):
        theta=np.asarray(theta,dtype=float);out=np.zeros_like(theta)
        with np.errstate(divide='ignore',invalid='ignore'):
            for m,h,f in zip(mm,hh,ff):
                if h:out-=2*h*np.log(np.abs(np.sin(m*theta)))
                if f:out-=2*f*np.log(np.abs(np.cos(m*theta)))
        return out
    def derivative(theta):
        out=np.zeros_like(theta)
        with np.errstate(divide='ignore',invalid='ignore'):
            for m,h,f in zip(mm,hh,ff):
                tangent=np.tan(m*theta)
                if h:out-=2*m*h/tangent
                if f:out+=2*m*f*tangent
        return out
    return mm,hh,ff,nll,derivative


def fit_counts(rows, *, p_min=0.,p_max=1.,confidence=.95,prior_evidence=None,
               experiment_type='unknown_probability',max_intervals=1000000):
    """Fit only count data and independently declared domain; no truth argument."""
    start=time.monotonic();rows=validate_rows(rows)
    if experiment_type not in ('unknown_probability','known_target_calibration'):
        raise ValueError('invalid experiment_type')
    if not all(math.isfinite(x) for x in (p_min,p_max,confidence)) or not 0<=p_min<p_max<=1 or not 0<confidence<1:
        raise ValueError('require finite 0<=p_min<p_max<=1 and 0<confidence<1')
    if (p_min,p_max)!=(0.,1.) and not str(prior_evidence or '').strip():
        raise ValueError('restricted search requires explicit prior_evidence; target truth is not an independent prior')
    # Guard allocation before creating arrays; do not silently switch to a grid.
    bound=sum(2*r['m']+2 for r in rows)+1
    if bound>integer(max_intervals,'max_intervals'):
        raise ValueError(f'likelihood search budget exceeded: pole bound {bound}>{max_intervals}')
    mm,hh,ff,nll,derivative=_functions(rows)
    low=math.asin(math.sqrt(p_min));high=math.asin(math.sqrt(p_max))
    poles=[np.array([low,high])]
    for m,h,f in zip(mm,hh,ff):
        j=np.arange(int(m)+1)
        take=((j%2==0)&(h>0))|((j%2==1)&(f>0))
        x=(j[take]/float(m))*(math.pi/2)
        poles.append(x[(x>low)&(x<high)])
    edges=np.unique(np.concatenate(poles));left=edges[:-1].copy();right=edges[1:].copy()
    for _ in range(64):
        mid=(left+right)/2;d=derivative(mid)
        right=np.where(d>=0,mid,right);left=np.where(d<0,mid,left)
    opt=(left+right)/2;values=nll(opt)
    for idx,endpoint in ((0,low),(-1,high)):
        value=float(nll(endpoint))
        if value<=values[idx]:opt[idx]=endpoint;values[idx]=value
    best_index=int(np.argmin(values));best=float(values[best_index]);cut=float(chi2.ppf(confidence,1));threshold=best+cut/2
    accepted=np.where(values<=threshold)[0]
    lo=edges[accepted].copy();hi=opt[accepted].copy()
    for _ in range(64):
        mid=(lo+hi)/2;outside=nll(mid)>threshold
        lo=np.where(outside,mid,lo);hi=np.where(outside,hi,mid)
    lower=(lo+hi)/2;lo=opt[accepted].copy();hi=edges[accepted+1].copy()
    for _ in range(64):
        mid=(lo+hi)/2;inside=nll(mid)<=threshold
        lo=np.where(inside,mid,lo);hi=np.where(inside,hi,mid)
    upper=(lo+hi)/2
    components=[{'p_low':float(np.sin(a)**2),'p_high':float(np.sin(b)**2),'mode_p':float(np.sin(opt[i])**2),'delta_nll':float(values[i]-best)} for i,a,b in zip(accepted,lower,upper)]
    modes=[{'p':float(np.sin(opt[i])**2),'negative_log_likelihood':float(values[i]),'delta_nll':float(values[i]-best)} for i in np.argsort(values)[:30]]
    warnings=['asymptotic_coverage_unverified']
    if len(components)>1:warnings.append('multi_peak_likelihood')
    if sum(r['success_count'] for r in rows)==0:warnings.append('zero_success_all_powers')
    if sum(r['success_count'] for r in rows)==sum(r['shots'] for r in rows):warnings.append('all_success_all_powers')
    n=sum(r['shots'] for r in rows);queries=sum(r['shots']*r['m'] for r in rows)
    return {'fit_status':'ambiguous_multimodal' if len(components)>1 else 'conditional_fit',
        'method':'all_pole_intervals_convex_derivative_bisection','p_mle':float(np.sin(opt[best_index])**2),
        'negative_log_likelihood':best,'p_min':float(p_min),'p_max':float(p_max),'confidence':float(confidence),
        'confidence_interval':([components[0]['p_low'],components[0]['p_high']] if len(components)==1 else None),
        'likelihood_set_components':components,'likelihood_set_hull':[components[0]['p_low'],components[-1]['p_high']],
        'likelihood_cutoff':cut,'number_of_accepted_modes':len(components),'interval_count':len(opt),
        'coverage_verified':False,'estimate_success':False,'fit_warning':warnings,'local_minima':modes,
        'rows_used':rows,'powers_used':sorted({r['m'] for r in rows}),'total_shots':n,
        'grover_iterations_total':queries,'total_oracle_calls':2*queries+n,
        'state_preparation_equivalent_calls':2*queries+n,'max_A_equivalent_calls_per_shot':max(2*r['m']+1 for r in rows),
        'experiment_type':experiment_type,'prior_evidence':prior_evidence,
        'design_independence_verified':False,'sampling_independence_verified':False,
        'low_p_suggestion':None,'uses_truth_for_fitting':False,'elapsed_seconds':time.monotonic()-start,
        'note':'Disconnected asymptotic likelihood set; hull is not an interval of accepted points. No finite-sample coverage or unknown-probability success is certified.'}


def evaluate_truth(fit, truth):
    """Post hoc evaluation only. Never called by fit_counts."""
    if truth is None:return None
    truth=float(truth)
    if not math.isfinite(truth) or not 0<=truth<=1:raise ValueError('evaluation truth must be finite in [0,1]')
    return {'exact_p':truth,'relative_deviation':None if truth==0 else (fit['p_mle']-truth)/truth,
            'in_likelihood_set':any(x['p_low']<=truth<=x['p_high'] for x in fit['likelihood_set_components']),
            'evaluation_only':True}
