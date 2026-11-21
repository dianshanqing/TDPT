"""Statistical helper functions for uncertainty and sigma reports."""


from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

def binom_se(p: float, n: int) -> float:

    n = int(n)
    if n <= 0:
        return float("nan")
    p = float(p)
    v = max(0.0, p * (1.0 - p))
    return math.sqrt(v / n)

@dataclass(frozen=True)
class SigmaReport:
    ref: float
    ref_se: float
    exp: float
    exp_se: Optional[float]
    abs_diff: float
    rel_diff: float
    z_sigma: float
    pass_3sigma: bool

def sigma_report(exp: float, ref: float, ref_se: float, exp_se: Optional[float] = None, eps: float = 1e-15) -> SigmaReport:

    exp = float(exp); ref = float(ref); ref_se = float(ref_se)
    abs_diff = abs(exp - ref)
    if not all(math.isfinite(v) for v in (exp, ref, ref_se)) or ref_se < 0:
        raise ValueError('values must be finite and standard errors nonnegative')
    if exp_se is not None and (not math.isfinite(float(exp_se)) or float(exp_se) < 0):
        raise ValueError('exp_se must be finite and nonnegative')
    rel_diff = abs_diff / abs(ref) if ref else (0.0 if abs_diff == 0 else math.inf)
    # Independent statistical standard errors combine in quadrature. Numerical
    # MPS bias is not a sampling standard error and must not be supplied here.
    denom_se = math.hypot(ref_se, 0.0 if exp_se is None else float(exp_se))
    z = abs_diff / denom_se if denom_se else (0.0 if abs_diff == 0 else math.inf)
    return SigmaReport(
        ref=ref,
        ref_se=ref_se,
        exp=exp,
        exp_se=None if exp_se is None else float(exp_se),
        abs_diff=abs_diff,
        rel_diff=rel_diff,
        z_sigma=z,
        pass_3sigma=(z < 3.0),
    )


def accuracy_from_interval(interval, reference, tolerance, *, mode='absolute'):
    """Precision acceptance is separate from a zero-bias significance/Z test."""
    lo,hi=map(float,interval);reference=float(reference);tolerance=float(tolerance)
    if not all(math.isfinite(x) for x in (lo,hi,reference,tolerance)):
        raise ValueError('Non-finite precision inputs')
    if not 0<=lo<=hi<=1 or not 0<=reference<=1 or tolerance<=0:
        raise ValueError('Invalid probability interval/reference/tolerance')
    if mode not in ('absolute','relative'):raise ValueError('mode must be absolute or relative')
    if mode=='relative' and reference==0:
        return {'verdict':'undefined','reason':'relative error at zero reference is undefined','mode':mode,'tolerance':tolerance}
    eps=tolerance if mode=='absolute' else tolerance*reference
    allowed=[max(0.,reference-eps),min(1.,reference+eps)]
    verdict='pass' if lo>=allowed[0] and hi<=allowed[1] else ('fail' if hi<allowed[0] or lo>allowed[1] else 'insufficient_evidence')
    return {'verdict':verdict,'interval':[lo,hi],'reference':reference,'mode':mode,
            'tolerance':tolerance,'absolute_tolerance':eps,'allowed_interval':allowed}


def sampling_intervals(counts, *, confidence=.95, family_size=None):
    """Simultaneous marginal intervals for a prespecified family; no bias bound."""
    from scipy.stats import beta
    if not isinstance(counts,dict) or not counts:raise ValueError('Nonempty raw counts required')
    if any(type(v) is not int or v<0 for v in counts.values()):raise ValueError('Raw integer counts required')
    n=sum(counts.values())
    if n<=0:raise ValueError('Positive sample size required')
    m=len(counts) if family_size is None else family_size
    if type(m) is not int or m<len(counts) or type(confidence) not in (int,float) or not math.isfinite(confidence) or not 0<confidence<1:raise ValueError('Invalid simultaneous confidence scope')
    alpha=(1-confidence)/m
    channels={}
    for channel,k in counts.items():
        interval=[0. if k==0 else float(beta.ppf(alpha/2,k,n-k+1)),
                  1. if k==n else float(beta.ppf(1-alpha/2,k+1,n-k))]
        channels[channel]={'count':k,'samples':n,'probability':k/n,'confidence_interval':interval}
    return {'channels':channels,'samples':n,'confidence':confidence,'family_size':m,
            'per_channel_confidence':1-alpha,'interval_method':'Bonferroni two-sided Clopper-Pearson',
            'family_scope':'prespecified family; larger than returned channel count may include other planned runs',
            'precision_certified':False,
            'scope':'sampling uncertainty conditional on a fixed output distribution; excludes MPS truncation, finite-bond and model bias'}


def precision_report(counts, reference, *, tolerance, mode='absolute', confidence=.95, family_size=None):
    """Apply a caller-supplied precision criterion to the shared sampling intervals."""
    if set(counts)!=set(reference) or not counts:raise ValueError('Counts/reference channels must agree')
    report=sampling_intervals(counts,confidence=confidence,family_size=family_size)
    results={channel:accuracy_from_interval(row['confidence_interval'],reference[channel],tolerance,mode=mode)
             for channel,row in report['channels'].items()}
    verdicts=[v['verdict'] for v in results.values()]
    verdict='fail' if 'fail' in verdicts else ('pass' if all(v=='pass' for v in verdicts) else ('undefined' if 'undefined' in verdicts else 'insufficient_evidence'))
    return {'verdict':verdict,'channels':results,'samples':report['samples'],'confidence':confidence,'family_size':report['family_size'],
            'interval_method':'Bonferroni two-sided Clopper-Pearson','mode':mode,'tolerance':tolerance,
            'criterion_status':'caller_supplied; not a declared manuscript acceptance standard',
            'scope':'sampling uncertainty conditional on a fixed output distribution; does not remove simulator bias'}
