import csv
import time
from pathlib import Path
from decimal import Decimal, ROUND_FLOOR
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from LowerBounds import FSB, RCB, HSB, SSB, ledm, slackngon, udisj, correl
from LowerBounds import Sdodecahedron, S24cell, Sicosidodecahedron, Scuboctahedron


QUICK       = True
METHODS     = ['FSB', 'RCB', 'HSB', 'SSB']
MATRICES    = None
SEEDS       = list(range(30))
TIME_LIMIT  = 3600
CHECK_SSB   = True
OUTPUT      = Path('results_lowerbounds')

INSTANCES = {
    'FSB': ['M10','M20','M30','M40','M50','M100','M200','M240',
            'S5','S10','S20','S30','S35','S38','S40','S43',
            'Dodecahedron','24-cell','Icosidodecahedron','Cuboctahedron',
            'U2','U3','U4','C2','C3','C4'],
    'RCB': ['M5','M7','M8','M10','M11','M15','M20'] +
           ['M'+str(n) for n in range(21,36)] +
           ['S5','S7','S10','S12','S13','S16','S19','S20',
            'Dodecahedron','24-cell','Icosidodecahedron','Cuboctahedron',
            'U2','U3','U4','U5','U6','C2','C3','C4','C5'],
    'HSB': ['S'+str(n) for n in range(5,25)] +
           ['M'+str(n) for n in range(5,31)] +
           ['Dodecahedron','24-cell','Icosidodecahedron','Cuboctahedron',
            'C2','C3','C4','C5'],
    'SSB': ['S'+str(n) for n in range(5,15)] +
           ['M'+str(n) for n in range(5,13)] +
           ['Dodecahedron','24-cell','Icosidodecahedron','Cuboctahedron',
            'C2','C3','C4']}

LONG_RUNS = {('HSB','S24'):7200, ('HSB','Icosidodecahedron'):25200,
             ('HSB','C5'):10800}
for name in ['S12','S13','S14','M10','M11','M12']:
    LONG_RUNS['SSB',name] = None
SINGLE_SEED = set(LONG_RUNS) | {
    ('FSB','24-cell'), ('RCB','S20'), ('RCB','24-cell'),
    ('RCB','Icosidodecahedron'), ('RCB','U5'), ('RCB','U6'), ('RCB','C5'),
    ('HSB','24-cell'), ('SSB','Dodecahedron'), ('SSB','24-cell'),
    ('SSB','Icosidodecahedron')}
SINGLE_SEED |= {('RCB','M'+str(n)) for n in range(21,36)}


def matrix(name):
    if name == 'Dodecahedron': return Sdodecahedron.copy()
    if name == '24-cell': return S24cell.copy()
    if name == 'Icosidodecahedron': return Sicosidodecahedron.copy()
    if name == 'Cuboctahedron': return Scuboctahedron.copy()
    n = int(name[1:])
    if name[0] == 'M': return ledm(n)
    if name[0] == 'S': return slackngon(n)
    if name[0] == 'U': return udisj(n)
    if name[0] == 'C': return correl(n)
    raise ValueError('Unknown matrix: '+name)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    gp.setParam('OutputFlag', 0)
    rows = []
    fields = ['method','matrix','seed','verification','completed','runtime',
              'a','b','ratio','B','upper_bound']
    with (OUTPUT / 'raw_runs.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for method in METHODS:
            names = MATRICES if MATRICES is not None else (['M5'] if QUICK else INSTANCES[method])
            for name in names:
                if method in ['HSB','SSB'] and name.startswith('U'): continue
                X = matrix(name)
                seeds = [0] if QUICK or (MATRICES is None and (method,name) in SINGLE_SEED) else SEEDS
                limit = LONG_RUNS.get((method,name), TIME_LIMIT) if MATRICES is None else TIME_LIMIT
                jobs = [(seed,False) for seed in seeds]
                if method == 'SSB' and CHECK_SSB: jobs.append((0,True))
                for seed,verification in jobs:
                    previous = [row for row in rows if row['method']==method and row['matrix']==name]
                    if verification and not all(row['completed'] for row in previous): continue
                    print('\nRunning',method,name,'seed',seed,'verification',verification)
                    a, b, ratio, B = 0.0, 0.0, 0.0, 0.0
                    upper_bound = min(X.shape)

                    if method == 'FSB':
                        gp.setParam('TimeLimit', GRB.INFINITY if limit is None else limit)
                        for r in range(1,min(X.shape)+2):
                            call_start = time.time()
                            try:
                                U,V,found,runtime = FSB(X,r,seed=seed)
                            except RuntimeError:
                                a += time.time()-call_start
                                continue
                            if found is not None:
                                B = r
                                a += runtime
                            else:
                                upper_bound = r-1
                                b = runtime
                                break
                        runtime = a+b
                        completed = B == upper_bound

                    elif method == 'RCB':
                        gp.setParam('TimeLimit', GRB.INFINITY if limit is None else limit)
                        B = 1
                        for r in range(min(X.shape),0,-1):
                            call_start = time.time()
                            try:
                                U,V,found,runtime = RCB(X,r,seed=seed)
                            except RuntimeError:
                                a += time.time()-call_start
                                continue
                            if found is not None:
                                upper_bound = r
                                a += runtime
                            else:
                                B = r+1
                                b = runtime
                                break
                        runtime = a+b
                        completed = B == upper_bound

                    else:
                        gp.setParam('TimeLimit', GRB.INFINITY)
                        function = HSB if method == 'HSB' else SSB
                        delta = 1e-4 if method == 'HSB' else 1e-2
                        eta = 1e-6 if verification else 1e-4
                        L,u,v,ratio,runtime,B,upper_bound = function(
                            X,tol=1e-6,delta=delta,mip_gap=eta,seed=seed,time_limit=limit)
                        completed = len(u)>0 and len(v)>0

                    row = dict(method=method,matrix=name,seed=seed,verification=verification,
                               completed=completed,runtime=runtime,a=a,b=b,ratio=ratio,
                               B=B,upper_bound=upper_bound)
                    rows.append(row)
                    writer.writerow(row)
                    file.flush()
                    gp.setParam('TimeLimit', GRB.INFINITY)
                    print('Completed:',completed,'  Bounds:',B,upper_bound,'  Time:',runtime)

    for method in METHODS:
        selected = [row for row in rows if row['method']==method]
        with (OUTPUT / ('table_'+method+'.tex')).open('w') as file:
            file.write('\\begin{table}[ht]\n\\centering\n')
            file.write('\\begin{tabular}{lrr}\n\\hline\nMatrix & Time [s] & '+method+' \\\\\n\\hline\n')
            for name in dict.fromkeys(row['matrix'] for row in selected):
                data = [row for row in selected if row['matrix']==name]
                runs = [row for row in data if not row['verification']]
                completed = all(row['completed'] for row in runs)
                lower = max(row['B'] for row in data)
                upper = min(row['upper_bound'] for row in data)
                runtime = np.mean([row['runtime'] for row in runs])
                if method in ['FSB','RCB']:
                    a = np.mean([row['a'] for row in runs])
                    b = np.mean([row['b'] for row in runs])
                    elapsed = f'{a:.3f} + {b:.3f}'
                    value = str(int(lower)) if completed else '$\\geq '+str(int(lower))+'^{\\dagger}$'
                else:
                    digits = 4 if method == 'HSB' else 3
                    elapsed = f'{runtime:.3f}'
                    if completed:
                        checked = [row for row in data if row['verification'] and row['completed']]
                        ratio = checked[-1]['ratio'] if checked else max(row['ratio'] for row in runs)
                        value = f'{ratio:.{digits}f}'
                        verified = (lower <= upper+1e-6 and
                                    f'{lower:.{digits}f}' == f'{upper:.{digits}f}' == value)
                        if not verified: value += '$^{*}$'
                    else:
                        rounded = Decimal.from_float(float(lower)).quantize(Decimal(10)**-digits, rounding=ROUND_FLOOR)
                        value = '$\\geq '+str(rounded)+'^{\\dagger}$'
                file.write(name+' & '+elapsed+' & '+value+' \\\\\n')
            file.write('\\hline\n\\end{tabular}\n')
            file.write('\\caption{'+method+' computations. Times are means over the selected seeds, '
                       'excluding SSB verification runs. A dagger marks an incomplete result; '
                       'its displayed value is a lower bound rounded down. '
                       'An asterisk marks an estimate whose rounding is not confirmed by the enclosure. '
                       'For complete FSB/RCB runs, $a+b$ separates the preceding checks from the final '
                       'infeasibility proof.}\n\\end{table}\n')
    return rows


if __name__ == '__main__':
    main()
