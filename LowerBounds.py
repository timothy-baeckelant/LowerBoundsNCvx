import numpy as np
import time
import itertools
from scipy.linalg import circulant
import gurobipy as gp
from gurobipy import GRB
from scipy.io import whosmat, loadmat
from pathlib import Path

PATH = Path(__file__).resolve().parent / "SlackMat.mat"

variables_info = whosmat(PATH)
data           = loadmat(PATH)

variables = {}
for variable_info in variables_info:
    name = variable_info[0]
    variables[name] = data[name]

Sdodecahedron      = variables['Sdodecahedron']
Scuboctahedron     = variables['Scuboctahedron']
Sicosidodecahedron = variables['Sicosidodecahedron']
S24cell            = variables['S24cell']

matrices = [Sdodecahedron, Scuboctahedron, Sicosidodecahedron, S24cell]


def udisj(n):
    V = np.array(list(itertools.product([0,1], repeat=n)))
    M = V @ V.T
    return np.where(M==0, 1, np.where(M==1, 0, -1))

def ledm(n):
    A = np.arange(0,n,1)*np.ones((n,n))
    X = (A-A.T)**2
    return X

def slackngon(n):
    X = circulant(np.cos(np.pi/n)-np.cos(np.pi/n + 2*np.pi*np.arange(0,n,1)/n))
    X[X < 1e-6] = 0.0
    return X

def correl(n):
    M = np.array(list(itertools.product([0, 1], repeat=n)), dtype=np.float64)
    X = (1-M@M.T)**2
    return X

def FSB(X, r, display = False):
    X    = X.copy()
    m, n = X.shape

    for i in range(m):
        for j in range(n):
            if   X[i,j] >  1e-6: X[i,j] =  1.0
            elif X[i,j] < -1e-6: X[i,j] = -1.0
            else:                X[i,j] =  0.0

    udisj = np.any(X < 0)

    start_time = time.time()

    model = gp.Model()
    if not display: model.Params.OutputFlag = 0
    model.Params.NonConvex  = 2

    U = model.addMVar((m, r), vtype=GRB.BINARY)
    V = model.addMVar((r, n), vtype=GRB.BINARY)

    P = (X >  0).astype(int)
    Z = (X == 0).astype(int)

    for k in range(r):
        model.addConstr(U[:, k].sum() == 1)
        model.addConstr(V[k, :].sum() == 1)

    for i in range(m):
        model.addConstr(U[i, :].sum() <= 1)

    for j in range(n):
        model.addConstr(V[:, j].sum() <= 1)

    model.addConstr(U @ V <= P)

    for k1 in range(r):
        for k2 in range(k1 + 1, r):
            somme = 0
            if udisj:
                for i in range(m):
                    for j in range(n):
                        if Z[i, j] == 1:
                            somme += U[i, k1] * V[k2, j] + U[i, k2] * V[k1, j]
                model.addConstr(somme >= 1)
            else:
                for i in range(m):
                    for j in range(n):
                        if P[i, j] == 1:
                            somme += U[i, k1] * V[k2, j] + U[i, k2] * V[k1, j]
                model.addConstr(somme <= 1)

    model.optimize()
    end_time = time.time()
    runtime  = end_time - start_time

    if model.status==2:
        print(f"OPTIMAL FOUND - ({m}x{n}), r={r} - time: {runtime} seconds")
        return U.X, V.X, r, runtime
    if model.status==3:
        print(f"OPTIMAL NOT FOUND - ({m}x{n}), r={r} - time: {runtime} seconds")
        print(f"No fooling set of size {r}.")
    return [], [], None, runtime

    print(f"Optimization ended with status {model.status} - ({m}x{n}), r={r} - time: {runtime} seconds")
    return [], [], None, runtime
        
def RCB(X, r, display = False):
    X     = X.copy()
    m, n  = X.shape
    for i in range(m):
        for j in range(n):
            if   X[i,j] >  1e-6: X[i,j] =  1.0
            elif X[i,j] < -1e-6: X[i,j] = -1.0
            else:                X[i,j] =  0.0
    
    udisj = np.any(X < 0)

    start_time = time.time()
    model      = gp.Model()

    if not display: model.Params.OutputFlag = 0
    model.Params.NonConvex  = 2
    #model.Params.MIPFocus   = 1
    #model.Params.Heuristics = 0.5
    U = model.addMVar((m, r), vtype=GRB.BINARY)
    V = model.addMVar((r, n), vtype=GRB.BINARY)
    Z = model.addMVar((m ,n), vtype=GRB.BINARY)
 
    model.addConstr((U @ V)/r <= Z)
    model.addConstr(Z <= U @ V)

    if udisj:
        P = (X  > 0).astype(int)  # Positive entries (must be covered)
        A = (X != 0).astype(int)  # Allowed entries (can be covered)
        model.addConstr(P <= Z)
        model.addConstr(Z <= A)
    else:
        model.addConstr(Z == X)
    
    model.optimize()
    end_time = time.time()
    runtime  = end_time - start_time

    if model.status==2:
        print(f"OPTIMAL FOUND - ({m}x{n}), r={r} - time: {runtime} seconds")
        return U.X, V.X, r, runtime
    if model.status==3:
        print(f"OPTIMAL NOT FOUND - ({m}x{n}), r={r} - time: {runtime} seconds")
        print(f"No rectangle covering of size {r}.")
    return [], [], None, end_time - start_time

    print(f"Optimization ended with status {model.status} - ({m}x{n}), r={r} - time: {runtime} seconds")
    return [], [], None, runtime
    
        
def HSB(X, display = False, tol = 1e-12, delta = None, steps = False):
    m, n    = X.shape
    X       = X/np.abs(X).max()

    start_time = time.time()
    modelLP = gp.Model()
    if not display: modelLP.Params.OutputFlag = 0
    modelLP.Params.Method = 1

    maskX_0         = X <= 1e-12
    L_lb            = np.full(X.shape, -GRB.INFINITY, dtype=float)
    L_lb[maskX_0]   = 0.0

    L_ub            = np.ones(X.shape, dtype=float)
    L_ub[maskX_0]   = 0.0
    Lvar            = modelLP.addMVar((m, n), lb=L_lb, ub=L_ub)

    modelLP.setObjective((Lvar*X).sum(),GRB.MAXIMIZE)    

    modelNCVX = gp.Model()
    if not display: modelNCVX.Params.OutputFlag = 0
    modelNCVX.Params.NonConvex = 2

    u_var     = modelNCVX.addMVar(m,vtype=GRB.BINARY)
    v_var     = modelNCVX.addMVar(n,vtype=GRB.BINARY)
    idx_X_0   = np.argwhere(maskX_0)

    for i, j in idx_X_0  : modelNCVX.addConstr(u_var[i] + v_var[j] <= 1)

    nbrect  = 0
    L       = L_ub.copy()
    
    u       = np.ones(m)
    v       = np.ones(n)

    HSBound = 0
    if delta is not None: print(f"HSB - delta = {delta}")
    while True:
        early_stop = False

        if delta is not None:
            modelNCVX.Params.BestObjStop = 1.0 + delta
            modelNCVX.Params.MIPFocus    = 1
            
        modelNCVX.setObjective(u_var @ L @ v_var, GRB.MAXIMIZE)
       
        u_var.start = u
        v_var.start = v
       
        modelNCVX.optimize()
        u, v   = u_var.X, v_var.X
        idx_u  = np.where(u > 0.5)[0]
        idx_v  = np.where(v > 0.5)[0]
        prodLR = L[np.ix_(idx_u, idx_v)].sum()
        
        if delta is not None and prodLR > 1.0 + delta:
            early_stop = True
            if steps: print(f"HSB - Iter={nbrect} - EarlyStop - <L,R> = {prodLR}")
       
        if delta is not None and prodLR <= 1.0 + delta:
            modelNCVX.Params.BestObjStop = GRB.INFINITY
            modelNCVX.Params.MIPFocus    = 3
           
            modelNCVX.optimize()
            u, v = u_var.X, v_var.X
            idx_u  = np.where(u > 0.5)[0]
            idx_v  = np.where(v > 0.5)[0]
            prodLR = L[np.ix_(idx_u, idx_v)].sum()

        if not early_stop:
            HSB_iter = float((L * X).sum())/prodLR
            HSBound  = max(HSBound, HSB_iter)

            if steps: print(f"HSB - Iter={nbrect} - <L,R> = {prodLR} - Current HSB: {HSB_iter} - Best HSB: {HSBound}")
       
        # Exact HSB reached at termination
        if prodLR <= 1 + tol:
            end_time = time.time()
            runtime  = end_time - start_time

            print(f"HSB = {HSBound} - time: {runtime} seconds")

            return L, u, v, HSBound, runtime
       
        nbrect += 1
        modelLP.addConstr(Lvar[np.ix_(idx_u, idx_v)].sum() <= 1)
        modelLP.optimize()
        L = Lvar.X

        
def SSB(X, display = False, tol = 1e-6, delta = None, steps = False):
    m, n    = X.shape
   
    start_time = time.time()
    modelLP = gp.Model()
    if not display: modelLP.Params.OutputFlag = 0

    maskX_pos       = X > 1e-12
    maskX_0         = ~maskX_pos
    L_lb = np.full(X.shape, -GRB.INFINITY, dtype=float)
    L_lb[maskX_0]   = 0.0

    L_ub            = np.zeros(X.shape, dtype=float)
    L_ub[maskX_pos] = 1.0 / X[maskX_pos]
    Lvar            = modelLP.addMVar((m, n), lb=L_lb, ub=L_ub)
   
    modelLP.setObjective((Lvar*X).sum(),GRB.MAXIMIZE)    

    modelNCVX = gp.Model()
    if not display: modelNCVX.Params.OutputFlag = 0
    modelNCVX.Params.NonConvex = 2

    u_var     = modelNCVX.addMVar(m, lb=0, ub=1.0)
    v_var     = modelNCVX.addMVar(n, lb=0, ub=X.max(axis=0))
    idx_X_pos = np.argwhere(X >  0)
    idx_X_0   = np.argwhere(X == 0)

    for i, j in idx_X_0  : modelNCVX.addConstr(u_var[i]*v_var[j] == 0)
    modelNCVX.addConstrs(u_var[i] * v_var[j] <= X[i, j] for i, j in idx_X_pos)

    nbrect  = 0
    L       = L_ub.copy()
    
    u       = np.ones(m)
    v       = X.max(axis=0)

    SSBound = 0
    if delta is not None: print(f"SSB - delta = {delta}")
    while True:
        early_stop = False
        
        if delta is not None:
            modelNCVX.Params.BestObjStop = 1.0 + delta
       
        modelNCVX.setObjective(u_var @ L @ v_var, GRB.MAXIMIZE)
       
        u_var.start = u
        v_var.start = v
       
        modelNCVX.optimize()
        u, v   = u_var.X, v_var.X
        prodLR = u@L@v
        
        if delta is not None and prodLR > 1.0 + delta:
            early_stop = True
            if steps: print(f"SSB - Iter={nbrect} - EarlyStop - <L,R> = {prodLR}")
       
        if delta is not None and prodLR <= 1.0 + delta:
            modelNCVX.Params.BestObjStop = GRB.INFINITY
           
            modelNCVX.optimize()
            u, v = u_var.X, v_var.X
            prodLR  = u@L@v

        if not early_stop:
            SSB_iter = float((L * X).sum())/prodLR
            SSBound  = max(SSBound, SSB_iter)

            if steps: print(f"SSB - Iter={nbrect} - <L,R> = {prodLR} - Current SSB: {SSB_iter} - Best SSB: {SSBound}")
       
        # Exact SSB reached at termination
        if prodLR <= 1 + tol:
            end_time = time.time()
            runtime  = end_time - start_time

            print(f"SSB = {SSBound} - time: {runtime} seconds")

            return L, u, v, SSBound, runtime
       
        nbrect += 1
        modelLP.addConstr(u@(Lvar@ v) <= 1)
        modelLP.optimize()
        L = Lvar.X

print(f"Gurobi version : {gp.gurobi.version()}\n\n")

# ****************************** TO TEST THE CODE ******************************

def test_FSB(X):
    r = 1
    time_feasible   = 0.0
    time_infeasible = 0.0

    best_U = []
    best_V = []
    FSB_value = 0

    while True:
        U, V, bound, runtime = FSB(X, r)

        if len(U) > 0:
            time_feasible += runtime
            best_U = U
            best_V = V
            FSB_value = r
            r += 1
        else:
            time_infeasible = runtime
            break

    total_time = time_feasible + time_infeasible

    print(f"FSB = {FSB_value} - time: {time_feasible} + {time_infeasible} seconds")

    return best_U, best_V, FSB_value, time_feasible, time_infeasible

def test_RCB(X, r_start = None):
    m, n = X.shape

    if r_start is None:
        r_start = min(m, n)

    r = r_start
    time_feasible   = 0.0
    time_infeasible = 0.0

    best_U = []
    best_V = []
    RCB_value = None

    while r >= 1:
        U, V, bound, runtime = RCB(X, r)

        if len(U) > 0:
            time_feasible += runtime
            best_U = U
            best_V = V
            RCB_value = r
            r -= 1
        else:
            time_infeasible = runtime
            break

    total_time = time_feasible + time_infeasible

    print(f"RCB = {RCB_value} - time: {time_feasible} + {time_infeasible} seconds")

    return best_U, best_V, RCB_value, time_feasible, time_infeasible

if __name__ == "__main__":
    
    print('************************* LEDM running example ************************')
    X = ledm(5)
    
    print('Matrix X =')
    print(X)
    print('************************************************************************')
    
    
    print('****************************** Fooling set *****************************')
    U_fsb, V_fsb, bound_fsb, time_fsb_feasible, time_fsb_infeasible = test_FSB(X)
    print('************************************************************************')
    
    
    print('*************************** Rectangle covering *************************')
    U_rcb, V_rcb, bound_rcb, time_rcb_feasible, time_rcb_infeasible = test_RCB(X)
    print('************************************************************************')
    
    print('****************************** Hyperplane bound ************************')
    L_hsb, u_hsb, v_hsb, bound_hsb, time_hsb = HSB(X.copy())
    
    print('************************************************************************')
    
    
    print('***************************** Self-scaled bound ************************')
    L_ssb, u_ssb, v_ssb, bound_ssb, time_ssb = SSB(X.copy())
    
    print('************************************************************************')
    
    
    print('******************************* Summary ********************************')
    print("Matrix : LEDM 5")
    print(f'FSB = {bound_fsb}')
    print(f'HSB = {bound_hsb}')
    print(f'SSB = {bound_ssb}')
    print(f'RCB = {bound_rcb}')
    print('************************************************************************')