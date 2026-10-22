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

def FSB(X, r, display = False, seed = None, time_limit = None):
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
    if seed is not None:       model.Params.Seed      = seed
    if time_limit is not None: model.Params.TimeLimit = time_limit

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
        print(f"FSB - Fooling set of size {r} - time: {runtime:.2f} seconds")
        return U.X, V.X, r, runtime
    if model.status==3:
        print(f"FSB - No fooling set of size {r} - time: {runtime:.2f} seconds")
        return [], [], None, runtime

    print(f"FSB - Optimization ended with status {model.status} - ({m}x{n}), r={r} - time: {runtime:.2f} seconds")
    raise RuntimeError("FSB: no feasibility or infeasibility conclusion (status " + str(model.Status) + ")")
        
def RCB(X, r, display = False, seed = None, time_limit = None):
    Z     = X.copy()
    m, n  = X.shape
    for i in range(m):
        for j in range(n):
            if   X[i,j] >  1e-6: Z[i,j] =  1.0
            elif X[i,j] < -1e-6: Z[i,j] = -1.0
            else:                Z[i,j] =  0.0
    
    udisj = np.any(Z < 0)

    start_time = time.time()
    model      = gp.Model()

    if not display: model.Params.OutputFlag = 0
    model.Params.NonConvex  = 2
    if seed is not None:       model.Params.Seed      = seed
    if time_limit is not None: model.Params.TimeLimit = time_limit
    #model.Params.MIPFocus   = 1
    #model.Params.Heuristics = 0.5
    U = model.addMVar((m, r), vtype=GRB.BINARY)
    V = model.addMVar((r, n), vtype=GRB.BINARY)
    
    if udisj:
        P = (Z  > 0).astype(int)  # Positive entries (must be covered)
        A = (Z != 0).astype(int)  # Allowed entries (can be covered)
        model.addConstr(U @ V >= P)
        model.addConstr(U @ V <= r * A)
        
    else:
        model.addConstr(U @ V >= Z)
        model.addConstr(U @ V <= r * Z)

    
    model.optimize()
    end_time = time.time()
    runtime  = end_time - start_time

    if model.status==2:
        print(f"RCB - Rectangle covering of size {r} - time: {runtime:.2f} seconds")
        return U.X, V.X, r, runtime
    if model.status==3:
        print(f"RCB - No rectangle covering of size {r} - time: {runtime:.2f} seconds")
        return [], [], None, end_time - start_time

    print(f"RCB - Optimization ended with status {model.status} -  r={r} - time: {runtime:.2f} seconds")
    raise RuntimeError("RCB: no feasibility or infeasibility conclusion (status " + str(model.Status) + ")")

        
def HSB(X, display = False, tol = 1e-12, delta = None, mip_gap = None, steps = False, seed = None, time_limit = None):
    m, n    = X.shape
    if not np.any(X):
        return np.zeros_like(X, dtype=float), np.zeros(m), np.zeros(n), 0.0, 0.0, 0.0, 0.0
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
    modelNCVX.Params.NonConvex  = 2
    modelNCVX.Params.BestBdStop = 1.0 + tol
    modelNCVX.Params.MIPGapAbs  = 0.0
    if mip_gap is not None: modelNCVX.Params.MIPGap = mip_gap
    mip_gap = modelNCVX.Params.MIPGap
    if seed    is not None: modelNCVX.Params.Seed   = seed

    u_var       = modelNCVX.addMVar(m,vtype=GRB.BINARY)
    v_var       = modelNCVX.addMVar(n,vtype=GRB.BINARY)
    idx_X_0     = np.argwhere(maskX_0)

    for i, j in idx_X_0  : modelNCVX.addConstr(u_var[i] + v_var[j] <= 1)

    nbrect      = 0
    L           = L_ub.copy()
    upper_bound = float((L * X).sum())
    
    u           = np.ones(m)
    v           = np.ones(n)

    HSBound = 0
    B       = 0
    
    if delta is not None: print(f"HSB - Delta = {delta}")
    while True:
        if time_limit is not None and time.time() - start_time >= time_limit: break
        early_stop = False
        modelNCVX.Params.MIPGap = mip_gap if delta is not None else 0.0

        if delta is not None:
            modelNCVX.Params.BestObjStop = 1.0 + delta
            modelNCVX.Params.MIPFocus    = 1
            
        modelNCVX.setObjective(u_var @ L @ v_var, GRB.MAXIMIZE)
       
        u_var.start = u
        v_var.start = v
       
        modelNCVX.Params.TimeLimit = GRB.INFINITY
        if time_limit is not None:
            modelNCVX.Params.TimeLimit = max(0, time_limit - (time.time() - start_time))
        modelNCVX.optimize()
        
        alpha_bar = getattr(modelNCVX, "ObjBound", GRB.INFINITY)
        
        if modelNCVX.Status in (GRB.OPTIMAL, GRB.USER_OBJ_LIMIT, GRB.TIME_LIMIT, GRB.INTERRUPTED) and 0 < alpha_bar < GRB.INFINITY:
            B     = HSBound = max(B, float((L * X).sum()) / alpha_bar)
        if modelNCVX.Status not in (GRB.OPTIMAL, GRB.USER_OBJ_LIMIT) or modelNCVX.SolCount == 0: break
    
        u, v      = u_var.X, v_var.X
        idx_u     = np.where(u > 0.5)[0]
        idx_v     = np.where(v > 0.5)[0]
        prodLR    = L[np.ix_(idx_u, idx_v)].sum()
        
        if delta is not None and prodLR > 1.0 + delta:
            early_stop = True
            if steps: print(f"HSB - Iter={nbrect} - EarlyStop - <L,R> = {prodLR} - <L,X> = {(L * X).sum()}")
       
        if delta is not None and prodLR <= 1.0 + delta:
            modelNCVX.Params.BestObjStop = 1.0 + max(tol, 1e-6)
            modelNCVX.Params.MIPGap      = 0.0
            modelNCVX.Params.MIPFocus    = 3
           
            modelNCVX.Params.TimeLimit = GRB.INFINITY
            if time_limit is not None:
                modelNCVX.Params.TimeLimit = max(0, time_limit - (time.time() - start_time))
            modelNCVX.optimize()
            
            alpha_bar = getattr(modelNCVX, "ObjBound", GRB.INFINITY)
            
            if modelNCVX.Status in (GRB.OPTIMAL, GRB.USER_OBJ_LIMIT, GRB.TIME_LIMIT, GRB.INTERRUPTED) and 0 < alpha_bar < GRB.INFINITY:
                B     = HSBound = max(B, float((L * X).sum()) / alpha_bar)
            if modelNCVX.Status not in (GRB.OPTIMAL, GRB.USER_OBJ_LIMIT) or modelNCVX.SolCount == 0: break
            
            u, v      = u_var.X, v_var.X
            idx_u     = np.where(u > 0.5)[0]
            idx_v     = np.where(v > 0.5)[0]
            prodLR    = L[np.ix_(idx_u, idx_v)].sum()

        if not early_stop and modelNCVX.Status == GRB.OPTIMAL:
            HSB_iter  = float((L * X).sum())/alpha_bar
            HSBound   = max(HSBound, HSB_iter)

            if steps: print(f"HSB - Iter={nbrect} - <L,R> = {prodLR} - Current HSB: {HSB_iter} - Best HSB: {HSBound}")
       
        # Numerical stopping criterion
        if 0 < alpha_bar <= 1 + tol:
            end_time  = time.time()
            runtime   = end_time - start_time

            print(f"HSB = {HSBound} - time: {runtime:.2f} seconds")
            print(f"Certificate : {B} <= HSB(X) <= {upper_bound}")

            return L, u, v, HSBound, runtime, B, upper_bound
       
        nbrect += 1
        modelLP.addConstr(Lvar[np.ix_(idx_u, idx_v)].sum() <= 1)
        modelLP.Params.TimeLimit = GRB.INFINITY
        if time_limit is not None:
            modelLP.Params.TimeLimit = max(0, time_limit - (time.time() - start_time))
        modelLP.optimize()
        if modelLP.Status != GRB.OPTIMAL: break
        L = Lvar.X
        upper_bound = float((L * X).sum())

    runtime = time.time() - start_time
    print(f"HSB interrupted - lower bound: {B} - upper bound: {upper_bound} - time: {runtime:.2f} seconds")
    return L, [], [], HSBound, runtime, B, upper_bound


def SSB(X, display = False, tol = 1e-6, delta = None, mip_gap = None, steps = False, seed = None, time_limit = None):
    m, n    = X.shape
    if not np.any(X):
        return np.zeros_like(X, dtype=float), np.zeros(m), np.zeros(n), 0.0, 0.0, 0.0, 0.0
   
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
    if mip_gap is not None: modelNCVX.Params.MIPGap = mip_gap
    if seed    is not None: modelNCVX.Params.Seed   = seed

    u_var     = modelNCVX.addMVar(m, lb=0, ub=1.0)
    v_var     = modelNCVX.addMVar(n, lb=0, ub=X.max(axis=0))
    idx_X_pos = np.argwhere(X >  0)
    idx_X_0   = np.argwhere(X == 0)

    for i, j in idx_X_0  : modelNCVX.addConstr(u_var[i]*v_var[j] == 0)
    modelNCVX.addConstrs(u_var[i] * v_var[j] <= X[i, j] for i, j in idx_X_pos)

    nbrect      = 0
    L           = L_ub.copy()
    upper_bound = float((L * X).sum())
    
    u           = np.ones(m)
    v           = X.max(axis=0)

    SSBound     = 0
    B           = 0 
    
    if delta is not None: print(f"SSB - Delta = {delta}")
    while True:
        if time_limit is not None and time.time() - start_time >= time_limit: break
        early_stop = False
        
        if delta is not None:
            modelNCVX.Params.BestObjStop = 1.0 + delta
       
        modelNCVX.setObjective(u_var @ L @ v_var, GRB.MAXIMIZE)
       
        u_var.start = u
        v_var.start = v
       
        modelNCVX.Params.TimeLimit = GRB.INFINITY
        if time_limit is not None:
            modelNCVX.Params.TimeLimit = max(0, time_limit - (time.time() - start_time))
        modelNCVX.optimize()
        
        alpha_bar = getattr(modelNCVX, "ObjBound", GRB.INFINITY)
        if modelNCVX.Status in (GRB.OPTIMAL, GRB.USER_OBJ_LIMIT, GRB.TIME_LIMIT, GRB.INTERRUPTED) and 0 < alpha_bar < GRB.INFINITY:
            B     = max(B, float((L * X).sum()) / alpha_bar)
        if modelNCVX.Status not in (GRB.OPTIMAL, GRB.USER_OBJ_LIMIT) or modelNCVX.SolCount == 0: break
    
        u, v      = u_var.X, v_var.X
        prodLR    = u@L@v
        
        if delta is not None and prodLR > 1.0 + delta:
            early_stop = True
            if steps: print(f"SSB - Iter={nbrect} - EarlyStop - <L,R> = {prodLR} - <L,X> = {(L * X).sum()}")
       
        if delta is not None and prodLR <= 1.0 + delta:
            modelNCVX.Params.BestObjStop = GRB.INFINITY
           
            modelNCVX.Params.TimeLimit = GRB.INFINITY
            if time_limit is not None:
                modelNCVX.Params.TimeLimit = max(0, time_limit - (time.time() - start_time))
            modelNCVX.optimize()
            
            alpha_bar = getattr(modelNCVX, "ObjBound", GRB.INFINITY)
            if modelNCVX.Status in (GRB.OPTIMAL, GRB.USER_OBJ_LIMIT, GRB.TIME_LIMIT, GRB.INTERRUPTED) and 0 < alpha_bar < GRB.INFINITY:
                B     = max(B, float((L * X).sum()) / alpha_bar)
            if modelNCVX.Status not in (GRB.OPTIMAL,) or modelNCVX.SolCount == 0: break
        
            u, v      = u_var.X, v_var.X
            prodLR    = u@L@v

        if not early_stop and modelNCVX.Status == GRB.OPTIMAL:
            SSB_iter  = float((L * X).sum())/prodLR
            SSBound   = max(SSBound, SSB_iter)

            if steps: print(f"SSB - Iter={nbrect} - <L,R> = {prodLR} - Current SSB: {SSB_iter} - Best SSB: {SSBound}")
       
        # Numerical stopping criterion
        if not early_stop and modelNCVX.Status == GRB.OPTIMAL and 0 < prodLR <= 1 + tol:
            end_time  = time.time()
            runtime   = end_time - start_time
            
            print(f"SSB = {SSBound} - time: {runtime:.2f} seconds")
            print(f"Certificate : {B} <= SSB(X) <= {upper_bound}")

            return L, u, v, SSBound, runtime, B, upper_bound
       
        nbrect += 1
        modelLP.addConstr(u@(Lvar@ v) <= 1)
        modelLP.Params.TimeLimit = GRB.INFINITY
        if time_limit is not None:
            modelLP.Params.TimeLimit = max(0, time_limit - (time.time() - start_time))
        modelLP.optimize()
        if modelLP.Status != GRB.OPTIMAL: break
        L = Lvar.X
        upper_bound = float((L * X).sum())

    runtime = time.time() - start_time
    print(f"SSB interrupted - lower bound: {B} - upper bound: {upper_bound} - time: {runtime:.2f} seconds")
    return L, [], [], SSBound, runtime, B, upper_bound

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
    L_hsb, u_hsb, v_hsb, bound_hsb, time_hsb, B_hsb, upper_bound_hsb = HSB(X.copy())
    
    print('************************************************************************')
    
    
    print('***************************** Self-scaled bound ************************')
    L_ssb, u_ssb, v_ssb, bound_ssb, time_ssb, B_ssb, upper_bound_ssb = SSB(X.copy())
    
    print('************************************************************************')
    
    
    print('******************************* Summary ********************************')
    print("Matrix : LEDM 5")
    print(f'FSB = {bound_fsb}')
    print(f'HSB = {bound_hsb}')
    print(f'SSB = {bound_ssb}')
    print(f'RCB = {bound_rcb}')
    print('************************************************************************')
