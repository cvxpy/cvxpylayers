"""Diagnostic for comparing Moreau and analytic PGD step gradients."""
import numpy as np
import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

h = .05
alpha = .1
T = 25
L_t = torch.tensor(np.sin(np.linspace(0, 3*np.pi, T))*.4 + .5, dtype=torch.float64)
P_t = torch.tensor(np.cos(np.linspace(0, 2*np.pi, T))*.4 + 1.5, dtype=torch.float64)
g = cp.Variable(name="g")
s = cp.Variable(name="s")
b = cp.Variable(name="b")
qplus = cp.Variable(name="qplus")
L = cp.Parameter(name="L")
P_over_alpha = cp.Parameter(name="P_over_alpha")
qtar = cp.Parameter(name="qtar")
q = cp.Parameter(name="q")
Q = cp.Parameter(name="Q")
S = cp.Parameter(name="S")
problem = cp.Problem(cp.Minimize(P_over_alpha*g*h + (qplus-qtar)**2 + b**2), [
    L == s+b+g, 0 <= s, s <= S,
    -Q/h/10 <= b, b <= Q/h/10, g >= 0,
    qplus == q-h*b, 0 <= qplus, qplus <= Q,
    0 <= L, L <= 1, 0 <= S, S <= .5,
    1.0/alpha <= P_over_alpha, P_over_alpha <= 2.0/alpha,
    .8*.2 <= qtar, qtar <= .8*1.0,
    0 <= q, q <= 1.0, .2 <= Q, Q <= 1.0,
])
params=[L,P_over_alpha,qtar,q,Q,S]
variables=[g,s,b,qplus]
moreau=CvxpyLayer(problem,parameters=params,variables=variables,solver="MOREAU")

def exact_step(L, P_over_alpha, Q, S, q):
    # Eliminate s,g,qplus. Stationary b for g=0 and g>0.
    delta=q-.8*Q
    b0=h*delta/(1+h*h)
    bg=(h*P_over_alpha+2*h*delta)/(2*(1+h*h))
    optimum=torch.maximum(b0,torch.minimum(bg,L-S))
    lower=torch.maximum(-Q/(10*h),(q-Q)/h)
    upper=torch.minimum(torch.minimum(Q/(10*h),q/h),L)
    b=torch.minimum(torch.maximum(optimum,lower),upper)
    s=torch.minimum(S,torch.maximum(torch.zeros_like(L),L-b))
    g=L-s-b
    return g,s,b,q-h*b

def solve_step(method,t,Q,S,q):
    L=L_t[t];P_over_alpha=P_t[t]/alpha
    if method=="analytic":
        return exact_step(L,P_over_alpha,Q,S,q)
    return moreau(L,P_over_alpha,.8*Q,q,Q,S)

def sim(method,theta):
    theta=torch.tensor(theta,dtype=torch.float64,requires_grad=True)
    Q,S=theta.unbind()
    q=.5*Q
    sols=[]
    for t in range(T):
        g,s,b,q=solve_step(method,t,Q,S,q)
        sols.append(torch.stack((g,s,b,q)))
    sol=torch.stack(sols)
    cost=(sol[:,0]@P_t)*h+Q**2+10*S**2
    grad=torch.autograd.grad(cost,theta)[0]
    return cost.item(),grad.detach().numpy(),sol.detach().numpy()

th=np.array([.8,.25])
for th in [np.array([.8,.25]),np.array([.76771164,.16985142]),np.array([.72107762,.10727486]),np.array([.62670699,.10584761])]:
    ea,ga,va=sim("analytic",th)
    em,gm,vm=sim("moreau",th)
    print("THETA",th,"exact cost",ea,"moreau cost",em,"exact grad",ga,
          "moreau grad",gm,"grad diff",gm-ga,"forward max",np.abs(vm-va).max(),flush=True)
    for t in range(T):
        diff=vm[t]-va[t]
        if np.abs(diff).max()>1e-6:
            print("  STEP",t,"forward delta g,s,b,qplus",diff,"exact",va[t],"moreau",vm[t],flush=True)

# Independent, per-solve Jacobians of (g,qplus) w.r.t. (Q,S,q).
# This removes the 25-step composition and isolates solver backward errors.
theta=np.array([.8,.25])
_,_,exact_outputs=sim("analytic",theta)
qin=np.r_[.5*theta[0],exact_outputs[:-1,3]]
for t in range(T):
    inp=torch.tensor([theta[0],theta[1],qin[t]],dtype=torch.float64,requires_grad=True)
    def local(method,p):
        Q,S,q=p.unbind()
        g,s,b,qp=solve_step(method,t,Q,S,q)
        return torch.stack((g,qp))
    ea=local("analytic",inp).detach().numpy()
    em=local("moreau",inp).detach().numpy()
    ja=torch.autograd.functional.jacobian(lambda p: local("analytic",p),inp).detach().numpy()
    jm=torch.autograd.functional.jacobian(lambda p: local("moreau",p),inp).detach().numpy()
    e=jm-ja
    if np.max(np.abs(e))>1e-5 or np.max(np.abs(ea-em))>1e-6:
        print("LOCAL",t,"Q,S,q=",inp.detach().numpy(),"delta outputs",em-ea,
              "J_exact=",ja,"J_moreau=",jm,"delta J=",e,flush=True)
print("DIAG END",flush=True)

for theta in [np.array([.72107762,.10727486]), np.array([.62670699,.10584761])]:
    _,_,exact_outputs=sim("analytic",theta)
    qin=np.r_[.5*theta[0],exact_outputs[:-1,3]]
    print("LOCAL THETA",theta,flush=True)
    for t in range(T):
        inp=torch.tensor([theta[0],theta[1],qin[t]],dtype=torch.float64,requires_grad=True)
        def local(method,p):
            Q,S,q=p.unbind()
            g,s,b,qp=solve_step(method,t,Q,S,q)
            return torch.stack((g,qp))
        ja=torch.autograd.functional.jacobian(lambda p: local("analytic",p),inp).detach().numpy()
        jm=torch.autograd.functional.jacobian(lambda p: local("moreau",p),inp).detach().numpy()
        e=jm-ja
        if np.max(np.abs(e))>1e-5:
            # Finite differences compare the actual Moreau solution map to its autograd.
            fd=[]
            for k in range(3):
                eps=1e-5
                pert=torch.zeros(3,dtype=torch.float64)
                pert[k]=eps
                yplus=local("moreau",inp.detach()+pert).detach().numpy()
                yminus=local("moreau",inp.detach()-pert).detach().numpy()
                fd.append((yplus-yminus)/(2*eps))
            fd=np.column_stack(fd)
            print("LOCAL",t,"Q,S,q=",inp.detach().numpy(),
                  "J_exact=",ja,"J_moreau=",jm,"J_FD=",fd,
                  "delta_jacobian=",e,flush=True)
print("DIAG FINISHED",flush=True)

# Bypass CVXPYlayers altogether: equivalent four-variable QP at t=23.
# This directly checks Moreau's native Jacobian on an interior q value.
import moreau as mo
import scipy.sparse as sp

theta=np.array([.72107762,.10727486])
_,_,trace=sim("analytic",theta)
qstate0=float(trace[22,3])
Q0,S0=map(float,theta)
L0=float(L_t[23])
c0=float(P_t[23])*h/alpha
P0=sp.csr_array(np.diag([0.,0.,2.,2.]))
A0=sp.csr_array(np.array([
    [1.,1.,1.,0.],     # g+s+b = L
    [0.,0.,h,1.],      # h*b+u = qstate
    [-1.,0.,0.,0.],    # g >= 0
    [0.,-1.,0.,0.],    # s >= 0
    [0.,1.,0.,0.],     # s <= S
    [0.,0.,-1.,0.],    # b >= -Q/(10h)
    [0.,0.,1.,0.],     # b <= Q/(10h)
    [0.,0.,0.,-1.],    # u >= 0
    [0.,0.,0.,1.],     # u <= Q
]))
tf=lambda arr: torch.tensor(np.array(arr),dtype=torch.float64)
pvals=tf(P0.data)
avals=tf(A0.data)
cost=tf([c0,0,0,-1.6*Q0])
cones=mo.Cones(num_zero_cones=2,num_nonneg_cones=7)
def rhs(qstate):
    return torch.stack((qstate.new_tensor(L0),qstate,
        qstate.new_tensor(0.),qstate.new_tensor(0.),
        qstate.new_tensor(S0),qstate.new_tensor(Q0/(10*h)),
        qstate.new_tensor(Q0/(10*h)),qstate.new_tensor(0.),
        qstate.new_tensor(Q0)))
for equilibrium in (True,False):
    settings=mo.Settings(
        solver="ipm", device="cpu",enable_grad=True,
        ipm_settings=mo.IPMSettings(
            diff_method="exact",equilibrate_enable=equilibrium,
            tol_gap_abs=1e-10,tol_gap_rel=1e-10,tol_feas=1e-10
        ),
    )
    native=mo.torch.Solver(n=4,m=9,
        P_row_offsets=torch.as_tensor(P0.indptr,dtype=torch.int64),
        P_col_indices=torch.as_tensor(P0.indices,dtype=torch.int64),
        A_row_offsets=torch.as_tensor(A0.indptr,dtype=torch.int64),
        A_col_indices=torch.as_tensor(A0.indices,dtype=torch.int64),
        cones=cones,settings=settings)
    qi=torch.tensor(qstate0,dtype=torch.float64,requires_grad=True)
    y=native.solve(pvals,avals,cost,rhs(qi)).x[0]
    y.backward()
    fd=[]
    for sign in (-1,1):
        qt=torch.tensor(qstate0+sign*1e-5,dtype=torch.float64)
        fd.append(native.solve(pvals,avals,cost,rhs(qt)).x[0].item())
    print("NATIVE MINI QP", "equil=",equilibrium,"q=",qstate0,
          "g=",y.item(),"expected_g=",L0-S0-qstate0/h,
          "dg/dq_native=",qi.grad.item(),"expected_dg/dq=",-1/h,
          "dg/dq_FD=",(fd[1]-fd[0])/(2e-5),flush=True)
print("NATIVE END",flush=True)
