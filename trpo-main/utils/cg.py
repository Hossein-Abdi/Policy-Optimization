import torch

def conjugate_gradient(fn_fvp, g, nsteps=10, residual_tol=1e-10):
    x = torch.zeros_like(g)
    r = g.clone()
    p = r.clone()
    rdotr = torch.dot(r, r)

    for i in range(nsteps):
        fvp = fn_fvp(p)
        alpha = rdotr / torch.dot(p, fvp)
        x += alpha * p
        r -= alpha * fvp   
        new_rdotr = torch.dot(r, r)
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x

def HS_conjugate_gradient(fn_fvp, g, H, S, damping, nsteps=10, residual_tol=1e-10):
    x = torch.zeros_like(g)
    r = g.clone()
    z = 1 / damping * (r - torch.mv(H.t(), torch.linalg.solve(S, torch.mv(H, r))))
    p = z.clone()
    rdotr = torch.dot(r, z)

    for i in range(nsteps):
        fvp = fn_fvp(p)
        alpha = rdotr / torch.dot(p, fvp)
        x += alpha * p
        r -= alpha * fvp
        z = 1 / damping * (r - torch.mv(H.t(), torch.linalg.solve(S, torch.mv(H, r))))
        new_rdotr = torch.dot(r, z)
        beta = new_rdotr / rdotr
        p = z + beta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x