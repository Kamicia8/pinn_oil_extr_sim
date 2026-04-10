import ufl

def get_nonlinear_K(u, Kq, u_coeff):
    return Kq * ufl.exp(u_coeff * u)

def get_variational_form(uh, u_n, v, dt, K_func, h, domain):
    dx = ufl.dx(domain=domain)
    F = ((uh - u_n)/dt * v * dx + ufl.inner(K_func * ufl.grad(uh), ufl.grad(v)) * dx - h * v * dx)
    return F