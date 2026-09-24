"""Deterministic 1-D analogue of the frozen host's Rp logistic/b_cap game.

The critic is D(x)=a*x+b*x*x, p=N(0,.07**2), q=N(mu,.029**2).
Only the fake mean mu is trainable.  D minimizes paired Rp softplus(D(y)-D(x))
plus the host's one-sided input-gradient cap; G minimizes the *non-saturating*
paired Rp softplus(D(x)-D(y)).  Tensor-product Gauss-Hermite quadrature makes
all expectations deterministic.  There are no empirical batches or seed runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from numpy.polynomial.hermite import hermgauss
from scipy.integrate import quad
from scipy.optimize import brentq, root
from scipy.special import expit, ndtr


SIGMA_P = .07
SIGMA_Q = .029
KAPPA = 1.
CAP_COEFF = 1.
MU_DIFFERENCE = 1e-5


class Game:
    def __init__(self, order: int):
        node, raw_weight = hermgauss(order)
        self.z = np.sqrt(2.) * node
        self.w = raw_weight / np.sqrt(np.pi)
        self.x = SIGMA_P * self.z
        self.pair_weight = np.outer(self.w, self.w)

    def coordinates(self, mu):
        x = self.x[:, None]
        y = (mu + SIGMA_Q*self.z)[None, :]
        u = x-y
        v = x*x-y*y
        return x, y, u, v

    @staticmethod
    def cap_terms(a, b, mean, sigma):
        """Exact Gaussian expectation of one host b_cap half-term.

        Hermite nodes poorly resolve the narrow moving activation boundary;
        truncated-normal moments avoid numerical artifacts in its Hessian.
        Returns E[cap^2], E[signed excess * (1,2X)], and
        E[active * (1,2X)(1,2X)^T].  The caller applies coeff/2 to the
        objective and coeff to the gradient/Hessian, as in GradRegularizer.
        """
        if abs(b)<1e-12:
            excess=max(abs(a)-KAPPA,0.)
            sign=np.sign(a)
            active=float(abs(a)>KAPPA)
            v=np.array([1.,2.*mean])
            second=np.array([[1.,2.*mean],
                             [2.*mean,4.*(mean*mean+sigma*sigma)]])
            return excess*excess,excess*sign*v,active*second
        m=a+2.*b*mean
        sd=abs(2.*b)*sigma
        upper=(KAPPA-m)/sd
        lower=(-KAPPA-m)/sd
        phi_upper=np.exp(-.5*upper*upper)/np.sqrt(2.*np.pi)
        phi_lower=np.exp(-.5*lower*lower)/np.sqrt(2.*np.pi)
        p_upper=ndtr(-upper)
        p_lower=ndtr(lower)
        s1_upper=m*p_upper+sd*phi_upper
        s1_lower=m*p_lower-sd*phi_lower
        s2_upper=(m*m+sd*sd)*p_upper+sd*(2.*m+sd*upper)*phi_upper
        s2_lower=(m*m+sd*sd)*p_lower-sd*(2.*m+sd*lower)*phi_lower
        cap2=(s2_upper-2.*KAPPA*s1_upper+KAPPA*KAPPA*p_upper
              +s2_lower+2.*KAPPA*s1_lower+KAPPA*KAPPA*p_lower)
        signed=s1_upper-KAPPA*p_upper+s1_lower+KAPPA*p_lower
        signed_times_s=(s2_upper-KAPPA*s1_upper+s2_lower+KAPPA*s1_lower)
        gradient=np.array([signed,(signed_times_s-a*signed)/b])
        p_active=p_upper+p_lower
        s1_active=s1_upper+s1_lower
        s2_active=s2_upper+s2_lower
        hessian=np.array([[p_active,(s1_active-a*p_active)/b],
                          [(s1_active-a*p_active)/b,
                           (s2_active-2.*a*s1_active+a*a*p_active)/(b*b)]])
        return float(cap2),gradient,hessian

    def d_terms(self, theta, mu, *, hessian=False):
        a, b = map(float, theta)
        _, _, u, v = self.coordinates(mu)
        t = a*u+b*v
        # D(real)-D(fake)=t.  This is exactly the Rp D logistic sign.
        loss = float(np.sum(self.pair_weight*np.logaddexp(0., -t)))
        neg = expit(-t)
        gradient = -np.array([np.sum(self.pair_weight*neg*u),
                              np.sum(self.pair_weight*neg*v)])
        if hessian:
            curvature = self.pair_weight*neg*(1.-neg)
            h = np.array([[np.sum(curvature*u*u), np.sum(curvature*u*v)],
                          [np.sum(curvature*u*v), np.sum(curvature*v*v)]])
        for mean,sigma in ((0.,SIGMA_P),(mu,SIGMA_Q)):
            cap2,cap_gradient,cap_hessian=self.cap_terms(a,b,mean,sigma)
            loss += CAP_COEFF*.5*cap2
            gradient += CAP_COEFF*cap_gradient
            if hessian:
                h += CAP_COEFF*cap_hessian
        return (loss, gradient, h) if hessian else (loss, gradient)

    def g_terms(self, theta, mu):
        a, b = map(float, theta)
        _, y, u, v = self.coordinates(mu)
        t = a*u+b*v
        # G(non-saturating Rp)=softplus(D(real)-D(fake)), not D's loss.
        sig = expit(t)
        slope = a+2.*b*y
        loss = float(np.sum(self.pair_weight*np.logaddexp(0., t)))
        field = float(np.sum(self.pair_weight*sig*(-slope)))
        curvature = float(np.sum(self.pair_weight*(sig*(1.-sig)*slope*slope
                                                   -2.*b*sig)))
        cross_a = float(np.sum(self.pair_weight*(sig*(1.-sig)*u*(-slope)-sig)))
        cross_b = float(np.sum(self.pair_weight*(sig*(1.-sig)*v*(-slope)-2.*y*sig)))
        return dict(loss=loss, gradient_mu=field, own_curvature_mu=curvature,
                    cross_critic=np.array([cross_a, cross_b]))

    def optimal_critic(self, mu, initial):
        solved = root(lambda t: self.d_terms(t, mu, hessian=True)[1], initial,
                      jac=lambda t: self.d_terms(t, mu, hessian=True)[2], tol=1e-11)
        if not solved.success and np.linalg.norm(solved.fun, ord=np.inf)>1e-10:
            raise RuntimeError(f'critic root failed: {solved.message}, {solved.fun}')
        objective, gradient, hessian = self.d_terms(solved.x, mu, hessian=True)
        if np.linalg.norm(gradient, ord=np.inf)>1e-10 or np.linalg.eigvalsh(hessian).min()<=0:
            raise RuntimeError('critic residual or local Hessian check failed')
        return solved.x, objective, gradient, hessian

    def matched_critic(self):
        # Reflection symmetry at mu=0 forces a*=0.  The convex b root pins
        # that stationary point without a tolerance-sensitive 2-D optimizer.
        b = brentq(lambda candidate: self.d_terms((0.,candidate),0.)[1][1],
                   0.,100.,xtol=1e-13)
        return self.optimal_critic(0., np.array([0.,b]))


def examine(order):
    game = Game(order)
    theta, d_loss, d_gradient, h_d = game.matched_critic()
    g = game.g_terms(theta,0.)
    real_cap,_,real_cap_h=game.cap_terms(*theta,0.,SIGMA_P)
    fake_cap,_,fake_cap_h=game.cap_terms(*theta,0.,SIGMA_Q)
    cap_penalty=CAP_COEFF*.5*(real_cap+fake_cap)
    eps = MU_DIFFERENCE
    theta_minus, _, _, _ = game.optimal_critic(-eps,theta)
    theta_plus, _, _, _ = game.optimal_critic(eps,theta)
    response_fd = (theta_plus-theta_minus)/(2.*eps)
    # The derivative of the *G partial gradient* along D*(mu) is the fast-D
    # field's local stiffness.  It is not the derivative of a composite value.
    field_minus = game.g_terms(theta_minus,-eps)['gradient_mu']
    field_plus = game.g_terms(theta_plus,eps)['gradient_mu']
    best_response_stiffness_fd = (field_plus-field_minus)/(2.*eps)
    d_cross = (game.d_terms(theta,-eps)[1]-game.d_terms(theta,eps)[1])/(-2.*eps)
    response_implicit = -np.linalg.solve(h_d,d_cross)
    best_response_stiffness_implicit = g['own_curvature_mu']+g['cross_critic']@response_implicit
    joint_jacobian = np.zeros((3,3))
    joint_jacobian[:2,:2] = h_d
    joint_jacobian[:2,2] = d_cross
    joint_jacobian[2,:2] = g['cross_critic']
    joint_jacobian[2,2] = g['own_curvature_mu']
    eigenvalues = np.linalg.eigvals(-joint_jacobian)
    a_mu = joint_jacobian[np.ix_([0,2],[0,2])]
    # At mu=0 the b direction is even and decouples.  For critic speed r and
    # generator speed 1, trace of the a/mu gradient-flow block is -r*Haa-K.
    critic_speed_threshold = -g['own_curvature_mu']/h_d[0,0]
    return dict(order=order, critic=theta.tolist(), d_loss=d_loss,
                d_logistic_loss=d_loss-cap_penalty,d_cap_penalty=cap_penalty,
                d_cap_active_real=real_cap_h[0,0],
                d_cap_active_fake=fake_cap_h[0,0],
                d_gradient=d_gradient.tolist(), d_hessian=h_d.tolist(),
                d_hessian_eigenvalues=np.linalg.eigvalsh(h_d).tolist(),
                g_loss=g['loss'], g_gradient_mu=g['gradient_mu'],
                g_own_fixed_d_curvature=g['own_curvature_mu'],
                g_cross_critic=g['cross_critic'].tolist(),
                d_gradient_cross_mu=d_cross.tolist(),
                d_response_mu_finite_difference=response_fd.tolist(),
                d_response_mu_implicit=response_implicit.tolist(),
                g_best_response_stiffness_finite_difference=best_response_stiffness_fd,
                g_best_response_stiffness_implicit=float(best_response_stiffness_implicit),
                joint_gradient_jacobian=joint_jacobian.tolist(),
                simultaneous_identity_gradient_flow_eigenvalues=[
                    [float(value.real),float(value.imag)] for value in eigenvalues],
                critic_speed_to_g_speed_trace_threshold=float(critic_speed_threshold),
                a_mu_jacobian_determinant=float(np.linalg.det(a_mu)),
                mu_difference=eps)


def derivative_check():
    """One off-equilibrium numerical check of the analytic cap/G derivatives."""
    game=Game(64)
    theta=np.array([.1,3.])
    mu=.001
    h=1e-5
    _,gradient,hessian=game.d_terms(theta,mu,hessian=True)
    fd_gradient=[]
    fd_hessian=[]
    for j in range(2):
        offset=np.eye(2)[j]*h
        fd_gradient.append((game.d_terms(theta+offset,mu)[0]
                            -game.d_terms(theta-offset,mu)[0])/(2.*h))
        fd_hessian.append((game.d_terms(theta+offset,mu)[1]
                           -game.d_terms(theta-offset,mu)[1])/(2.*h))
    fd_gradient=np.array(fd_gradient)
    fd_hessian=np.column_stack(fd_hessian)
    g=game.g_terms(theta,mu)
    fd_g_own=(game.g_terms(theta,mu+h)['gradient_mu']
              -game.g_terms(theta,mu-h)['gradient_mu'])/(2.*h)
    fd_g_cross=[]
    for j in range(2):
        offset=np.eye(2)[j]*h
        fd_g_cross.append((game.g_terms(theta+offset,mu)['gradient_mu']
                           -game.g_terms(theta-offset,mu)['gradient_mu'])/(2.*h))
    result=dict(d_gradient_max_error=float(np.max(np.abs(gradient-fd_gradient))),
                d_hessian_max_error=float(np.max(np.abs(hessian-fd_hessian))),
                g_own_curvature_error=abs(g['own_curvature_mu']-fd_g_own),
                g_cross_max_error=float(np.max(np.abs(g['cross_critic']-fd_g_cross))))
    cap_errors=[]
    for mean,sigma in ((0.,SIGMA_P),(mu,SIGMA_Q)):
        cap_exact=game.cap_terms(*theta,mean,sigma)[0]
        left=(-KAPPA-theta[0])/(2.*theta[1])
        right=(KAPPA-theta[0])/(2.*theta[1])
        def integrand(x):
            slope=theta[0]+2.*theta[1]*x
            density=np.exp(-.5*((x-mean)/sigma)**2)/(sigma*np.sqrt(2.*np.pi))
            return max(abs(slope)-KAPPA,0.)**2*density
        cap_numeric=(quad(integrand,-np.inf,left,epsabs=1e-12)[0]
                     +quad(integrand,right,np.inf,epsabs=1e-12)[0])
        cap_errors.append(abs(cap_exact-cap_numeric))
    result['cap_exact_vs_independent_integral_max_error']=max(cap_errors)
    if max(result.values())>1e-6:
        raise RuntimeError(f'analytic derivative check failed: {result}')
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    # One primary order and one predeclared quadrature check, not a sweep.
    derivatives=derivative_check()
    base=examine(64)
    check=examine(96)
    if abs(base['critic'][1]-check['critic'][1])>0.1 or abs(
            base['g_best_response_stiffness_finite_difference']-
            check['g_best_response_stiffness_finite_difference'])>0.1:
        raise RuntimeError('quadrature convergence check exceeded declared tolerance')
    if max(abs(x-y) for x,y in zip(base['d_response_mu_finite_difference'],
                                   base['d_response_mu_implicit']))>1e-4:
        raise RuntimeError('implicit critic response disagrees with refitting')
    if abs(base['g_best_response_stiffness_finite_difference']-
           base['g_best_response_stiffness_implicit'])>1e-4:
        raise RuntimeError('best-response field derivative disagrees with refitting')
    result=dict(declaration=dict(scope='one_dimensional_population_analogue',
                                 quadrature_orders=[64,96],mu_difference=MU_DIFFERENCE,
                                 sigma_p=SIGMA_P,sigma_q=SIGMA_Q,kappa=KAPPA,
                                 cap_coefficient=CAP_COEFF,critic='a*x+b*x*x',
                                 python=platform.python_version(),numpy=np.__version__,
                                 scipy=scipy.__version__,
                                 source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
                derivative_check=derivatives,primary=base,quadrature_check=check)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
