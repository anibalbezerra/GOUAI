"""
targets.py
----------
What the network actually learns: the residual of each observable beyond a smooth trend in S.

    y = g(S) * exp(d),        d = log y - log g(S)

S is the total initial entropy (pixel sum of the input grid). g(S) is a polynomial in log-log space,

    log g(S) = P(u),          u = log(S / S_ref),    P of degree targets.poly_degree (3),

fitted ONCE on the training events and then kept fixed. The network regresses the standardized residual

    z = (d - mu) / sigma

separately for N_ch and [pT] (mu, sigma = mean and spread of d on the training events).

Why: g(S) alone explains > 99 % of the variance of log N_ch, which spans three orders of magnitude.
If the network had to learn y directly, the loss would be dominated by that trivial trend, and the
small event-by-event differences (geometry, fluctuations: the information that makes the network
better than S alone) would be lost in the noise. Standardizing also gives both targets equal weight.

Outside the training range of u, P is continued linearly (value and slope at the edge), so predictions
for events beyond the training range never evaluate a cubic far from where it was fitted.

This file is identical in content to the one used for the published v6 models.
"""
import numpy as np

TARGETS = ('nch', 'pt')


class Trend:
    """log g(S) = polyval(coef, log(S / S_ref)), with linear continuation outside [u_min, u_max]."""

    def __init__(self, coef, S_ref, u_min, u_max):
        self.coef = np.asarray(coef, dtype=np.float64)
        self.S_ref = float(S_ref)
        self.u_min = float(u_min)
        self.u_max = float(u_max)

    @classmethod
    def fit(cls, S, y, degree):
        S = np.asarray(S, np.float64)
        y = np.asarray(y, np.float64)
        assert np.all(S > 0) and np.all(y > 0), 'log-log trend needs positive S and y'
        S_ref = float(np.median(S))
        u = np.log(S / S_ref)
        coef = np.polyfit(u, np.log(y), deg=degree)
        return cls(coef, S_ref, u.min(), u.max())

    def log_g(self, S):
        u = np.log(np.asarray(S, np.float64) / self.S_ref)
        uc = np.clip(u, self.u_min, self.u_max)
        slope = np.polyval(np.polyder(self.coef), uc)
        return np.polyval(self.coef, uc) + slope * (u - uc)

    def __call__(self, S):
        return np.exp(self.log_g(S))

    def to_dict(self):
        return {'coef': self.coef.tolist(), 'S_ref': self.S_ref,
                'u_min': self.u_min, 'u_max': self.u_max,
                'space': 'log g = polyval(coef, log(S/S_ref)), linear continuation outside [u_min,u_max]'}

    @classmethod
    def from_dict(cls, d):
        return cls(d['coef'], d['S_ref'], d['u_min'], d['u_max'])


class ResidualSpec:
    """Trend + standardization for both targets; saved as spec.json next to every trained network."""

    def __init__(self, trends, mu, sigma):
        self.trends = trends    # {'nch': Trend, 'pt': Trend}
        self.mu = mu            # {'nch': float, 'pt': float}
        self.sigma = sigma

    @classmethod
    def fit(cls, S, nch, pt, degree):
        trends, mu, sigma = {}, {}, {}
        for name, y in zip(TARGETS, (nch, pt)):
            t = Trend.fit(S, y, degree)
            d = np.log(y) - t.log_g(S)
            trends[name], mu[name], sigma[name] = t, float(d.mean()), float(d.std(ddof=1))
        return cls(trends, mu, sigma)

    def residual(self, S, y, name):
        return np.log(y) - self.trends[name].log_g(S)

    def transform(self, S, nch, pt):
        """Physical (nch, pt) -> standardized residual z, shape (N, 2)."""
        cols = [(self.residual(S, y, n) - self.mu[n]) / self.sigma[n]
                for n, y in zip(TARGETS, (nch, pt))]
        return np.stack(cols, axis=1)

    def inverse(self, S, z):
        """Standardized residual z (N, 2) -> physical (nch, pt)."""
        z = np.asarray(z, np.float64)
        out = [np.exp(self.trends[n].log_g(S) + self.mu[n] + self.sigma[n] * z[:, i])
               for i, n in enumerate(TARGETS)]
        return out[0], out[1]

    def to_dict(self):
        return {'trends': {n: t.to_dict() for n, t in self.trends.items()},
                'mu': self.mu, 'sigma': self.sigma,
                'targets': 'z = (log y - log g(S) - mu) / sigma ; y = g(S) exp(mu + sigma z)'}

    @classmethod
    def from_dict(cls, d):
        return cls({n: Trend.from_dict(t) for n, t in d['trends'].items()}, d['mu'], d['sigma'])
