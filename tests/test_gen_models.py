from informed_classification import generative_models
import pytest


def test_pdf(x, model):
    """For input x and a model, tests that the pdf is close"""
    mu = model.dist.mean
    cov = model.dist.cov
    k = model.dim

    a = (2*np.pi)**(-k/2)
    b = np.linalg.det(cov)**(-1/2)
    print(np.linalg.det(cov))
    c = -1/2*(x-mu).T@np.linalg.inv(cov)@(x-mu)
    d = np.exp(c)

    assert np.allclose(model.pdf(x), a*b*d)