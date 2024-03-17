import os                                                                                                                                                                                                      
import sys                                                                                                                                                                                                     
test_directory = os.path.dirname(__file__)                                                                                                                                                                     
src_dir = os.path.join(test_directory, '..', 'src')                                                                                                                                                            
sys.path.append(src_dir)         
from hypothesis import given
from hypothesis.strategies import floats, tuples
import Optimizer.optimizer as src
TOL = 3e-2

#############################
### TEST SCALAR MINIMIZER ###
#############################
def test_basic_quadratic():
    def quadratic(x): return (3.0 * x[0] + 2.0)**2 + 4.0
    x = [1.0]
    for _ in range(1024):
        x = src.scalar_minimizer(quadratic, x, step_size=1e-2)
    
    assert abs(x[0] + 2.0/3.0) < TOL


@given(floats(min_value=-10, max_value=10), floats(min_value=-10, max_value=10), floats(min_value=-1.0, max_value=1.0))
def test_quadratic(a: float, b: float, x0: float):
    def quadratic(x): return (x[0] + a)**2 + b
    x = [x0]
    for _ in range(2048):
        x = src.scalar_minimizer(quadratic, x, step_size=1e-2)
    
    assert abs(x[0] + a) < TOL


@given(floats(min_value=-2.0, max_value=-1.0), \
       floats(min_value=1.0,  max_value=2.0), \
       floats(min_value=-5.0, max_value=5.0))
def test_quadratic_with_bounds(lower: float, upper: float, param: float):
    def quadratic(x): return (x[0] + param)**2
    x = [0.0]
    for _ in range(2048):
        x = src.scalar_minimizer(quadratic, x, step_size=1e-2, bs=(lower, upper))
    
    assert abs(x[0] + param) < TOL or abs(x[0]-lower) < TOL or abs(x[0]-upper) < TOL


#############################
### TEST GRADIENT DESCENT ###
#############################
def test_1d_basic_quadratic():
    def quadratic(x): return (3.0 * x[0] + 2.0)**2 + 4.0
    x = [1.0]
    for _ in range(2048):
        x = src.gradient_descent(quadratic, x, step_size=1e-2, lr=0.1)
    
    assert abs(x[0] + 2.0/3.0) < TOL


@given(floats(min_value=-5, max_value=5), floats(min_value=-5, max_value=5), floats(min_value=-1.0, max_value=1.0))
def test_1d_quadratic(a: float, b: float, x0: float):
    def quadratic(x): return (x[0] + a)**2 + b
    x = [x0]
    for _ in range(2048):
        x = src.gradient_descent(quadratic, x, step_size=1e-2, lr=0.1)
    
    assert abs(x[0] + a) < TOL


def test_nd_quadratic_basic():
    a = 1.0
    b = 2.0
    c = 3.0
    d = 4.0
    def quadratic(x):
        return (x[0] + a)**2 + (x[1] + b)**2 + (x[2] + c)**2 + d
    x = [-0.8, -2.1, -3.6]
    for _ in range(2048):
        x = src.gradient_descent(quadratic, x, step_size=1e-2, lr=0.1)
    
    assert abs(x[0] + a) < TOL
    assert abs(x[1] + b) < TOL
    assert abs(x[2] + c) < TOL


@given(tuples(floats(min_value=-5, max_value=5),
              floats(min_value=-5, max_value=5),
              floats(min_value=-5, max_value=5),
              floats(min_value=-5, max_value=5)),
       tuples(floats(min_value=-1.0, max_value=1.0),
              floats(min_value=-1.0, max_value=1.0),
              floats(min_value=-1.0, max_value=1.0)))
def test_nd_quadratic(c: tuple[float, float, float, float], 
                      x0: tuple[float, float, float]):
    def quadratic(x):
        return (x[0] + c[0])**2 + (x[1] + c[1])**2 + (x[2] + c[2])**2 + c[3]
    x = list(x0)
    for _ in range(2048):
        x = src.gradient_descent(quadratic, x, step_size=1e-2, lr=0.1)
    
    assert abs(x[0] + c[0]) < TOL
    assert abs(x[1] + c[1]) < TOL
    assert abs(x[2] + c[2]) < TOL