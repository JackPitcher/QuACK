import os                                                                                                                                                                                                      
import sys                                                                                                                                                                                                     
test_directory = os.path.dirname(__file__)                                                                                                                                                                     
src_dir = os.path.join(test_directory, '..', 'src')                                                                                                                                                            
sys.path.append(src_dir)         
from hypothesis import given
from hypothesis.strategies import floats, tuples
import optimizer.optimizer as src
import unittest
TOL = 3e-2


#############################
### TEST GRADIENT DESCENT ###
#############################
TOL = 1e-5
def test_1d_basic_quadratic():
    def quadratic(x): return (3.0 * x[0] + 2.0)**2 + 4.0
    schedule = [[512, 1024, 2048], [0.5, 0.1, 0.05]]
    x = [1.0]
    gd = src.GradientDescent(schedule, 1e-2, quadratic, x)
    x = gd.run()
    
    assert abs(x[0] + 2.0/3.0) < TOL


@given(floats(min_value=-5, max_value=5), floats(min_value=-5, max_value=5), floats(min_value=-1.0, max_value=1.0))
def test_1d_quadratic(a: float, b: float, x0: float):
    def quadratic(x): return (x[0] + a)**2 + b
    schedule = [[512, 1024, 1536, 2048], [0.5, 0.1, 0.05, 0.01]]
    x = [x0]
    gd = src.GradientDescent(schedule, 1e-2, quadratic, x)
    x = gd.run()
    
    assert abs(x[0] + a) < TOL


def test_nd_quadratic_basic():
    a = 1.0
    b = 2.0
    c = 3.0
    d = 4.0
    def quadratic(x):
        return (x[0] + a)**2 + (x[1] + b)**2 + (x[2] + c)**2 + d
    schedule = [[512, 1024, 2048], [0.5, 0.1, 0.05]]
    x = [-0.8, -2.1, -3.6]
    gd = src.GradientDescent(schedule, 1e-2, quadratic, x)
    x = gd.run()
    
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
    schedule = [[512, 1024, 1536, 2048], [0.5, 0.1, 0.05, 0.01]]
    x = list(x0)
    gd = src.GradientDescent(schedule, 1e-2, quadratic, x)
    x = gd.run()
    
    assert abs(x[0] + c[0]) < TOL
    assert abs(x[1] + c[1]) < TOL
    assert abs(x[2] + c[2]) < TOL


#################
### TEST ADAM ###
#################
TOL = 1e-5
class TestAdam(unittest.TestCase):
    def test_1d_basic_quadratic(self):
        def quadratic(x): return (3.0 * x[0] + 2.0)**2 + 4.0
        x = [1.0]
        opt = src.Adam(quadratic, x)
        x = opt.run(max_iter=1e4)
        assert abs(x[0] + 2.0/3.0) < TOL

    @given(floats(min_value=-5, max_value=5), floats(min_value=-5, max_value=5), floats(min_value=-1.0, max_value=1.0))
    def test_1d_quadratic(self, a: float, b: float, x0: float):
        def quadratic(x): return (x[0] + a)**2 + b
        x = [x0]
        opt = src.Adam(quadratic, x)
        x = opt.run(max_iter=1e4)
        print(x, a, b, x0)
        assert abs(x[0] + a) < TOL