use crate::matrix::*;
use crate::mlp::*;

type Function = fn(f32) -> f32;

pub struct Activation {
    forward: Function,
    derivative: Function,
    cached_activation: Matrix,
}

impl Activation {
    pub fn new(forward: Function, derivative: Function) -> Self {
        return Self { forward: forward, derivative: derivative, cached_activation: Matrix::new(0, 0) };
    }

    pub fn leaky_relu() -> Self {
        return Activation::new(
            |x| if x > 0.0 { x } else {x * 0.15},
            |x| if x > 0.0 {1.0} else {    0.15},
        );
    }

    pub fn tanh() -> Self {
        return Activation::new(
            |x| x.tanh(),
            |x| 1.0 - x.tanh().powi(2),
        );
    }
}

impl Layer for Activation {
    fn forward(&mut self, input: &mut Matrix, train: bool) {
        if train {self.cached_activation.clone_from(input);}
        
        for val in input.values_mut() {
            *val = (self.forward)(*val);
        }
    }

    fn backward(&mut self, gradient: &mut Matrix) {
        for (g, &act) in gradient.values_mut().iter_mut().zip(self.cached_activation.values().iter()) {
            *g *= (self.derivative)(act);
        }
    }

    fn get_params(&mut self) -> Vec<(&mut Matrix, &mut Matrix)> {
        return vec![];
    }

    fn get_params_shape(&self) -> Vec<(usize, usize)> {
        return vec![];
    }
}