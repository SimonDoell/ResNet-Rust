use crate::matrix::*;
use crate::layer::*;

type Function = fn (f32) -> f32;

pub struct Activation {
    forward:    Function,
    derivative: Function,
}

impl Activation {
    pub fn new_tanh() -> Self {
        return Self {
            forward:    |x| x.tanh(),
            derivative: |x| 1.0 - x.tanh().powi(2)
        }
    }

    pub fn new_relu() -> Self {
        return Self {
            forward:    |x| if x < 0.0 {0.0} else { x },
            derivative: |x| if x < 0.0 {0.0} else {1.0}
        };
    }

    pub fn new_leaky_relu() -> Self {
        return Self {
            forward:    |x| if x < 0.0 {x * 0.1} else { x },
            derivative: |x| if x < 0.0 {    0.1} else {1.0}
        }
    }
}

impl Layer for Activation {
    fn forward(&mut self, input: &Matrix, output: &mut Matrix, train: bool) {
        output.clone_from(input);

        for v in output.iter_mut() {
            *v = (self.forward)(*v);
        }
    }

    fn backward(&mut self, input: &Matrix, gradient_input: &Matrix, gradient_output: &mut Matrix) {
        gradient_output.clone_from(gradient_input);

        for (g, v) in gradient_output.iter_mut().zip(input.iter()) {
            *g *= (self.derivative)(*v);
        }
    }

    fn get_param_shapes(&self) -> Box<[(usize, usize)]> {
        return vec![].into_boxed_slice();
    }

    fn get_params(&mut self)   -> Box<[(&mut Matrix, &mut Matrix)]> {
        return vec![].into_boxed_slice();
    }
}