use crate::matrix::*;

pub trait Layer {
    fn forward(&mut self, input: &Matrix, output: &mut Matrix, train: bool);
    fn backward(&mut self, input: &Matrix, gradient_input: &Matrix, gradient_output: &mut Matrix);
    fn get_param_shapes(&self) -> Box<[(usize, usize)]>;
    fn get_params(&mut self)   -> Box<[(&mut Matrix, &mut Matrix)]>;
}