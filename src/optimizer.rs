use crate::matrix::*;

pub trait Optimizer {
    fn reshape(&mut self, shape: (usize, usize));
    fn update(&mut self, param: (&mut Matrix, &mut Matrix));
}