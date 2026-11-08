use crate::matrix::*;
use crate::mlp::*;
use rand::*;

pub struct Dropout {
    p: f32,
    cached_dropout: Box<[bool]>,
}

impl Dropout {
    pub fn new(percentage: f32) -> Self {
        return Dropout { p: percentage, cached_dropout: Box::default() };
    }
}

impl Layer for Dropout {
    fn forward(&mut self, input: &mut Matrix, train: bool) {
        if !train || self.p == 0.0 {return;}
        
        let n = input.get_rows() * input.get_cols();
        
        if self.cached_dropout.len() != n {
            self.cached_dropout = vec![false; n].into_boxed_slice();
        }

        let scale = 1.0 / (1.0 - self.p);

        for i in 0..n {
            if rand::random_range(0.0..1.0) < self.p {
                input[i] = 0.0;
                self.cached_dropout[i] = true;
            } else {
                input[i] *= scale;
                self.cached_dropout[i] = false;
            }
        }
    }

    fn backward(&mut self, gradient: &mut Matrix) {
        let n = gradient.get_rows() * gradient.get_cols();

        let scale = 1.0 / (1.0 - self.p);

        for i in 0..n {
            if self.cached_dropout[i] {
                gradient[i] = 0.0;
            } else {
                gradient[i] *= scale;
            }
        }
    }

    fn get_params(&mut self) -> Vec<(&mut Matrix, &mut Matrix)> {
        return vec![];
    }

    fn get_params_shape(&self) -> Vec<(usize, usize)> {
        return vec![];
    }
}