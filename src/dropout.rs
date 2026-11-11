use crate::matrix::*;
use crate::layer::*;
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
    fn forward(&mut self, input: &Matrix, output: &mut Matrix, train: bool) {
        let n = input.elem_count();
        
        if !train || self.p == 0.0 {
            output.clone_from(input);
        } else {
            output.ensure_shape((n, 1));
        }

        let scale = 1.0 / (1.0 - self.p);

        if n != self.cached_dropout.len() {
            self.cached_dropout = vec![false; n as usize].into();
        }

        for i in 0..n {
            if rand::random_range(0.0..1.0) < self.p {
                output[i] = 0.0;
                self.cached_dropout[i] = true
            } else {
                output[i] = input[i] * scale;
                self.cached_dropout[i] = false;
            }
        }
    }

    fn backward(&mut self, input: &Matrix, gradient_input: &Matrix, gradient_output: &mut Matrix) {
        let n = gradient_input.elem_count();

        gradient_output.ensure_shape((n, 1));

        let scale = 1.0 / (1.0 - self.p);

        for i in 0..n {
            if self.cached_dropout[i] {
                gradient_output[i] = 0.0;
            } else {
                gradient_output[i] *= scale;
            }
        }
    }

    fn get_param_shapes(&self) -> Box<[(usize, usize)]> {
        return vec![].into();
    }

    fn get_params(&mut self)   -> Box<[(&mut Matrix, &mut Matrix)]> {
        return vec![].into();
    }
}