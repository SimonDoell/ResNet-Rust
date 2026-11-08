use crate::matrix::*;
use crate::mlp::*;

pub struct LayerNorm {
    scale_shift:       Matrix,
    grad_scale_shift:  Matrix,
    pre_scaling:       Matrix,
    cached_activation: Matrix,
    cached_mean:       f32,
    cached_std:        f32,
}

impl LayerNorm {
    pub fn new() -> Self {
        return Self {
            scale_shift: Matrix::new_from(2, 1, vec![1.0, 0.0]),
            grad_scale_shift: Matrix::new(2, 1),
            pre_scaling:       Matrix::new(0, 0),
            cached_activation: Matrix::new(0, 0),
            cached_mean: 0.0,
            cached_std:  0.0,
        };
    }
}

impl Layer for LayerNorm {
    fn forward(&mut self, input: &mut Matrix, train: bool) {
        if train {self.cached_activation.clone_from(input);}

        let n = input.get_rows() * input.get_cols();

        let scale = self.scale_shift[0];
        let shift = self.scale_shift[1];
        
        let mean = input.sum() / n as f32;

        let mut variance = 0.0;

        for i in 0..n {
            variance += (input[i] - mean).powi(2);
        }

        variance = (variance / n as f32).sqrt();

        for i in 0..n {
            input[i] = ((input[i] - mean) / (variance + 1e-5));
        }

        if train {self.pre_scaling.clone_from(input);}

        for i in 0..n {
            input[i] = scale * input[i] + shift;
        }

        if train {
            self.cached_mean = mean;
            self.cached_std = variance;
        }
    }

    fn backward(&mut self, gradient: &mut Matrix) {        
        let n = gradient.get_rows() * gradient.get_cols();
        let eps = 1e-5;
        
        for i in 0..n {
            self.grad_scale_shift[0] += self.pre_scaling[i] * gradient[i];
            self.grad_scale_shift[1] += gradient[i];
        }

        let mut prev_grad = Matrix::new(n, 1);

        let mut sum0: f32 = gradient.sum() / n as f32;
        let mut sum1: f32 = 0.0;

        for i in 0..n {
            sum1 += gradient[i] * (self.cached_activation[i] - self.cached_mean);
        }

        let a = self.scale_shift[0];
        let v = self.cached_std;
        let m = self.cached_mean;

        for i in 0..n {
            let center = self.cached_activation[i] - m;
            let v_sqr  = (v + eps).powi(2);
            prev_grad[i]  = a / (v + eps) * (gradient[i] - sum0);
            prev_grad[i] -= (a * center * sum1) / (n as f32 * v * v_sqr);
        }

        *gradient = prev_grad;
    }

    fn get_params(&mut self) -> Vec<(&mut Matrix, &mut Matrix)> {
        return vec![(&mut self.scale_shift, &mut self.grad_scale_shift)];
    }

    fn get_params_shape(&self) -> Vec<(usize, usize)> {
        return vec![self.scale_shift.get_shape()];
    }
}