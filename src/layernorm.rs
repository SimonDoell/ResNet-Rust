use crate::matrix::*;
use crate::layer::*;

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
            scale_shift: Matrix::new_from(2, 1, vec![1.0, 0.0].as_slice()),
            grad_scale_shift: Matrix::new(2, 1),
            pre_scaling:       Matrix::new(0, 0),
            cached_activation: Matrix::new(0, 0),
            cached_mean: 0.0,
            cached_std:  0.0,
        };
    }
}

impl Layer for LayerNorm {
    fn forward(&mut self, input: &Matrix, output: &mut Matrix, train: bool) {
        if train {self.cached_activation.clone_from(input);}

        output.ensure_shape(input.shape());

        let n = input.elem_count();

        let scale = self.scale_shift[0];
        let shift = self.scale_shift[1];
        
        let mean = input.sum() / n as f32;

        let mut variance = 0.0;

        for i in 0..n {
            variance += (input[i] - mean).powi(2);
        }

        variance = (variance / n as f32).sqrt();

        for i in 0..n {
            output[i] = ((input[i] - mean) / (variance + 1e-5));
        }

        if train {self.pre_scaling.clone_from(output);}

        for i in 0..n {
            output[i] = scale * input[i] + shift;
        }

        if train {
            self.cached_mean = mean;
            self.cached_std = variance;
        }
    }

    fn backward(&mut self, input: &Matrix, gradient_input: &Matrix, gradient_output: &mut Matrix) {       
        let n = gradient_input.elem_count();
        let eps = 1e-5;

        gradient_output.ensure_shape(gradient_input.shape());
        
        for i in 0..n {
            self.grad_scale_shift[0] += self.pre_scaling[i] * gradient_input[i];
            self.grad_scale_shift[1] += gradient_input[i];
        }

        let mut sum0: f32 = gradient_input.sum() / n as f32;
        let mut sum1: f32 = 0.0;

        for i in 0..n {
            sum1 += gradient_input[i] * (self.cached_activation[i] - self.cached_mean);
        }

        let a = self.scale_shift[0];
        let v = self.cached_std;
        let m = self.cached_mean;

        for i in 0..n {
            let center = self.cached_activation[i] - m;
            let v_sqr  = (v + eps).powi(2);
            gradient_output[i]  = a / (v + eps) * (gradient_input[i] - sum0);
            gradient_output[i] -= (a * center * sum1) / (n as f32 * v * v_sqr);
        }
    }

    fn get_param_shapes(&self) -> Box<[(usize, usize)]> {
        return vec![self.scale_shift.shape()].into_boxed_slice();
    }

    fn get_params(&mut self)   -> Box<[(&mut Matrix, &mut Matrix)]> {
        return vec![(&mut self.scale_shift, &mut self.grad_scale_shift)].into_boxed_slice();
    }
}