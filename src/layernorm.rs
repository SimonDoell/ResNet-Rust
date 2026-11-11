use crate::matrix::*;
use crate::layer::*;

pub struct LayerNorm {
    scale:             Matrix,
    shift:             Matrix,
    grad_scale:        Matrix,
    grad_shift:        Matrix,
    
    pre_scaling:       Matrix,
    cached_activation: Matrix,
    cached_mean:       f32,
    cached_std:        f32,
}

impl LayerNorm {
    pub fn new(features: usize) -> Self {
        return Self {
            scale:             Matrix::new_from_value(features, 1, 1.0),
            shift:             Matrix::new_from_value(features, 1, 0.0),
            grad_scale:        Matrix::new(features, 1),
            grad_shift:        Matrix::new(features, 1),
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

        let mean = input.mean();

        let mut variance = 0.0;

        for i in 0..n {
            variance += (input[i] - mean).powi(2);
        }

        variance = (variance / n as f32 + 1e-5).sqrt();

        for i in 0..n {
            output[i] = ((input[i] - mean) / variance);
        }

        if train {self.pre_scaling.clone_from(output);}

        for i in 0..n {
            output[i] = self.scale[i] * output[i] + self.shift[i];
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
            self.grad_scale[i] += self.pre_scaling[i] * gradient_input[i];
            self.grad_shift[i] += gradient_input[i];
        }

        let mut sum_dxhat = 0.0;
        let mut sum_dxhat_xhat = 0.0;
        for i in 0..n {
            let dxhat = gradient_input[i] * self.scale[i];
            sum_dxhat += dxhat;
            sum_dxhat_xhat += dxhat * self.pre_scaling[i];
        }
        
        let inv_std = 1.0 / self.cached_std;
        for i in 0..n {
            let dxhat = gradient_input[i] * self.scale[i];
            gradient_output[i] = inv_std * (dxhat - (sum_dxhat + self.pre_scaling[i] * sum_dxhat_xhat) / n as f32);
        }
    }

    fn get_param_shapes(&self) -> Box<[(usize, usize)]> {
        return vec![self.scale.shape(), self.shift.shape()].into_boxed_slice();
    }

    fn get_params(&mut self)   -> Box<[(&mut Matrix, &mut Matrix)]> {
        return vec![(&mut self.scale, &mut self.grad_scale), (&mut self.shift, &mut self.grad_shift)].into_boxed_slice();
    }
}