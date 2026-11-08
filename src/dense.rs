use crate::matrix::*;
use crate::mlp::*;

pub struct Dense {
    weights: Matrix,
    biases:  Matrix,
    w_grad:  Matrix,
    b_grad:  Matrix,
    cached_activation: Matrix,
}

impl Dense {
    pub fn new(fan_in: usize, fan_out: usize) -> Self {
        let mut result = Self {
            weights: Matrix::new(fan_out, fan_in),
            w_grad:  Matrix::new(fan_out, fan_in),
            biases:  Matrix::new(fan_out, 1),
            b_grad:  Matrix::new(fan_out, 1),
            cached_activation: Matrix::new(fan_in, 1)
        };

        let range = (6.0 / (fan_in + fan_out) as f32).sqrt();

        result.weights.fill_random(-range, range);

        return result;
    }
}

impl Layer for Dense {
    fn forward(&mut self, input: &mut Matrix, train: bool) {
        if train {self.cached_activation.clone_from(input);}

        let mut result = Matrix::new(0, 0);
        
        Matrix::mul_into(&self.weights, input, &mut result, true);
        Matrix::add_into(&mut result, &self.biases);

        *input = result;
    }

    fn backward(&mut self, gradient: &mut Matrix) {
        let mut prev_grad = Matrix::new(0, 0);
        Matrix::mul_into(&self.weights.transposed(), gradient, &mut prev_grad, true);
        
        for (bg, &g) in self.b_grad.values_mut().iter_mut().zip(gradient.values().iter()) {
            *bg += g;
        }

        let rows = self.weights.get_rows();
        let cols = self.weights.get_cols();
        let grad_slice = gradient.values();
        let act_slice = self.cached_activation.values();
        let w_grad_slice = self.w_grad.values_mut();

        for j in 0..cols {
            let act = act_slice[j];
            let w_col = &mut w_grad_slice[j * rows..(j + 1) * rows];
            for (w, &g) in w_col.iter_mut().zip(grad_slice.iter()) {
                *w += act * g;
            }
        }
        
        *gradient = prev_grad;
    }

    fn get_params(&mut self) -> Vec<(&mut Matrix, &mut Matrix)> {
        return vec![(&mut self.weights, &mut self.w_grad), (&mut self.biases, &mut self.b_grad)];
    }

    fn get_params_shape(&self) -> Vec<(usize, usize)> {
        return vec![self.weights.get_shape(), self.biases.get_shape()];
    }
}