use crate::matrix::*;
use crate::layer::*;

pub struct Dense {
    weights: Matrix,
    biases:  Matrix,
    w_grad:  Matrix,
    b_grad:  Matrix,
}

impl Dense {
    pub fn new(fan_in: usize, fan_out: usize) -> Self {
        let mut result = Self {
            weights: Matrix::new(fan_out, fan_in),
            w_grad:  Matrix::new(fan_out, fan_in),
            biases:  Matrix::new(fan_out, 1),
            b_grad:  Matrix::new(fan_out, 1),
        };

        let range = (6.0 / (fan_in + fan_out) as f32).sqrt();

        result.weights.random_uniform(-range..range);

        return result;
    }
}

impl Layer for Dense {
    fn forward(&mut self, input: &Matrix, output: &mut Matrix, train: bool) {
        Matrix::mul_into(&self.weights, input, output, false, false, true);
        *output += &self.biases;
    }

    fn backward(&mut self, input: &Matrix, gradient_input: &Matrix, gradient_output: &mut Matrix) {
        Matrix::mul_into(&self.weights, gradient_input, gradient_output, true, false, true);

        self.b_grad += gradient_input;

        // self.w_grad.add_outer_product(gradient_input, input);

        for i in 0..self.weights.get_rows() {
            for j in 0..self.weights.get_cols() {
                self.w_grad[(i, j)] += input[j] * gradient_input[i];
            }
        }
    }

    fn get_param_shapes(&self) -> Box<[(usize, usize)]> {
        vec![self.weights.shape(), self.biases.shape()].into_boxed_slice()
    }

    fn get_params(&mut self) -> Box<[(&mut Matrix, &mut Matrix)]> {
        vec![(&mut self.weights, &mut self.w_grad), (&mut self.biases, &mut self.b_grad)].into_boxed_slice()
    }
}
