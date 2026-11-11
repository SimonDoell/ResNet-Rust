use crate::matrix::*;
use crate::layer::*;

pub struct PositionalEncoding {
    frequencys: usize,
}

impl PositionalEncoding {
    pub fn new(frequencys: usize) -> Self {
        return Self { frequencys: frequencys };
    }
}

impl Layer for PositionalEncoding {
    fn forward(&mut self, input: &Matrix, output: &mut Matrix, train: bool) {
        let n = input.elem_count();
        
        output.ensure_shape((2 * n * self.frequencys + n, 1));

        for i in 0..n {
            output[i] = input[i];
        }

        for k in 0..self.frequencys {
            let freq = 2.0_f32.powi(k as i32);
            for i in 0..n {
                let index = 2 * (k * n + i) + n;
                let phase = input[i] * 2.0 * std::f32::consts::PI * freq;
                output[index + 0] = phase.sin();
                output[index + 1] = phase.cos();
            }
        }
    }

    fn backward(&mut self, input: &Matrix, gradient_input: &Matrix, gradient_output: &mut Matrix) {
        
    }

    fn get_param_shapes(&self) -> Box<[(usize, usize)]> {
        return vec![].into();
    }

    fn get_params(&mut self)   -> Box<[(&mut Matrix, &mut Matrix)]> {
        return vec![].into();
    }
}