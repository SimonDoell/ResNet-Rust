use crate::matrix::*;

pub struct AdamWContext {
    momentum: Matrix,
    variance: Matrix,
}

impl AdamWContext {
    pub fn new(shape: (usize, usize)) -> Self {
        return Self { momentum: Matrix::new(shape.0, shape.1), variance: Matrix::new(shape.0, shape.1) };
    }
    
    pub fn update(&mut self, weights: &mut Matrix, gradient: &mut Matrix, lr: f32, lambda: f32, b1: f32, b2: f32, b1_t: f32 /*1.0 - b1^t*/, b2_t: f32 /*1.0 - b2^t*/) {
        for (((w, g), m), v) in
            weights.iter_mut()
            .zip(gradient.iter())
            .zip(self.momentum.iter_mut())
            .zip(self.variance.iter_mut())
        {
            *m = b1 * *m + (1.0 - b1) * g;
            *v = b2 * *v + (1.0 - b2) * g * g;

            let m_hat = *m / b1_t;
            let v_hat = *v / b2_t;

            *w -= lr * (m_hat / (v_hat.sqrt() + 1e-5) + lambda * *w);
        }
    }
}