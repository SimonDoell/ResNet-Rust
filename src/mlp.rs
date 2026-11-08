use crate::matrix::*;

pub trait Layer {
    fn forward(&mut self, input: &mut Matrix, train: bool);
    fn backward(&mut self, gradient: &mut Matrix);
    fn get_params(&mut self) -> Vec<(&mut Matrix, &mut Matrix)>; // (weights, gradient)
    fn get_params_shape(&self) -> Vec<(usize, usize)>;
}

struct AdamWContext {
    momentum: Matrix,
    variance: Matrix,
}

impl AdamWContext {
    fn new((rows, cols): (usize, usize)) -> Self {
        return Self { momentum: Matrix::new(rows, cols), variance: Matrix::new(rows, cols) };
    }

    fn apply(&mut self, (weight, gradient): (&mut Matrix, &mut Matrix), lr: f32, lambda: f32, betta_1: f32, betta_2: f32, betta_1_t: f32, betta_2_t: f32) {
        for i in 0..weight.get_rows() * weight.get_cols() {
            let mut w = &mut weight[i];
            let mut g = gradient[i];
            let mut m = &mut self.momentum[i];
            let mut v = &mut self.variance[i];

            // AdamW update rule:
            // m = betta_1 * m + (1.0 - betta_1) * g
            // v = betta_2 * v + (1.0 - betta_2) * g * g
            // 
            // m_hat = m / (1.0 - betta_1.pow(t))
            // v_hat = v / (1.0 - betta_2.pow(t))
            // 
            // w -= lr * (m_hat / (v_hat.sqrt() + eps) + lambda * w)
            
            *m = betta_1 * *m + (1.0 - betta_1) * g;
            *v = betta_2 * *v + (1.0 - betta_2) * g * g;
            let m_hat = *m / (1.0 - betta_1_t);
            let v_hat = *v / (1.0 - betta_2_t);
            *w -= lr * (m_hat / (v_hat.sqrt() + 1e-6) + lambda * *w)
        }

        gradient.zero();
    }
}

pub struct MLP {
    pub learning_rate: f32,
    pub lambda:        f32,
    pub betta_1:       f32,
    pub betta_2:       f32,
    pub mini_batch:    usize,

    layers: Vec<Box<dyn Layer>>,
    adam_w: Vec<AdamWContext>,
    t: usize,
}

impl MLP {
    pub fn new() -> Self {
        return MLP {
            learning_rate: 0.01,
            lambda:        0.001,
            betta_1:       0.9,
            betta_2:       0.999,
            mini_batch:    5,
            t: 1,
            layers: Vec::new(),
            adam_w: Vec::new()
        };
    }
    
    pub fn push_layer<L: Layer + 'static>(&mut self, layer: L) {
        let mut shapes = layer.get_params_shape();
        
        self.layers.push(Box::new(layer));

        for shape in shapes {
            self.adam_w.push(AdamWContext::new(shape));
        }
    }
    
    pub fn forward(&mut self, input: &mut Matrix) {
        for layer in self.layers.iter_mut() {
            layer.forward(input, false);
        }
    }

    fn backward(&mut self, gradient: &mut Matrix) {
        for layer in self.layers.iter_mut().rev() {
            layer.backward(gradient);
        }
    }

    fn apply_gradients(&mut self) {
        let mut i = 0;
        let mut k = 0;

        while i < self.layers.len() {
            let params = self.layers[i].get_params();

            for param in params {
                self.adam_w[k].apply(param, self.learning_rate, self.lambda, self.betta_1, self.betta_2, self.betta_1.powf(self.t as f32), self.betta_2.powf(self.t as f32));
                k += 1;
            }

            i += 1;
        }
    }
    
    pub fn train(&mut self, input: &Matrix, desired_output: &Matrix) {
        let mut activation = input.clone();
        
        for layer in self.layers.iter_mut() {
            layer.forward(&mut activation, true);
        }

        Matrix::sub_into(&mut activation, desired_output);

        self.backward(&mut activation);

        if self.t % self.mini_batch == 0 {
            self.apply_gradients();
        }
        
        self.t += 1;
    }
}