use crate::matrix::*;
use crate::adamw ::*;
use crate::layer ::*;

type NodeRef = u32;

fn split_vec<T>(slice: &mut [T], element: usize) -> (&mut [T], &mut T, &mut [T]) {
    let (left, right) = slice.split_at_mut(element);
    let (element, right) = right.split_first_mut().unwrap();

    (left, element, right)
}

struct Node {
    layer: Box<dyn Layer>,
    
    input:       Matrix,
    output:      Matrix,
    grad_input:  Matrix,
    grad_output: Matrix,
}

impl Node {
    fn new<L: Layer + 'static>(layer: L) -> Self {
        return Self {
            layer:       Box::new(layer),
            input:       Matrix::empty(),
            output:      Matrix::empty(),
            grad_input:  Matrix::empty(),
            grad_output: Matrix::empty()
        }
    }
}

struct AdjacencyStorage {
    // Implemented as a adjacency matrix
    count: u32,
    edges: Vec<bool>, // count^2
}

impl AdjacencyStorage {
    fn new() -> Self {
        return AdjacencyStorage { count: 0, edges: Vec::new() };
    }

    fn resize(&mut self, count: u32) {
        self.count = count;
        self.edges.resize((count * count) as usize, false);
    }
    
    fn index(&self, a: NodeRef, b: NodeRef) -> u32 {
        return a + b * self.count;
    }

    fn set_connection(&mut self, from: NodeRef, to: NodeRef, connected: bool) {
        let index = self.index(from, to) as usize;
        self.edges[index] = connected;
    }

    fn get_connection(&self, from: NodeRef, to: NodeRef) -> bool {
        return self.edges[self.index(from, to) as usize];
    }

    fn get_connections_from(&self, from: NodeRef) -> Vec<NodeRef> {
        let mut connections: Vec<NodeRef> = Vec::new();

        for i in 0..self.count {
            if self.get_connection(from, i) {
                connections.push(i);
            }
        }

        return connections;
    }

    fn get_connections_to(&self, to: NodeRef) -> Vec<NodeRef> {
        let mut connections: Vec<NodeRef> = Vec::new();

        for i in 0..self.count {
            if self.get_connection(i, to) {
                connections.push(i);
            }
        }

        return connections;
    }

    fn get_in_degree(&self, node: NodeRef) -> u32 {
        let mut counter = 0u32;

        for i in 0..self.count {
            if self.get_connection(i, node) {
                counter += 1;
            }
        }

        return counter;
    }
}

pub struct Graph {
        learning_rate:  f32,
    pub lr_start:       f32,
    pub lr_end:         f32,
    pub total_steps:    u32,
    pub betta_1:        f32,
    pub betta_2:        f32,
    pub lambda:         f32,
    pub mini_batch:     u32,
    pub clip_threshold: f32,
    
    nodes:    Vec<Node>,
    adjacent: AdjacencyStorage,
    exec:     Option<Vec<NodeRef>>,
    optim:    Option<Vec<AdamWContext>>,
    start:    NodeRef,
    end:      NodeRef,

    step: u32, // total training steps which accumulated gradients
    t:    u32, // steps the optimizer took
}

impl Graph {
    pub fn new() -> Self {
        return Self {
            lr_start:         0.005,
            lr_end:         0.0005,
            total_steps:    1_000_000,
            learning_rate:  0.0,
            betta_1:        0.9,
            betta_2:        0.999,
            lambda:         0.001,
            mini_batch:     1,
            clip_threshold: 1.0,
            
            nodes:    Vec::new(),
            adjacent: AdjacencyStorage::new(),
            exec:     None,
            optim:    None,
            start:    0,
            end:      0,
            t:        1,
            step:     0,
        };
    }

    #[must_use]
    pub fn push_layer<L: Layer + 'static>(&mut self, layer: L) -> NodeRef {
        self.nodes.push(Node::new(layer));
        let n = self.nodes.len() - 1;
        self.adjacent.resize((n + 1) as u32);
        return n as NodeRef;
    }

    fn build_optimizer(&mut self) {
        self.optim = Some(Vec::new());

        let Some(optimizer) = &mut self.optim else {panic!("Unexpected issue with the optimizer!");};
        
        for node in &mut self.nodes {
            for shape in node.layer.get_param_shapes() {
                optimizer.push(AdamWContext::new(shape));
            }
        }
    }

    pub fn build_exec(&mut self) {
        if self.exec.is_some() {return;}

        let n = self.nodes.len();

        let mut in_degrees: Vec<u32> = vec![0; n];
        let mut queu: Vec<NodeRef>   = Vec::new();
        let mut exec: Vec<NodeRef>   = Vec::new();

        for i in 0..n {
            in_degrees[i] = self.adjacent.get_in_degree(i as NodeRef);
            if in_degrees[i] == 0 {
                queu.push(i as NodeRef);
            }
        }

        while !queu.is_empty() {
            let top = queu.pop().expect("Unexpected end of queue reached!");
            exec.push(top);

            for i in 0..n {
                if self.adjacent.get_connection(top, i as NodeRef) {
                    in_degrees[i] -= 1;
                    if in_degrees[i] == 0 {
                        queu.push(i as NodeRef);
                    }
                }
            }
        }

        self.exec = Some(exec);
        self.build_optimizer();
    }

    pub fn connect_layers(&mut self, a: NodeRef, b: NodeRef) {
        assert!((a as usize) < self.nodes.len());
        assert!((b as usize) < self.nodes.len());
        assert!(a != b);

        self.adjacent.set_connection(a, b, true);

        self.exec  = None;
        self.optim = None;
    }

    pub fn connect_layer_list(&mut self, list: &[NodeRef]) {
        for i in 0..(list.len()-1) {
            self.connect_layers(list[i], list[i + 1]);
        }
    }

    pub fn set_start_end(&mut self, start: NodeRef, end: NodeRef) {
        assert!(start != end);
        assert!((start as usize) < self.nodes.len());
        assert!((end   as usize) < self.nodes.len());

        self.start = start;
        self.end   = end;
    }

    fn forward_layer(&mut self, i: usize, train: bool) {
        let connections = self.adjacent.get_connections_to(i as u32);

        let (left, node, right) = split_vec(&mut self.nodes, i);

        for c in &connections {
            let input_node = if (*c as usize) < i {
                &mut left[*c as usize]
            } else {
                &mut right[*c as usize - i - 1]
            };
            
            if *c == *connections.first().unwrap() {
                node.input.assign_to(&input_node.output);
            } else {
                node.input += &input_node.output;
            }
        }

        node.layer.forward(&node.input, &mut node.output, train);
    }

    fn forward_helper(&mut self, input: &mut Matrix, train: bool) {
        let Some(execution) = &self.exec.clone() else {
            panic!("Build the exec tree first using .build_exec()")
        };

        let n = self.nodes.len();
        
        if n < 2 {panic!("Too few layers are present; There must be at least 2: an input and output layer");}

        for i in execution {
            if *i == self.start {
                let node = &mut self.nodes[*i as usize];
                node.input.assign_to(input);
                node.layer.forward(&node.input, &mut node.output, train);
            } else {
                self.forward_layer(*i as usize, train);
            }
        }

        input.clone_from(&self.nodes[self.end as usize].output);
    }

    pub fn forward(&mut self, input: &mut Matrix) {
        self.forward_helper(input, false);
    }

    fn backward(&mut self, gradient: &Matrix) {
        let Some(execution) = &self.exec else {
            panic!("Build the exec tree first using .build_exec()")
        };

        let n = self.nodes.len();

        if n < 2 {panic!("Too few layers are present; There must be at least 2: an input and output layer");}

        for i in execution.iter().rev() {
            let mut gradient_input_sum = Matrix::empty();
            
            if *i == self.end {
                gradient_input_sum.clone_from(gradient);
            } else {                
                for c in self.adjacent.get_connections_from(*i) {
                    if gradient_input_sum.is_empty() {
                        gradient_input_sum = self.nodes[c as usize].grad_output.clone();
                    } else {
                        gradient_input_sum += &self.nodes[c as usize].grad_output;
                    }
                }
            }

            let mut node = &mut self.nodes[(*i) as usize];
            node.grad_input = gradient_input_sum;

            node.layer.backward(&node.input, &node.grad_input, &mut node.grad_output);
        }
    }

    fn average_gradients(&mut self) {
        for node in &mut self.nodes {
            for (_, gradient) in node.layer.get_params() {
                *gradient /= self.mini_batch as f32;
            }
        }
    }

    fn clip_gradients(&mut self) {
        let mut norm_sqr = 0.0;

        for node in &mut self.nodes {
            for (_, gradient) in node.layer.get_params() {
                for v in gradient.iter() {
                    norm_sqr += (*v as f64).powi(2);
                }
            }
        }

        let norm = norm_sqr.sqrt();
        
        if norm > self.clip_threshold as f64 {
            let factor = (self.clip_threshold as f64 / norm) as f32;

            for node in &mut self.nodes {
                for (_, gradient) in node.layer.get_params() {
                    *gradient *= factor;
                }
            }
        }
    }

    fn zero_gradients(&mut self) {
        for node in &mut self.nodes {
            for (_, gradient) in node.layer.get_params() {
                gradient.zero();
            }
        }
    }

    fn apply_optimizer(&mut self) {
        let mut index: usize = 0;

        let Some(optimizer) = &mut self.optim else {panic!("Tried to retrieve optimizer from None!");};

        let betta_1_t = 1.0 - self.betta_1.powi(self.t as i32);
        let betta_2_t = 1.0 - self.betta_2.powi(self.t as i32);

        for layer in &mut self.nodes.iter_mut().map(|l| &mut l.layer) {
            for (weights, gradient) in layer.get_params() {
                optimizer[index].update(weights, gradient, self.learning_rate, self.lambda, self.betta_1, self.betta_2, betta_1_t, betta_2_t);
                index += 1;
            }
        }

        self.t += 1;
    }

    fn apply_gradients(&mut self) {
        self.average_gradients();
        self.clip_gradients();
        self.apply_optimizer();
        self.zero_gradients();
    }

    pub fn train(&mut self, input: &Matrix, output: &Matrix) {
        let progress = (self.step as f32 / self.total_steps as f32).clamp(0.0, 1.0);
        self.learning_rate = self.lr_end + 0.5 * (self.lr_start - self.lr_end) * (1.0 + (std::f32::consts::PI * progress).cos());
        
        let mut actv = input.clone();

        self.forward_helper(&mut actv, true);

        actv -= output; // MSE derivative

        actv /= actv.elem_count() as f32;
        
        self.backward(&actv);

        self.step += 1;

        if (self.step % self.mini_batch) == 0 {
            self.apply_gradients();
        }
    }
}