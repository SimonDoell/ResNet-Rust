use std::arch::x86_64::_CMP_NEQ_UQ;

use crate::matrix::*;
use crate::layer::*;

type NodeRef = u32;

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
    nodes:    Vec<Node>,
    adjacent: AdjacencyStorage,
    exec:     Option<Vec<NodeRef>>,
    start:    NodeRef,
    end:      NodeRef
}

impl Graph {
    pub fn new() -> Self {
        return Self { nodes: Vec::new(), adjacent: AdjacencyStorage::new(), exec: None, start: 0, end: 0 };
    }

    #[must_use]
    pub fn push_layer<L: Layer + 'static>(&mut self, layer: L) -> NodeRef {
        self.nodes.push(Node::new(layer));
        let n = self.nodes.len() - 1;
        self.adjacent.resize((n + 1) as u32);
        return n as NodeRef;
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
    }

    pub fn connect_layers(&mut self, a: NodeRef, b: NodeRef) {
        assert!((a as usize) < self.nodes.len());
        assert!((b as usize) < self.nodes.len());
        assert!(a != b);

        self.adjacent.set_connection(a, b, true);

        self.exec = None;
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

    pub fn forward(&mut self, input: &mut Matrix) {
        let Some(execution) = &self.exec else {
            panic!("Build the exec tree first using .build_exec()")
        };

        let n = self.nodes.len();
        
        if n < 2 {panic!("Too few layers are present; There must be at least 2: an input and output layer");}

        for i in execution {
            let mut output_sum = Matrix::empty();

            if *i == self.start {
                output_sum = input.clone();
            } else {
                for c in self.adjacent.get_connections_to(*i) {
                    if output_sum.is_empty() {
                        output_sum = self.nodes[c as usize].output.clone();
                    } else {
                        output_sum += &self.nodes[c as usize].output;
                    }
                }
            }
            
            let mut layer = &mut self.nodes[*i as usize];
            layer.input = output_sum;

            layer.layer.forward(&layer.input, &mut layer.output, false);
        }

        input.clone_from(&self.nodes[self.end as usize].output);
    }
}