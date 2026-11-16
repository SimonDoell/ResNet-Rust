use crate::graph::*;
use crate::matrix::*;
use std::marker::PhantomData;

pub trait Action {
    fn random() -> Self;
    fn from_output(output: &Matrix) -> Self;
}

pub trait State {
    fn to_input(&self) -> Matrix;
}

pub trait Environment {
    type A: Action;
    type S: State;
    
    fn new() -> Self;
    fn state(&mut self) -> Self::S;
    fn step(&mut self, action: Self::A);
    fn reset(&mut self);
}

pub struct ActorCritic<E: Environment> {
    pub mini_batch: u32,
    
    actor:  Graph,
    critic: Graph,
    pub environment: E,
}

impl<E: Environment> ActorCritic<E> {
    pub fn new() -> Self {        
        Self { mini_batch: 1, actor: Graph::new(), critic: Graph::new(), environment: E::new() }
    }
    
    pub fn step() {}
}