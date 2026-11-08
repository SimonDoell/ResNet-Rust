use std::ops::{Add, Sub, Mul, Div, AddAssign, SubAssign, MulAssign, DivAssign, Index, IndexMut};
use std::iter::{Iterator};

#[derive(Debug, Clone)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    values: Box<[f32]>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        return Matrix { rows: rows, cols: cols, values: vec![0.0; rows * cols].into_boxed_slice() };
    }

    pub fn empty() -> Self {
        return Matrix { rows: 0, cols: 0, values: Box::new([0.0; 0]) };
    }

    
}