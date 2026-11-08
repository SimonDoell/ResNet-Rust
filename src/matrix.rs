use std::ops::{Add, Sub, Mul, Div, AddAssign, SubAssign, MulAssign, DivAssign, Index, IndexMut, Neg};
use std::iter::{Iterator};

#[derive(Debug, Clone)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    values: Box<[f32]>,
}

impl Index<usize> for Matrix {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.elem_count());
        return &self.values[index];
    }
}

impl IndexMut<usize> for Matrix {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        assert!(index < self.elem_count());
        return &mut self.values[index];
    }
}

impl Index<(usize, usize)> for Matrix {
    type Output = f32;
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        assert!(row < self.rows);
        assert!(col < self.cols);
        return &self.values[Matrix::index(row, col, self.rows)];
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        assert!(row < self.rows);
        assert!(col < self.cols);
        return &mut self.values[Matrix::index(row, col, self.rows)];
    }
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        return Matrix { rows: rows, cols: cols, values: vec![0.0; rows * cols].into_boxed_slice() };
    }

    pub fn empty() -> Self {
        return Matrix { rows: 0, cols: 0, values: Box::new([0.0; 0]) };
    }

    fn index(row: usize, col: usize, rows: usize) -> usize {
        return row + col * rows; // Column-major!
    }

    pub fn shape(&self) -> (usize, usize) {
        return (self.rows, self.cols);
    }

    pub fn elem_count(&self) -> usize {
        return self.rows * self.cols;
    }

    pub fn resize(&mut self, shape: (usize, usize)) {
        self.rows = shape.0;
        self.cols = shape.1;
        self.values = vec![0.0; self.rows * self.cols].into_boxed_slice();
    }

    pub fn ensure_shape(&mut self, shape: (usize, usize)) {
        if self.shape() == shape {return;}
        self.resize(shape);
    }

    pub fn sum(&self) -> f32 {return self.values.iter().sum();}

    pub fn values_mut(&mut self) -> &mut [f32] {return &mut self.values;}
}

impl AddAssign<&Matrix> for Matrix {
    fn add_assign(&mut self, rhs: &Matrix) {
        assert!(self.shape() == rhs.shape());
        for i in 0..self.elem_count() {
            self[i] += rhs[i];
        }
    }
}

impl AddAssign<Matrix> for Matrix {
    fn add_assign(&mut self, rhs: Matrix) {
        *self += &rhs;
    }
}

impl Add<&Matrix> for Matrix {
    type Output = Matrix;
    fn add(mut self, rhs: &Matrix) -> Self::Output {self += rhs; self}
}

impl Add<Matrix> for &Matrix {
    type Output = Matrix;
    fn add(self, mut rhs: Matrix) -> Self::Output {rhs += self; rhs}
}

impl Add<Matrix> for Matrix {
    type Output = Matrix;
    fn add(self, rhs: Matrix) -> Self::Output {self + &rhs}
}

impl Add<&Matrix> for &Matrix {
    type Output = Matrix;
    fn add(self, rhs: &Matrix) -> Self::Output {
        let mut result = self.clone();
        result += rhs;
        return result;
    }
}

impl SubAssign<&Matrix> for Matrix {
    fn sub_assign(&mut self, rhs: &Matrix) {
        assert!(self.shape() == rhs.shape());
        for i in 0..self.elem_count() {
            self[i] -= rhs[i];
        }
    }
}

impl SubAssign<Matrix> for Matrix {
    fn sub_assign(&mut self, rhs: Matrix) {
        *self += &rhs;
    }
}

impl Sub<&Matrix> for Matrix {
    type Output = Matrix;
    fn sub(mut self, rhs: &Matrix) -> Self::Output {self += rhs; self}
}

impl Sub<Matrix> for &Matrix {
    type Output = Matrix;
    fn sub(self, mut rhs: Matrix) -> Self::Output {rhs += self; rhs}
}

impl Sub<Matrix> for Matrix {
    type Output = Matrix;
    fn sub(self, rhs: Matrix) -> Self::Output {self + &rhs}
}

impl Sub<&Matrix> for &Matrix {
    type Output = Matrix;
    fn sub(self, rhs: &Matrix) -> Self::Output {
        let mut result = self.clone();
        result += rhs;
        return result;
    }
}

impl MulAssign<f32> for Matrix {
    fn mul_assign(&mut self, rhs: f32) {
        for i in 0..self.elem_count() {self[i] *= rhs;}
    }
}

impl Mul<f32> for Matrix {
    type Output = Matrix;
    fn mul(mut self, rhs: f32) -> Self::Output {self *= rhs; self}
}

impl Mul<f32> for &Matrix {
    type Output = Matrix;
    fn mul(self, rhs: f32) -> Self::Output {self.clone() * rhs}
}

impl Mul<Matrix> for f32 {
    type Output = Matrix;
    fn mul(self, rhs: Matrix) -> Self::Output {rhs * self}
}

impl Mul<&Matrix> for f32 {
    type Output = Matrix;
    fn mul(self, rhs: &Matrix) -> Self::Output {rhs * self}
}

impl DivAssign<f32> for Matrix {
    fn div_assign(&mut self, rhs: f32) {
        for v in self.values_mut() {*v /= rhs;}
    }
}

impl Div<f32> for Matrix {
    type Output = Matrix;
    fn div(mut self, rhs: f32) -> Self::Output {self *= rhs; self}
}

impl Div<f32> for &Matrix {
    type Output = Matrix;
    fn div(self, rhs: f32) -> Self::Output {self.clone() * rhs}
}

impl Div<Matrix> for f32 {
    type Output = Matrix;
    fn div(self, rhs: Matrix) -> Self::Output {rhs / self}
}

impl Div<&Matrix> for f32 {
    type Output = Matrix;
    fn div(self, rhs: &Matrix) -> Self::Output {rhs / self}
}

impl Neg for Matrix {
    type Output = Matrix;
    fn neg(mut self) -> Self::Output {
        for i in 0..self.elem_count() {self[i] = -self[i];}
        self
    }
}

impl Neg for &Matrix {
    type Output = Matrix;
    fn neg(self) -> Self::Output {-self.clone()}
}