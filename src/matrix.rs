use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, RangeBounds, Sub, SubAssign};
use std::iter::{Iterator};

#[derive(Debug, Clone, PartialEq)]
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

    pub fn new_from(rows: usize, cols: usize, values: &[f32]) -> Self {
        assert!(values.len() == rows * cols);
        return Matrix { rows: rows, cols: cols, values: values.to_vec().into_boxed_slice() };
    }

    pub fn new_from_value(rows: usize, cols: usize, value: f32) -> Self {
        return Matrix { rows: rows, cols: cols, values: vec![value; rows * cols].into_boxed_slice() };
    }

    pub fn empty() -> Self {
        return Matrix { rows: 0, cols: 0, values: Box::new([0.0; 0]) };
    }

    pub fn is_empty(&self) -> bool {
        return self.rows == 0 && self.cols == 0;
    }

    pub fn zero(&mut self) {
        for v in &mut self.values {
            *v = 0.0;
        }
    }

    pub fn transposed(&self) -> Matrix {
        let mut result = Matrix::new(self.cols, self.rows);

        for i in 0..self.rows {
            for j in 0..self.cols {
                result[(j, i)] = (*self)[(i, j)];
            }
        }

        return result;
    }

    fn index(row: usize, col: usize, rows: usize) -> usize {
        return row + col * rows; // Column-major!
    }

    pub fn random_uniform<R: rand::distr::uniform::SampleRange<f32> + Clone>(&mut self, range: R) {
        for v in &mut self.values {
            *v = rand::random_range(range.clone());
        }
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

    pub fn mean(&self) -> f32 {return self.sum() / self.elem_count() as f32;}

    pub fn values_mut(&mut self) -> &mut [f32] {return &mut self.values;}

    pub fn get_rows(&self) -> usize {return self.rows;}
    pub fn get_cols(&self) -> usize {return self.cols;}

    pub fn assign_to(&mut self, rhs: &Matrix) {
        if self.shape() != rhs.shape() {
            self.resize(rhs.shape());
        }

        self.values.clone_from(&rhs.values);
    }

    pub fn iter_mut<'l>(&'l mut self) -> std::slice::IterMut<'l, f32> {
        return self.values.iter_mut();
    }

    pub fn iter<'l>(&'l self) -> std::slice::Iter<'l, f32> {
        return self.values.iter();
    }

    pub fn as_slice(&self) -> &[f32] {
        return &self.values;
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        return &mut self.values;
    }

    pub fn add_outer_product(&mut self, a: &Matrix, b: &Matrix) {
        assert!(self.rows == a.elem_count());
        assert!(self.cols == b.elem_count());

        for j in 0..self.cols {
            for i in 0..self.rows {
                self[(i, j)] += a[i] * b[j];
            }
        }
    }

    fn mul_into_nn(a: &Matrix, b: &Matrix, out: &mut Matrix) {
        let m = a.rows;
        let n = b.cols;
        let p = a.cols;
        
        assert_eq!(p, b.rows);
    
        let a_data = &a.values;
        let b_data = &b.values;
        let out_data = &mut out.values;
    
        for k in 0..n {
            for j in 0..p {
                let b_jk = b_data[j + k * p];
    
                let a_offset = j * m;
                let out_offset = k * m;
    
                for i in 0..m {
                    out_data[out_offset + i] +=
                        a_data[a_offset + i] * b_jk;
                }
            }
        }
    }

    pub fn mul_into_nt(a: &Matrix, b: &Matrix, out: &mut Matrix) {
        let m = a.rows;
        let n = b.rows;
        let p = a.cols;

        assert_eq!(p, b.cols);

        let a_data = &a.values;
        let b_data = &b.values;
        let out_data = &mut out.values;

        for k in 0..n {
            let out_offset = k * m;

            for j in 0..p {
                let b_kj = b_data[k + j * n];
                let a_offset = j * m;

                for i in 0..m {
                    out_data[out_offset + i] +=
                        a_data[a_offset + i] * b_kj;
                }
            }
        }
    }

    pub fn mul_into_tn(a: &Matrix, b: &Matrix, out: &mut Matrix) {
        let m = a.cols;
        let n = b.cols;
        let p = a.rows;

        assert_eq!(p, b.rows);

        let a_data = &a.values;
        let b_data = &b.values;
        let out_data = &mut out.values;

        for k in 0..n {
            let b_offset = k * p;
            let out_offset = k * m;

            for i in 0..m {
                let a_offset = i * p;
                let mut sum = 0.0;

                for j in 0..p {
                    sum += a_data[a_offset + j] * b_data[b_offset + j];
                }

                out_data[out_offset + i] += sum;
            }
        }
    }

    pub fn mul_into_tt(a: &Matrix, b: &Matrix, out: &mut Matrix) {
        let m = a.cols;
        let n = b.rows;
        let p = a.rows;

        assert_eq!(p, b.cols);

        let a_data = &a.values;
        let b_data = &b.values;
        let out_data = &mut out.values;

        for k in 0..n {
            let out_offset = k * m;

            for j in 0..p {
                let b_kj = b_data[k + j * n];

                for i in 0..m {
                    out_data[out_offset + i] +=
                        a_data[j + i * p] * b_kj;
                }
            }
        }
    }

    pub fn mul_into(a: &Matrix, b: &Matrix, out: &mut Matrix, a_t: bool, b_t: bool, zero: bool) {
        let a_rows = if a_t {a.get_cols()} else {a.get_rows()};
        let b_cols = if b_t {b.get_rows()} else {b.get_cols()};
        
        if zero {out.zero();}
        out.ensure_shape((a_rows, b_cols));

        match (a_t, b_t) {
            (false, false) => Self::mul_into_nn(a, b, out),
            (true,  false) => Self::mul_into_tn(a, b, out),
            (false, true ) => Self::mul_into_nt(a, b, out),
            (true,  true ) => Self::mul_into_tt(a, b, out),
        }
    }
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
        *self -= &rhs;
    }
}

impl Sub<&Matrix> for Matrix {
    type Output = Matrix;
    fn sub(mut self, rhs: &Matrix) -> Self::Output {self -= rhs; self}
}

impl Sub<Matrix> for &Matrix {
    type Output = Matrix;
    fn sub(self, mut rhs: Matrix) -> Self::Output {
        for i in 0..rhs.elem_count() {
            rhs[i] = self[i] - rhs[i];
        }
        rhs
    }
}

impl Sub<Matrix> for Matrix {
    type Output = Matrix;
    fn sub(self, rhs: Matrix) -> Self::Output {self - &rhs}
}

impl Sub<&Matrix> for &Matrix {
    type Output = Matrix;
    fn sub(self, rhs: &Matrix) -> Self::Output {
        let mut result = self.clone();
        result -= rhs;
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
    fn div(mut self, rhs: f32) -> Self::Output {self /= rhs; self}
}

impl Div<f32> for &Matrix {
    type Output = Matrix;
    fn div(self, rhs: f32) -> Self::Output {self.clone() / rhs}
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