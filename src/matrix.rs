use std::{ops::{Add, Div, Index, IndexMut, Mul, Sub}, sync::mpsc::Iter};

use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    values: Box<[f32]>
}

fn mat_index(row: usize, col: usize, rows: usize) -> usize {
    return row + col * rows;
}

impl Index<usize> for Matrix {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.rows * self.cols);
        return &self.values[index];
    }
}

impl IndexMut<usize> for Matrix {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        assert!(index < self.rows * self.cols);
        return &mut self.values[index];
    }
}

impl Index<(usize, usize)> for Matrix {
    type Output = f32;
    
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        assert!(row < self.rows);
        assert!(col < self.cols);
        return &self.values[mat_index(row, col, self.rows)];
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        assert!(row < self.rows);
        assert!(col < self.cols);
        return &mut self.values[mat_index(row, col, self.rows)];
    }
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        return Self {
            rows: rows,
            cols: cols,
            values: vec![0.0; rows * cols].into_boxed_slice()
        };
    }

    pub fn new_from(rows: usize, cols: usize, values: Vec<f32>) -> Self {
        if rows * cols != values.len() {
            return Self::new(rows, cols);
        }

        return Self {
            rows: rows,
            cols: cols,
            values: values.into_boxed_slice(),
        };
    }

    pub fn set_values(&mut self, values: &[f32]) {
        assert!(self.rows * self.cols == values.len());
        self.values.clone_from_slice(values);
    }

    pub fn ensure_shape(&mut self, rows: usize, cols: usize) {
        if self.rows == rows && self.cols == cols {return;}
        *self = Matrix::new(rows, cols);
    }

    pub fn transposed(&self) -> Self {
        let mut result = Matrix::new(self.cols, self.rows);
        let src = &self.values;
        let dst = &mut result.values;
        let rows = self.rows;
        let cols = self.cols;

        for j in 0..cols {
            let src_col = &src[j * rows..(j + 1) * rows];
            for (i, &val) in src_col.iter().enumerate() {
                dst[j + i * cols] = val;
            }
        }

        return result;
    }

    pub fn fill_random(&mut self, min: f32, max: f32) {
        for i in 0..self.rows*self.cols {
            self[i] = rand::random_range(min..max);
        }
    }

    pub fn zero(&mut self) {
        self.values.fill(0.0);
    }

    pub fn get_rows (&self) -> usize {return self.rows;}
    pub fn get_cols (&self) -> usize {return self.cols;}
    pub fn get_shape(&self) -> (usize, usize) {return (self.rows, self.cols);}

    #[inline(always)]
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    #[inline(always)]
    pub fn values_mut(&mut self) -> &mut [f32] {
        &mut self.values
    }

    pub fn add_into(dest: &mut Matrix, src: &Matrix) {
        assert_eq!(dest.get_shape(), src.get_shape());

        for (d, s) in dest.values.iter_mut().zip(src.values.iter()) {
            *d += *s;
        }
    }

    pub fn sub_into(dest: &mut Matrix, src: &Matrix) {
        assert_eq!(dest.get_shape(), src.get_shape());

        for (d, s) in dest.values.iter_mut().zip(src.values.iter()) {
            *d -= *s;
        }
    }

    pub fn mul_into(a: &Matrix, b: &Matrix, out: &mut Matrix, zero: bool) {
        assert_eq!(a.cols, b.rows);

        out.ensure_shape(a.rows, b.cols);

        let m = a.rows;
        let k_dim = a.cols;
        let n = b.cols;

        if m == 0 || n == 0 {
            return;
        }

        let a_slice = &a.values;
        let b_slice = &b.values;

        // When workload is large enough, parallelize using Rayon
        const PARALLEL_OPS_THRESHOLD: usize = 16_384;
        let total_ops = m * n * k_dim;

        if total_ops >= PARALLEL_OPS_THRESHOLD && n >= 2 {
            // Parallelize across columns of output
            out.values
                .par_chunks_exact_mut(m)
                .enumerate()
                .for_each(|(k, out_col)| {
                    let b_col = &b_slice[k * k_dim..(k + 1) * k_dim];
                    compute_column(out_col, a_slice, b_col, m, k_dim, zero);
                });
        } else if total_ops >= PARALLEL_OPS_THRESHOLD && n == 1 && m >= 64 {
            // Parallelize across row chunks when n == 1 but m * k is large
            let out_col = &mut out.values[..m];
            if zero {
                out_col.fill(0.0);
            }
            let chunk_size = (m / rayon::current_num_threads()).max(32);
            out_col
                .par_chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    let row_offset = chunk_idx * chunk_size;
                    let chunk_len = out_chunk.len();
                    for j in 0..k_dim {
                        let b_val = b_slice[j];
                        if b_val == 0.0 {
                            continue;
                        }
                        let a_sub_col = &a_slice[j * m + row_offset..j * m + row_offset + chunk_len];
                        for (out_elem, &a_elem) in out_chunk.iter_mut().zip(a_sub_col.iter()) {
                            *out_elem += a_elem * b_val;
                        }
                    }
                });
        } else {
            // Sequential column-by-column computation
            out.values
                .chunks_exact_mut(m)
                .enumerate()
                .for_each(|(k, out_col)| {
                    let b_col = &b_slice[k * k_dim..(k + 1) * k_dim];
                    compute_column(out_col, a_slice, b_col, m, k_dim, zero);
                });
        }
    }

    pub fn sum(&self) -> f32 {
        return self.values.iter().sum();
    }
}

#[inline(always)]
fn compute_column(
    out_col: &mut [f32],
    a_values: &[f32],
    b_col: &[f32],
    m: usize,
    k_dim: usize,
    zero: bool,
) {
    if zero {
        out_col.fill(0.0);
    }
    for j in 0..k_dim {
        let b_val = b_col[j];
        if b_val == 0.0 {
            continue;
        }
        let a_col = &a_values[j * m..(j + 1) * m];
        for (out_elem, &a_elem) in out_col.iter_mut().zip(a_col.iter()) {
            *out_elem += a_elem * b_val;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_mul(a: &Matrix, b: &Matrix) -> Matrix {
        let mut out = Matrix::new(a.rows, b.cols);
        for i in 0..a.rows {
            for k in 0..b.cols {
                for j in 0..a.cols {
                    out[(i, k)] += a[(i, j)] * b[(j, k)];
                }
            }
        }
        out
    }

    #[test]
    fn test_mul_small() {
        let mut a = Matrix::new(3, 4);
        let mut b = Matrix::new(4, 2);
        a.fill_random(-1.0, 1.0);
        b.fill_random(-1.0, 1.0);

        let mut out = Matrix::new(0, 0);
        Matrix::mul_into(&a, &b, &mut out, true);
        let expected = naive_mul(&a, &b);

        for i in 0..out.rows * out.cols {
            assert!((out[i] - expected[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_mul_parallel_multi_col() {
        // Shapes that exceed PARALLEL_OPS_THRESHOLD (16384) with n >= 2
        let mut a = Matrix::new(128, 64);
        let mut b = Matrix::new(64, 32);
        a.fill_random(-1.0, 1.0);
        b.fill_random(-1.0, 1.0);

        let mut out = Matrix::new(0, 0);
        Matrix::mul_into(&a, &b, &mut out, true);
        let expected = naive_mul(&a, &b);

        for i in 0..out.rows * out.cols {
            assert!((out[i] - expected[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn test_mul_parallel_single_col() {
        // Shapes that exceed PARALLEL_OPS_THRESHOLD with n == 1
        let mut a = Matrix::new(256, 128);
        let mut b = Matrix::new(128, 1);
        a.fill_random(-1.0, 1.0);
        b.fill_random(-1.0, 1.0);

        let mut out = Matrix::new(0, 0);
        Matrix::mul_into(&a, &b, &mut out, true);
        let expected = naive_mul(&a, &b);

        for i in 0..out.rows * out.cols {
            assert!((out[i] - expected[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn test_transposed() {
        let mut a = Matrix::new(5, 7);
        a.fill_random(-1.0, 1.0);
        let at = a.transposed();
        assert_eq!(at.rows, 7);
        assert_eq!(at.cols, 5);
        for i in 0..5 {
            for j in 0..7 {
                assert_eq!(a[(i, j)], at[(j, i)]);
            }
        }
    }
}