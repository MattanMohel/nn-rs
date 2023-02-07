use std::ops::{Index, IndexMut};

use matrixmultiply::sgemm;
use rand::{Rng, distributions::Uniform, prelude::Distribution};
use serde::{Serialize, Deserialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct Mat {
    buf: Vec<f32>,
    row: usize,
    col: usize
}

#[derive(Clone, Copy)]
pub struct Transpose<'a> {
    view: &'a Mat
}

pub trait MatBase: 
where
    Self: Clone + Index<(usize, usize), Output=f32> 
{
    fn data(&self) -> &Vec<f32>;
    fn shape(&self) -> (usize, usize);
    fn row_stride(&self) -> isize;
    fn col_stride(&self) -> isize;

    fn row(&self) -> usize {
        self.shape().0
    }

    fn col(&self) -> usize {
        self.shape().1
    }

    fn to_index(&self, i: usize) -> (usize, usize) {
        (i / self.col(), i % self.col())    
    }

    fn mul_to<M: MatBase>(&self, rhs: &M, out: &mut Mat) {
        assert_eq!(self.col(), rhs.row());
        assert_eq!(self.row(), out.row());
        assert_eq!(rhs.col(),  out.col());

        out.buf.fill(0.0);

        unsafe {
            sgemm(
                self.row(), // m dimension
                self.col(), // k dimension
                rhs.col(),  // n dimension
                1.0,
                self.data().as_ptr(), // m x k matrix
                self.row_stride(),   // row stride
                self.col_stride(),   // col stride
                rhs.data().as_ptr(),  // k x n matrix
                rhs.row_stride(),    // row stride
                rhs.col_stride(),    // col stride
                1_f32,
                out.buf.as_mut_ptr(), // m x n buffer 
                out.row_stride(),     // row stride
                out.col_stride()      // col stride
            );
        }
    }
 
    fn add_to<M: MatBase>(&self, rhs: &M, out: &mut Mat) {
        assert_eq!(self.shape(), rhs.shape());
        assert_eq!(self.shape(), out.shape());

        for (i, n) in out.buf.iter_mut().enumerate() {
            *n = self[self.to_index(i)] + rhs[rhs.to_index(i)];
        }
    }

    fn sub_to<M: MatBase>(&self, rhs: &M, out: &mut Mat) {
        assert_eq!(self.shape(), rhs.shape());
        assert_eq!(self.shape(), out.shape());

        for (i, n) in out.buf.iter_mut().enumerate() {
            *n = self[self.to_index(i)] - rhs[rhs.to_index(i)];
        }
    }
}

impl MatBase for Mat {
    fn shape(&self) -> (usize, usize) {
        (self.row, self.col)
    }

    fn row_stride(&self) -> isize {
        self.col as isize
    }

    fn col_stride(&self) -> isize {
        1
    }
    
    fn data(&self) -> &Vec<f32> {
        &self.buf
    }
}

impl Index<(usize, usize)> for Mat {
    type Output = f32;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        &self.buf[row * self.col() + col]
    }
}

impl IndexMut<(usize, usize)> for Mat {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        let index = row * self.col() + col;
        &mut self.buf[index]
    }
}

impl<'a> MatBase for Transpose<'a> {
    fn shape(&self) -> (usize, usize) {
        (self.view.col, self.view.row)
    }

    fn row_stride(&self) -> isize {
        1
    }

    fn col_stride(&self) -> isize {
        self.row() as isize
    }
    
    fn data(&self) -> &Vec<f32> {
        &self.view.buf
    }
}

impl<'a> Index<(usize, usize)> for Transpose<'a> {
    type Output = f32;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        &self.view.buf[row + col * self.row()]
    }
}

impl Mat {
    pub fn from_vec((row, col): (usize, usize), buf: Vec<f32>) -> Self {
        Self {
            buf,
            row,
            col
        }
    }

    pub fn from_arr<const R: usize, const C: usize>(arr: [[f32; C]; R]) -> Self {
        let buf = arr
            .iter()
            .copied()
            .flatten()
            .collect();

        Self {
            buf,
            row: R,
            col: C
        }
    }

    pub fn from_fn<F>((row, col): (usize, usize), f: F) -> Self 
    where 
        F: Fn((f32, f32)) -> f32 
    {
        let buf = (0..row*col)
            .map(|i| f(
                ((i / col) as f32, 
                 (i % col) as f32))
            )
            .collect();

        Self {
            buf,
            row,
            col
        }
    }

    pub fn zeros((row, col): (usize, usize)) -> Self {
        Self {
            buf: vec![0.0; row*col],
            row,
            col
        }
    }

    pub fn filled((row, col): (usize, usize), value: f32) -> Self {
        Self {
            buf: vec![value; row*col],
            row,
            col
        }
    }

    pub fn random((row, col): (usize, usize), min: f32, max: f32) -> Self {
        let uniform = Uniform::from(min..max);
        let mut rng = rand::thread_rng();
        
        let buf = (0..row*col)
            .map(|_| uniform.sample(&mut rng))
            .collect();

        Self {
            buf,
            row,
            col
        }
    }

    pub fn transposed(&self) -> Transpose<'_> {
        Transpose { 
            view: self, 
        }
    }

    pub fn add_assign<M: MatBase>(&mut self, rhs: &M) {
        assert_eq!(self.shape(), rhs.shape());

        for (i, n) in self.buf.iter_mut().enumerate() {
            *n += rhs[rhs.to_index(i)];
        }
    }

    pub fn sub_assign<M: MatBase>(&mut self, rhs: &M) {
        assert_eq!(self.shape(), rhs.shape());

        for (i, n) in self.buf.iter_mut().enumerate() {
            *n -= rhs[rhs.to_index(i)];
        }
    }

    pub fn elem_mul_assign<M: MatBase>(&mut self, rhs: &M) {
        assert_eq!(self.shape(), rhs.shape());

        for (i, n) in self.buf.iter_mut().enumerate() {
            *n *= rhs[rhs.to_index(i)];
        }
    }

    pub fn map<F>(&self, f: F) -> Mat 
    where 
        F: Fn(f32) -> f32 
    {
        let buf = self.buf
            .iter()
            .map(|n| f(*n))
            .collect();

        Mat::from_vec(self.shape(), buf)
    }

    pub fn map_assign<F>(&mut self, f: F) 
    where 
        F: Fn(&mut f32) 
    {
        for n in self.buf.iter_mut() {
            f(n);
        }
    }

    pub fn scale(&self, scalar: f32) -> Mat {
        let buf = self.buf
            .iter()
            .map(|n| *n * scalar)
            .collect();

        Mat::from_vec(self.shape(), buf)
    }

    pub fn fill(&mut self, value: f32) {
        for n in self.buf.iter_mut() {
            *n = value;
        }
    }

    pub fn max_index(&self) -> (usize, usize) {
        let res = self.buf
            .iter()
            .enumerate()
            .reduce(|max, n| if max.1 >= n.1 { max } else { n });

        match res {
            Some((i, _)) => self.to_index(i),
            None => panic!()
        }
    }
}