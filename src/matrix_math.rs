use std::{ops::{Mul, AddAssign, MulAssign, SubAssign, DivAssign, Sub, Div, Neg}, mem::MaybeUninit};

use array__ops::{Array2dOps, ArrayOps, min_len, max_len};
use num::{One, Zero};

use crate::{ArrayMath};

#[const_trait]
pub trait MatrixMath<T, const M: usize, const N: usize>: ~const Array2dOps<T, M, N>
{
    fn identity_matrix() -> Self
    where
        T: Zero + One;

    fn mul_matrix<Rhs, const P: usize>(&self, rhs: &Self::Array2d<Rhs, N, P>) -> Self::Array2d<<T as Mul<Rhs>>::Output, M, P>
    where
        T: /*~const*/ Mul<Rhs, Output: /*~const*/ AddAssign + /*~const*/ Zero> + Copy,
        Rhs: Copy;
    
    fn pivot_matrix(&self) -> [[T; M]; M]
    where
        T: Neg<Output = T> + Zero + One + PartialOrd + Copy;
    
    fn lup_matrix(&self) -> ([[T; max_len(M, N)]; M], [[T; N]; max_len(M, N)], [[T; M]; M])
    where
        T: Neg<Output = T> + Zero + One + PartialOrd + Mul<Output = T> + AddAssign + Copy + Sub<Output = T> + Div<Output = T>;
            
    fn lu_matrix(&self) -> ([[T; max_len(M, N)]; M], [[T; N]; max_len(M, N)])
    where
        T: Zero + One + PartialOrd + Mul<Output = T> + Sub<Output = T> + Div<Output = T> + AddAssign + Copy;

    fn det_matrix(&self) -> T
    where
        T: Zero + Neg<Output = T> + One + Copy + AddAssign + PartialOrd + Zero + Mul<Output = T> + MulAssign + Sub<Output = T> + Div<Output = T>,
        [(); max_len(M, N)]:;
    
    fn is_matrix_invertible(&self) -> bool
    where
        T: Zero + Neg<Output = T> + One + Copy + AddAssign + PartialOrd + MulAssign + Sub<Output = T> + Div<Output = T>,
        [(); max_len(M, N)]:;
}

impl<T, const M: usize, const N: usize> MatrixMath<T, M, N> for [[T; N]; M]
{
    fn identity_matrix() -> Self
    where
        T: Zero + One
    {
        ArrayOps::fill(|m| ArrayOps::fill(|n| if m == n
        {
            One::one()
        }
        else
        {
            Zero::zero()
        }))
    }

    fn mul_matrix<Rhs, const P: usize>(&self, rhs: &Self::Array2d<Rhs, N, P>) -> Self::Array2d<<T as Mul<Rhs>>::Output, M, P>
    where
        T: Mul<Rhs, Output: AddAssign + Zero> + Copy,
        Rhs: Copy
    {
        let mut prod: [[<T as Mul<Rhs>>::Output; P]; M] = unsafe {MaybeUninit::assume_init(MaybeUninit::uninit())};
        let mut m = 0;
        while m != M
        {
            let mut p = 0;
            while p != P
            {
                prod[m][p] = Zero::zero();
                let mut n = 0;
                while n != N
                {
                    prod[m][p] += self[m][n]*rhs[n][p];
                    n += 1;
                }
                p += 1;
            }
            m += 1;
        }
    
        prod
    }
    
    fn pivot_matrix(&self) -> [[T; M]; M]
    where
        T: Neg<Output = T> + Zero + One + PartialOrd + Copy
    {
        let mut p = <[[T; M]; M]>::identity_matrix();
        
        let mut n = 0;
        while n < M.min(N)
        {
            let mut row_max = n;
            let mut e_abs_max = if self[n][n] >= T::zero() {self[n][n]} else {-self[n][n]};
            
            let mut m = n + 1;
            while m < M
            {
                let e_abs = if self[m][n] >= T::zero() {self[m][n]} else {-self[m][n]};
                if e_abs > e_abs_max
                {
                    row_max = m;
                    e_abs_max = e_abs;
                }
                m += 1;
            }
    
            if row_max != n
            {
                let p = p.each_mut2().map(|p| p as *mut [T; M]);
                unsafe {core::ptr::swap(p[n], p[row_max])};
            }
    
            n += 1;
        }
    
        p
    }
    
    fn lup_matrix(&self) -> ([[T; max_len(M, N)]; M], [[T; N]; max_len(M, N)], [[T; M]; M])
    where
        T: Neg<Output = T> + Zero + One + PartialOrd + Mul<Output = T> + AddAssign + Copy + Sub<Output = T> + Div<Output = T>
    {
        let mut l = [[Zero::zero(); max_len(M, N)]; M];
        let mut u = [[Zero::zero(); N]; max_len(M, N)];
    
        let p = self.pivot_matrix();
        let pa = p.mul_matrix(self);
    
        for n in 0..N
        {
            if n < M
            {
                l[n][n] = One::one();
            }
    
            for m in 0..(n + 1).min(M)
            {
                let mut s = T::zero();
                for k in 0..m
                {
                    s += u[k][n]*l[m][k];
                }
                u[m][n] = pa[m][n] - s;
            }
            for i in n..M
            {
                let mut s = Zero::zero();
                for k in 0..n
                {
                    s += u[k][n]*l[i][k];
                }
                if !(pa[i][n] - s).is_zero()
                {
                    l[i][n] = (pa[i][n] - s)/u[n][n];
                }
            }
        }
    
        (l, u, p)
    }
            
    fn lu_matrix(&self) -> ([[T; max_len(M, N)]; M], [[T; N]; max_len(M, N)])
    where
        T: Zero + One + PartialOrd + Mul<Output = T> + Sub<Output = T> + Div<Output = T> + AddAssign + Copy
    {
        let mut l = [[Zero::zero(); max_len(M, N)]; M];
        let mut u = [[Zero::zero(); N]; max_len(M, N)];
    
        for n in 0..N
        {
            if n < M
            {
                l[n][n] = One::one();
            }
    
            for m in 0..(n + 1).min(M)
            {
                let mut s = T::zero();
                for k in 0..m
                {
                    s += u[k][n]*l[m][k];
                }
                u[m][n] = self[m][n] - s;
            }
            for i in n..M
            {
                let mut s = Zero::zero();
                for k in 0..n
                {
                    s += u[k][n]*l[i][k];
                }
                if !(self[i][n] - s).is_zero()
                {
                    l[i][n] = (self[i][n] - s)/u[n][n];
                }
            }
        }
    
        (l, u)
    }

    fn det_matrix(&self) -> T
    where
        T: Zero + Neg<Output = T> + One + Copy + AddAssign + PartialOrd + Zero + Mul<Output = T> + MulAssign + Sub<Output = T> + Div<Output = T>,
        [(); max_len(M, N)]:
    {
        let (p, l, u) = self.lup_matrix();
    
        let mut det_abs = One::one();
    
        let mut n = 0;
        while n != min_len(M, N)
        {
            det_abs *= l[n][n];
            det_abs *= u[n][n];
            n += 1;
        }
    
        let mut s = false;
    
        let mut m = 0;
        while m != M
        {
            let mut n = 0;
            while n != N && p[m][n].is_zero()
            {
                n += 1;
            };
            if n != m
            {
                s = !s;
            }
            m += 1;
        }
    
        return if !s {det_abs} else {-det_abs}
    }
    
    fn is_matrix_invertible(&self) -> bool
    where
        T: Zero + Neg<Output = T> + One + Copy + AddAssign + PartialOrd + MulAssign + Sub<Output = T> + Div<Output = T>,
        [(); max_len(M, N)]:
    {
        let (_, l, u) = self.lup_matrix();
        
        let mut n = 0;
        while n != N
        {
            if l[n][n].is_zero()
            {
                return false
            }
            if u[n][n].is_zero()
            {
                return false
            }
            n += 1;
        }
    
        true
    }
}