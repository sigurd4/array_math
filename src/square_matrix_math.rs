use std::{f64::EPSILON, ops::{AddAssign, Div, DivAssign, Mul, SubAssign}};

use array__ops::{max_len, Array2dOps, ArrayOps};
use num::{complex::ComplexFloat, Float, One, Signed, Zero};

use crate::MatrixMath;

#[const_trait]
pub trait SquareMatrixMath<T, const N: usize>: ~const MatrixMath<T, N, N>
{
    fn inv_matrix(&self) -> Option<[[T; N]; N]>
    where
        T: Signed + PartialOrd + One + Zero + Copy + SubAssign + DivAssign + AddAssign,
        [(); max_len(N, N)]:;
        
    fn solve_matrix(&self, b: &[T; N]) -> [T; N]
    where
        T: Copy + Signed + Zero + One + PartialOrd + AddAssign + SubAssign + DivAssign,
        [(); max_len(N, N)]:;
        
    fn hessenberg_matrix(&self) -> [[T; N]; N]
    where
        T: ComplexFloat + AddAssign + SubAssign + DivAssign<T::Real> + Div<T::Real, Output = T> + Mul<T::Real, Output = T> + Copy;
    fn hessenbergp_matrix(&self) -> ([[T; N]; N], [[T; N]; N])
    where
        T: ComplexFloat + AddAssign + SubAssign + DivAssign<T::Real> + Div<T::Real, Output = T> + Mul<T::Real, Output = T> + Copy;

    fn eigenvalues(&self) -> Option<[T; N]>
    where
        T: ComplexFloat + AddAssign + SubAssign + DivAssign<T::Real> + Div<T::Real, Output = T> + Mul<T::Real, Output = T> + Copy;

}

impl<T, const N: usize> SquareMatrixMath<T, N> for [[T; N]; N]
{
    fn inv_matrix(&self) -> Option<[[T; N]; N]>
    where
        T: Signed + PartialOrd + One + Zero + Copy + SubAssign + DivAssign + AddAssign,
        [(); max_len(N, N)]:
    {
        let (p, l, u) = self.lup_matrix();
        
        let mut n = 0;
        while n != N
        {
            if l[n][n].is_zero()
            {
                return None
            }
            if u[n][n].is_zero()
            {
                return None
            }
            n += 1;
        }
    
        let mut ia = [[T::zero(); N]; N];
    
        let mut j = 0;
        while j < N
        {
            let mut i = 0;
            while i < N
            {
                ia[i][j] = p[i][j];
    
                let mut k = 0;
                while k != i
                {
                    ia[i][j] -= l[i][k]*ia[k][j];
                    k += 1;
                }
    
                i += 1;
            }
    
            let mut i = N;
            while i != 0
            {
                i -= 1;
    
                let mut k = i + 1;
                while k != N
                {
                    ia[i][j] -= u[i][k]*ia[k][j];
                    k += 1;
                }
    
                ia[i][j] /= u[i][i];
            }
    
            j += 1;
        }
    
        Some(ia)
    }

    fn solve_matrix(&self, b: &[T; N]) -> [T; N]
    where
        T: Copy + Signed + Zero + One + PartialOrd + AddAssign + SubAssign + DivAssign,
        [(); max_len(N, N)]:
    {
        let (l, u, p) = self.lup_matrix();
    
        let [bp] = core::array::from_ref(b).mul_matrix(&p);
    
        let mut x = bp;
        
        let mut m = 0;
        while m != N
        {
            let mut k = 0;
            while k != m
            {
                x[m] -= l[m][k] * x[k];
                k += 1;
            }
            
            m += 1;
        }
    
        let mut m = N;
        while m != 0
        {
            m -= 1;
    
            let mut k = m + 1;
            while k != N
            {
                x[m] -= u[m][k] * x[k];
                k += 1;
            }
    
            x[m] /= u[m][m];
        }
    
        x
    }

    fn hessenberg_matrix(&self) -> [[T; N]; N]
    where
        T: ComplexFloat + AddAssign + SubAssign + DivAssign<T::Real> + Div<T::Real, Output = T> + Mul<T::Real, Output = T> + Copy
    {
        let two = T::Real::one() + T::Real::one();
        let mut a = *self;
        for i in 0..(N - 1)
        {
            let mut s = a[i + 1][i];
            if !s.is_zero()
            {
                s /= s.abs();
            }
            else
            {
                s = T::one();
            }
            let mut x_abs_sqr = T::zero();
            for k in (i + 1)..N
            {
                x_abs_sqr += a[k][i].conj()*a[k][i];
            }
            if x_abs_sqr.is_zero()
            {
                continue
            }
            let x1 = -s*x_abs_sqr.sqrt();

            let u0 = if !a[i + 1][i].is_zero()
            {
                ((T::one() - a[i + 1][i]/x1)/two).sqrt()
            }
            else
            {
                if x1.is_zero()
                {
                    continue;
                }
                else
                {
                    (T::one()/two).sqrt()
                }
            };
            let mut u = [T::zero(); N];
            u[i + 1] = u0;
            for k in (i + 2)..N
            {
                u[k] = a[k][i]/(-u0*x1*two);
            }

            let mut h = <[[T; N]; N]>::identity_matrix();
            for j in (i + 1)..N
            {
                for k in (i + 1)..N
                {
                    h[j][k] -= u[j]*u[k].conj()*two
                }
            }
            a = h.transpose_conj().mul_matrix(&a.mul_matrix(&h))
        }
        for i in 0..(N - 2)
        {
            for k in (i + 2)..N
            {
                a[k][i] = T::zero();
            }
        }
        a
    }
    fn hessenbergp_matrix(&self) -> ([[T; N]; N], [[T; N]; N])
    where
        T: ComplexFloat + AddAssign + SubAssign + DivAssign<T::Real> + Div<T::Real, Output = T> + Mul<T::Real, Output = T> + Copy
    {
        let p = self.rpivot_matrix_complex();
        let pa = p.mul_matrix(self);
    
        let h = pa.hessenberg_matrix();   
    
        (h, p)
    }
    
    fn eigenvalues(&self) -> Option<[T; N]>
    where
        T: ComplexFloat + AddAssign + SubAssign + DivAssign<T::Real> + Div<T::Real, Output = T> + Mul<T::Real, Output = T> + Copy
    {
        let mut a = self.hessenberg_matrix();

        let mut is_done = false;
        for _ in 0..1024
        {
            let (q, r, p) = a.qrp_matrix();
            a = r.mul_matrix(&p.transpose()).mul_matrix(&q);
            is_done = true;
            for k in 0..(N - 1)
            {
                for j in (k + 1)..N
                {
                    if Float::is_normal(a[j][k].abs())
                    {
                        is_done = false;
                        break;
                    }
                }
                if !is_done
                {
                    break
                }
            }
            if is_done
            {
                break;
            }
        }

        if !is_done
        {
            return None
        }
        Some(ArrayOps::fill(|i| a[i][i]))
    }
}