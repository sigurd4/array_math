use std::ops::{SubAssign, DivAssign, AddAssign};

use array__ops::max_len;
use num::{One, Signed, Zero};

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
}