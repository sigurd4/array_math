use std::{any::Any, ops::{Add, AddAssign, Div, DivAssign, Mul, SubAssign}};

use array__ops::{max_len, Array2dOps, ArrayOps};
use num::{complex::ComplexFloat, Complex, Float, One, Signed, Zero};

use crate::{ArrayMath, MatrixMath};

#[const_trait]
pub trait SquareMatrixMath<T, const N: usize>: ~const MatrixMath<T, N, N>
{
    fn inv_matrix(&self) -> Option<[[T; N]; N]>
    where
        T: Signed + PartialOrd + One + Zero + Copy + SubAssign + DivAssign + AddAssign,
        [(); max_len(N, N)]:;
    fn inv_matrix_complex(&self) -> Option<[[T; N]; N]>
    where
        T: ComplexFloat + SubAssign + DivAssign + AddAssign + Copy,
        [(); max_len(N, N)]:;
        
    fn solve_matrix(&self, b: &[T; N]) -> [T; N]
    where
        T: Signed + Zero + One + PartialOrd + AddAssign + SubAssign + DivAssign + Copy,
        [(); max_len(N, N)]:;
    fn solve_matrix_complex(&self, b: &[T; N]) -> [T; N]
    where
        T: ComplexFloat + AddAssign + SubAssign + DivAssign + Copy,
        [(); max_len(N, N)]:;
        
    fn upper_hessenberg_matrix(&self) -> [[T; N]; N]
    where
        T: ComplexFloat + AddAssign + SubAssign + DivAssign<T::Real> + Div<T::Real, Output = T> + Mul<T::Real, Output = T> + Copy;
    fn hessenbergp_matrix(&self) -> ([[T; N]; N], [[T; N]; N])
    where
        T: ComplexFloat + AddAssign + SubAssign + DivAssign<T::Real> + Div<T::Real, Output = T> + Mul<T::Real, Output = T> + Copy;

    fn eigenvalues(&self) -> [T; N]
    where
        T: ComplexFloat<Real: 'static> + AddAssign + SubAssign + DivAssign<T::Real> + Div<T::Real, Output = T> + Mul<T::Real, Output = T> + Copy + 'static;
    fn eigen(&self) -> ([T; N], [[T; N]; N])
    where
        T: ComplexFloat<Real: 'static> + AddAssign + SubAssign + DivAssign + DivAssign<T::Real> + Add<T::Real, Output = T> + Div<T::Real, Output = T> + Mul<T::Real, Output = T> + Copy + 'static,
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
    
    fn inv_matrix_complex(&self) -> Option<[[T; N]; N]>
    where
        T: ComplexFloat + SubAssign + DivAssign + AddAssign + Copy,
        [(); max_len(N, N)]:
    {
        let (p, l, u) = self.lup_matrix_complex();
        
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
            
            if !x[m].is_zero()
            {
                x[m] /= u[m][m];
            }
        }
    
        x
    }
    fn solve_matrix_complex(&self, b: &[T; N]) -> [T; N]
    where
        T: ComplexFloat + AddAssign + SubAssign + DivAssign + Copy,
        [(); max_len(N, N)]:
    {
        let (l, u, p) = self.lup_matrix_complex();
    
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
    
            if !x[m].is_zero()
            {
                x[m] /= u[m][m];
            }
        }
    
        x
    }

    fn upper_hessenberg_matrix(&self) -> [[T; N]; N]
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
    
        let h = pa.upper_hessenberg_matrix();   
    
        (h, p)
    }
    
    fn eigenvalues(&self) -> [T; N]
    where
        T: ComplexFloat<Real: 'static> + AddAssign + SubAssign + DivAssign<T::Real> + Div<T::Real, Output = T> + Mul<T::Real, Output = T> + Copy + 'static
    {
        let mut t = self.upper_hessenberg_matrix();

        for _ in 0..2048
        {
            let p = t.cpivot_matrix_complex();
            let mut a = t.mul_matrix(&p);
            let mut gamma = {
                if N < 3
                {
                    a[N - 1][N - 1]
                }
                else
                {
                    let theta = T::one();
                    let beta_nm1 = a[N - 1][N - 2];
                    let beta_nm2 = a[N - 2][N - 3];

                    if (beta_nm2*theta).abs() < beta_nm1.abs()
                    {
                        a[N - 1][N - 1]
                    }
                    else
                    {
                        let a_sub2 = <[_; 2]>::fill(|i| <[_; 2]>::fill(|j| a[N - 2 + i][N - 2 + j]));
                        let [gamma1, gamma2] = a_sub2.eigenvalues();
                        
                        let delta = |gamma: T| {
                            (gamma - a[N - 2][N - 2])*(gamma - a[N - 1][N - 1]) - t[N - 2][N - 1]
                        };

                        let sqrt_alpha_nm1_n_beta_nm1 = Float::sqrt(beta_nm1.abs()*a[N - 2][N - 1].abs());

                        let wilk_test = |gamma: T| {
                            let dgamma_alpha_nm1_nm1 = (gamma - a[N - 2][N - 2]).abs();
                            let dgamma_alpha_n_n = (gamma - a[N - 1][N - 1]).abs();

                            dgamma_alpha_n_n <= sqrt_alpha_nm1_n_beta_nm1
                                && sqrt_alpha_nm1_n_beta_nm1 <= dgamma_alpha_nm1_nm1
                                && delta(gamma).abs() < T::Real::epsilon()
                        };

                        if wilk_test(gamma1)
                        {
                            gamma1
                        }
                        else if wilk_test(gamma2)
                        {
                            gamma2
                        }
                        else
                        {
                            T::zero()
                        }
                    }
                }
            };
            if let Some(gamma) = <dyn Any>::downcast_mut::<Complex<T::Real>>(&mut gamma as &mut dyn Any)
            {
                *gamma = *gamma + Complex::new(T::Real::zero(), T::Real::epsilon())
            }
            if gamma.is_finite() && !gamma.is_zero()
            {
                for k in 0..N
                {
                    a[k][k] -= gamma;
                }
            }
            let (q, r) = a.qr_matrix();
            a = r.mul_matrix(&q);
            if gamma.is_finite() && !gamma.is_zero()
            {
                for k in 0..N
                {
                    a[k][k] += gamma;
                }
            }
            a = a.mul_matrix(&p.transpose());
            let mut is_done = true;
            if a.iter().any(|a| a.iter().any(|a| Float::is_nan(a.abs())))
            {
                break;
            }
            t = a;
            'lp:
            for k in 0..(N - 1)
            {
                for j in (k + 1)..N
                {
                    let x = t[j][k].abs();
                    if !(!Float::is_normal(x) || x < T::Real::epsilon())
                    {
                        is_done = false;
                        break 'lp;
                    }
                }
            }
            if is_done
            {
                break;
            }
        }

        /*if !is_done
        {
            let nan = T::from(T::Real::nan()).unwrap();
            return ArrayOps::fill(|_| nan)
        }*/
        ArrayOps::fill(|i| t[i][i])
    }
    
    fn eigen(&self) -> ([T; N], [[T; N]; N])
    where
        T: ComplexFloat<Real: 'static> + AddAssign + SubAssign + DivAssign + DivAssign<T::Real> + Add<T::Real, Output = T> + Div<T::Real, Output = T> + Mul<T::Real, Output = T> + Copy + 'static,
        [(); max_len(N, N)]:
    {
        #[cfg(test)]
        const TEST: bool = true;
        #[cfg(not(test))]
        const TEST: bool = false;
        const EPS: f64 = 0.00001;

        let a = *self; //self.upper_hessenberg_matrix();
        let mut t = a;
        let mut u = <[[T; N]; N]>::identity_matrix();

        for _ in 0..2048
        {
            let p = t.cpivot_matrix_complex();
            let mut a = t.mul_matrix(&p);
            let mut gamma = {
                if N < 3
                {
                    a[N - 1][N - 1]
                }
                else
                {
                    let theta = T::one();
                    let beta_nm1 = a[N - 1][N - 2];
                    let beta_nm2 = a[N - 2][N - 3];

                    if (beta_nm2*theta).abs() < beta_nm1.abs()
                    {
                        a[N - 1][N - 1]
                    }
                    else
                    {
                        let a_sub2 = <[_; 2]>::fill(|i| <[_; 2]>::fill(|j| a[N - 2 + i][N - 2 + j]));
                        let [gamma1, gamma2] = a_sub2.eigenvalues();
                        
                        let delta = |gamma: T| {
                            (gamma - a[N - 2][N - 2])*(gamma - a[N - 1][N - 1]) - t[N - 2][N - 1]
                        };

                        let sqrt_alpha_nm1_n_beta_nm1 = Float::sqrt(beta_nm1.abs()*a[N - 2][N - 1].abs());

                        let wilk_test = |gamma: T| {
                            let dgamma_alpha_nm1_nm1 = (gamma - a[N - 2][N - 2]).abs();
                            let dgamma_alpha_n_n = (gamma - a[N - 1][N - 1]).abs();

                            dgamma_alpha_n_n <= sqrt_alpha_nm1_n_beta_nm1
                                && sqrt_alpha_nm1_n_beta_nm1 <= dgamma_alpha_nm1_nm1
                                && delta(gamma).abs() < T::Real::epsilon()
                        };

                        if wilk_test(gamma1)
                        {
                            gamma1
                        }
                        else if wilk_test(gamma2)
                        {
                            gamma2
                        }
                        else
                        {
                            T::zero()
                        }
                    }
                }
            };
            if let Some(gamma) = <dyn Any>::downcast_mut::<Complex<T::Real>>(&mut gamma as &mut dyn Any)
            {
                *gamma = *gamma + Complex::new(T::Real::zero(), T::Real::epsilon())
            }
            if gamma.is_finite() && !gamma.is_zero()
            {
                for k in 0..N
                {
                    a[k][k] -= gamma;
                }
            }
            let (q, r) = a.qr_matrix();
            a = r.mul_matrix(&q);
            u = u.mul_matrix(&q);
            if gamma.is_finite() && !gamma.is_zero()
            {
                for k in 0..N
                {
                    a[k][k] += gamma;
                }
            }
            a = a.mul_matrix(&p.transpose());
            u = u.mul_matrix(&p.transpose());
            let mut is_done = true;
            if a.iter().any(|a| a.iter().any(|a| Float::is_nan(a.abs())))
            {
                break;
            }
            t = a;
            'lp:
            for k in 0..(N - 1)
            {
                for j in (k + 1)..N
                {
                    let x = t[j][k].abs();
                    if !(!Float::is_normal(x) || x < T::Real::epsilon())
                    {
                        is_done = false;
                        break 'lp;
                    }
                }
            }
            if is_done
            {
                break;
            }
        }

        /*if !is_done
        {
            let nan = T::from(T::Real::nan()).unwrap();
            return ArrayOps::fill(|_| nan)
        }*/

        let lambda = <[T; N]>::fill(|i| t[i][i]);
        
        if TEST
        {
            // Test if AU == UR
            let au = a.mul_matrix(&u);
            let ut = u.mul_matrix(&t);
            for i in 0..N
            {
                for j in 0..N
                {
                    let d = (au[i][j] - ut[i][j]).abs();
                    if d > T::from(EPS).unwrap().re()
                    {
                        panic!("AU != UR")
                    }
                }
            }
        }

        let mut v = <[[T; N]; N]>::identity_matrix();
        for i in 1..N
        {
            let mut lambda_mt = [[T::zero(); N]; N];
            let mut r = [T::zero(); N];
            for k in 0..i
            {
                r[k] = t[k][i];
                lambda_mt[k][k] = t[i][i]
            }
            for j in 0..i
            {
                for k in 0..i
                {
                    lambda_mt[j][k] -= t[j][k];
                }
            }
            let vi = lambda_mt.solve_matrix_complex(&r);
            for k in 0..i
            {
                v[k][i] = vi[k]
            }
        }

        if TEST
        {
            // Test if TV == Vlambda 
            let tv = t.mul_matrix(&v);
            let vlambda = v.mul_matrix(&<[_; N]>::fill(|i| <[_; N]>::fill(|j| if i == j {lambda[i]} else {T::zero()})));
            for i in 0..N
            {
                for j in 0..N
                {
                    let d = (tv[i][j] - vlambda[i][j]).abs();
                    if d > T::from(EPS).unwrap().re()
                    {
                        panic!("RV != Vlambda")
                    }
                }
            }
        }

        let w = u.mul_matrix(&v).transpose().map(|w| w.normalize_complex());

        if TEST
        {
            // Test if AW == Wlambda 
            let aw = self.mul_matrix(&w.transpose());
            let wlambda = w.transpose().mul_matrix(&<[_; N]>::fill(|i| <[_; N]>::fill(|j| if i == j {lambda[i]} else {T::zero()})));
            for i in 0..N
            {
                for j in 0..N
                {
                    let d = (aw[i][j] - wlambda[i][j]).abs();
                    if d > T::from(EPS).unwrap().re()
                    {
                        panic!("AW != Wlambda")
                    }
                }
            }
        }

        (lambda, w)
    }
}