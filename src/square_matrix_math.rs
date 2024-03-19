use std::{any::Any, f64::{consts::SQRT_3, EPSILON}, ops::{Add, AddAssign, Div, DivAssign, Mul, Sub, SubAssign}};

use array__ops::{max_len, Array2dOps, ArrayNdOps, ArrayOps, CollumnArrayOps};
use num::{complex::ComplexFloat, Complex, Float, NumCast, One, Signed, Zero};

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

    /// Returns the eigenvalues of the given matrix
    fn eigenvalues(&self) -> [Complex<T::Real>; N]
    where
        Complex<T::Real>: From<T> + AddAssign + SubAssign + DivAssign<T::Real>,
        T: ComplexFloat;
    /// Returns the eigenvalues and eigenvectors of the given matrix.
    /// 
    /// The method uses the algorithm described in [Convergence of the Shifted QR Algorithm for Unitary Hessenberg Matrices - Tai-Lin Wang and William B. Gragg](https://www.ams.org/journals/mcom/2002-71-240/S0025-5718-01-01387-4/S0025-5718-01-01387-4.pdf).
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// use num::Complex;
    /// use array_math::{SquareMatrixMath, MatrixMath, ArrayOps};
    /// 
    /// let a = [
    ///     [Complex::new(1.0, -3.0), Complex::new(2.0, 2.0)],
    ///     [Complex::new(3.0, 1.0), Complex::new(4.0, -4.0)]
    /// ];
    /// 
    /// let (e, v) = a.eigen();
    /// 
    /// for (e, v) in e.zip(v)
    /// {
    ///     let av = a.mul_matrix(v.as_collumn()).map(|[av]| av);
    ///     let vlambda = v.mul_all(e);
    ///     
    ///     for (avi, vlambdai) in av.zip(vlambda)
    ///     {
    ///         let d = (avi - vlambdai).norm();
    ///         assert!(d < 1e-10)
    ///     }
    /// }
    /// ```
    fn eigen(&self) -> ([Complex<T::Real>; N], [[Complex<T::Real>; N]; N])
    where
        Complex<T::Real>: From<T> + AddAssign + SubAssign + DivAssign + DivAssign<T::Real>,
        T: ComplexFloat,
        [(); max_len(N, N)]:;

}

impl<T, const N: usize> SquareMatrixMath<T, N> for [[T; N]; N]
{
    fn inv_matrix(&self) -> Option<[[T; N]; N]>
    where
        T: Signed + PartialOrd + One + Zero + Copy + SubAssign + DivAssign + AddAssign,
        [(); max_len(N, N)]:
    {
        let (l, u, p) = self.lup_matrix();
        
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
        let (l, u, p) = self.lup_matrix_complex();
        
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
        let (l, u, p, q) = self.lupq_matrix();
    
        let [bp] = core::array::from_ref(b).mul_matrix(&p);
        let qbp = q.mul_matrix(&bp.as_collumn()).into_uncollumn();
    
        let mut x = qbp;
        
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
        let (l, u, p, q) = self.lupq_matrix_complex();
        //let (l, u, p) = self.lup_matrix_complex();
    
        let mut x = p.mul_matrix(b.as_collumn()).into_uncollumn();
        
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
    
        q.mul_matrix(&x.as_collumn()).into_uncollumn()
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
    
    fn eigenvalues(&self) -> [Complex<T::Real>; N]
    where
        Complex<T::Real>: From<T> + AddAssign + SubAssign + DivAssign<T::Real>,
        T: ComplexFloat
    {
        if N == 0
        {
            return [Zero::zero(); N]
        }

        let mut t = self.map(|a| a.map(|a| <Complex::<T::Real> as From<_>>::from(a)));

        for i in 0..
        {
            let mut t_next = t;
            let lambda = qr_shift(&t_next, i);
            if lambda.is_finite() && !lambda.is_zero()
            {
                for k in 0..N
                {
                    t_next[k][k] -= lambda;
                }
            }
            let (q, r) = t_next.qr_matrix();
            t_next = r.mul_matrix(&q);
            if lambda.is_finite() && !lambda.is_zero()
            {
                for k in 0..N
                {
                    t_next[k][k] += lambda;
                }
            }
            if t_next.iter().any(|t| t.iter().any(|t| Float::is_nan(t.abs())))
            {
                break;
            }
            t = t_next;
            
            let mut is_done = true;
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
    
    fn eigen(&self) -> ([Complex<T::Real>; N], [[Complex<T::Real>; N]; N])
    where
        Complex<T::Real>: From<T> + AddAssign + SubAssign + DivAssign + DivAssign<T::Real>,
        T: ComplexFloat,
        [(); max_len(N, N)]:
    {
        if N == 0
        {
            return ([Zero::zero(); N], [[Zero::zero(); N]; N])
        }

        #[cfg(test)]
        const TEST: bool = true;
        #[cfg(not(test))]
        const TEST: bool = false;
        const TEST_EPSILON: f64 = 0.0001;

        let a = self.map(|a| a.map(|a| <Complex::<T::Real> as From<_>>::from(a))); //self.upper_hessenberg_matrix();
        let mut t = a;
        let mut u = <[[Complex<T::Real>; N]; N]>::identity_matrix();

        for i in 0..
        {
            let mut t_next = t;
            let lambda = qr_shift(&t_next, i);
            if lambda.is_finite() && !lambda.is_zero()
            {
                for k in 0..N
                {
                    t_next[k][k] -= lambda;
                }
            }
            let (q, r) = t_next.qr_matrix();
            t_next = r.mul_matrix(&q);
            if lambda.is_finite() && !lambda.is_zero()
            {
                for k in 0..N
                {
                    t_next[k][k] += lambda;
                }
            }
            if t_next.iter().any(|t| t.iter().any(|t| Float::is_nan(t.abs())))
            {
                break;
            }
            u = u.mul_matrix(&q);
            t = t_next;
            
            let mut is_done = true;
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

        for k in 0..N
        {
            for i in (k + 1)..N
            {
                t[i][k] = Complex::zero()
            }
        }

        /*if !is_done
        {
            let nan = T::from(T::Real::nan()).unwrap();
            return ArrayOps::fill(|_| nan)
        }*/

        let lambda = <[Complex<T::Real>; N]>::fill(|i| t[i][i]);
        
        if TEST
        {
            // Test if AU == UT
            let au = a.mul_matrix(&u);
            let ut = u.mul_matrix(&t);
            for i in 0..N
            {
                for j in 0..N
                {
                    let d = (au[i][j] - ut[i][j]).abs();
                    if !(d < T::from(TEST_EPSILON).unwrap().re())
                    {
                        panic!("AU != UT")
                    }
                }
            }
        }

        let mut v = <[[Complex<T::Real>; N]; N]>::identity_matrix();
        for i in 1..N
        {
            let mut lambda_mt = [[Complex::zero(); N]; N];
            let mut r = [Complex::zero(); N];
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
            let vlambda = v.mul_matrix(&<[_; N]>::fill(|i| <[_; N]>::fill(|j| if i == j {lambda[i]} else {Zero::zero()})));
            for i in 0..N
            {
                for j in 0..N
                {
                    let d = (tv[i][j] - vlambda[i][j]).abs();
                    if !(d < T::from(TEST_EPSILON).unwrap().re())
                    {
                        panic!("AU != UT")
                    }
                }
            }
        }

        let w = u.mul_matrix(&v).transpose().map(|w| w.normalize_complex());

        if TEST
        {
            // Test if AW == Wlambda 
            let aw = a.mul_matrix(&w.transpose());
            let wlambda = w.transpose().mul_matrix(&<[_; N]>::fill(|i| <[_; N]>::fill(|j| if i == j {lambda[i]} else {Zero::zero()})));
            for i in 0..N
            {
                for j in 0..N
                {
                    let d = (aw[i][j] - wlambda[i][j]).abs();
                    if !(d < T::from(TEST_EPSILON).unwrap().re())
                    {
                        panic!("AW != Wlambda")
                    }
                }
            }
        }

        (lambda, w)
    }
}

fn qr_shift<T, const N: usize>(a: &[[Complex<T>; N]; N], i: usize) -> Complex<T>
where
    T: Float,
    Complex<T>: ComplexFloat<Real = T> + AddAssign + SubAssign + DivAssign<T> + Div<T, Output = Complex<T>> + Mul<T, Output = Complex<T>> + Add<T, Output = Complex<T>> + Sub<T, Output = Complex<T>>
{
    if N < 1
    {
        return Zero::zero()
    }

    let mut beta_nm1 = Zero::zero();
    let mut beta_nm2 = Zero::zero();
    
    if N >= 2
    {
        beta_nm1 = a[N - 1][N - 2];
    }
    if N >= 3
    {
        beta_nm2 = a[N - 2][N - 3];
    }

    let beta_nm1_abs = beta_nm1.abs();
    let beta_nm2_abs = beta_nm2.abs();

    let two = T::one() + T::one();
    
    let exceptional = i > 1 && (i - 1) % 10 == 0;

    if exceptional
    {
        let beta = beta_nm1_abs + beta_nm2_abs;

        let b = -a[N - 1][N - 1]*two - beta*T::from(3.0/2.0).unwrap();
        let c = (a[N - 1][N - 1] + beta)*(a[N - 1][N - 1] + beta) - a[N - 1][N - 1]*beta/two;

        let re = -b/T::from(2.0).unwrap();
        let im_sqr = c - re*re;
        let im = im_sqr.sqrt();
        let lambda = Complex::new(re.re() + im.im(), im.re() + re.im());
        
        return lambda
    }

    if N < 3
    {
        // R-shift
        a[N - 1][N - 1]
    }
    else
    {
        let beta_nm1 = a[N - 1][N - 2];
        let beta_nm2 = a[N - 2][N - 3];
        let beta_nm1_abs = beta_nm1.abs();
        let beta_nm2_abs = beta_nm2.abs();

        let phi = Float::recip(Float::sqrt(two - beta_nm1_abs*beta_nm1_abs));
        let psi = if beta_nm2_abs > T::from(SQRT_3/2.0).unwrap()
        {
            beta_nm2_abs
        }
        else
        {
            Float::sqrt(T::one() + Float::recip(Float::sqrt(T::one() - beta_nm2_abs*beta_nm2_abs)))/two
        };
        let theta = phi.min(psi);

        let a_sub2 = <[_; 2]>::fill(|i| <[_; 2]>::fill(|j| a[N - 2 + i][N - 2 + j]));

        if beta_nm2_abs*theta < beta_nm1_abs && !a[N - 1][N - 1].is_zero()
        {
            // R-shift
            return a[N - 1][N - 1]
        }
        else if beta_nm2_abs*theta >= beta_nm1_abs
        {
            let [gamma1, gamma2] = a_sub2.eigenvalues();
        
            let delta = |gamma: Complex<T>| {
                (gamma - a[N - 2][N - 2])*(gamma - a[N - 1][N - 1]) - a[N - 2][N - 1]
            };

            let sqrt_alpha_nm1_n_beta_nm1 = Float::sqrt(beta_nm1.abs()*a[N - 2][N - 1].abs());

            let wilk_test = |gamma: Complex<T>| {
                let dgamma_alpha_nm1_nm1 = (gamma - a[N - 2][N - 2]).abs();
                let dgamma_alpha_n_n = (gamma - a[N - 1][N - 1]).abs();

                if dgamma_alpha_n_n <= sqrt_alpha_nm1_n_beta_nm1
                    && sqrt_alpha_nm1_n_beta_nm1 <= dgamma_alpha_nm1_nm1
                {
                    let delta = delta(gamma).abs();
                    if delta <= T::from(0.001).unwrap()
                    {
                        Some(delta)
                    }
                    else
                    {
                        None
                    }
                }
                else
                {
                    None
                }
            };

            // W-shift
            match (wilk_test(gamma1), wilk_test(gamma2))
            {
                (Some(_), None) => return gamma1,
                (None, Some(_)) => return gamma2,
                (Some(delta1), Some(delta2)) => if delta1 <= delta2
                {
                    return gamma1
                }
                else
                {
                    return gamma2
                },
                (None, None) => ()
            }
        }
        
        let trhh = a_sub2[0][0] + a_sub2[1][1];
        let dethh = a_sub2[0][0]*a_sub2[1][1] - a_sub2[0][1]*a_sub2[1][0];
        let trhh_sqr = (trhh.conj()*trhh).re();
        let deth4 = T::from(4.0).unwrap()*dethh.re();

        if dethh.im().is_zero() && trhh_sqr >= deth4
        {
            let s = Float::sqrt(trhh_sqr - deth4);
            let lhh1 = (trhh + s)/T::from(2.0).unwrap();
            let lhh2 = (trhh - s)/T::from(2.0).unwrap();
            if (lhh1 - a[N - 1][N - 1]).abs() < (lhh2 - a[N - 1][N - 1]).abs()
            {
                return lhh1
            }
            else
            {
                return lhh2
            }
        }
        else
        {
            let b = -trhh;
            let c = dethh;

            let re = -b/T::from(2.0).unwrap();
            let im_sqr = c - re*re;
            let im = im_sqr.sqrt();
            let lambda = Complex::new(re.re() + im.im(), im.re() + re.im());

            lambda
        }
    }
}