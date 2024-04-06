use core::{any::Any, ops::Add};
use std::{iter::Sum, mem::MaybeUninit, ops::{AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign}};

use array__ops::{max_len, min_len, Array2dOps, ArrayOps, CollumnArrayOps};
use num::{complex::ComplexFloat, Complex, Float, NumCast, One, Signed, Zero};

use crate::{ArrayMath, SquareMatrixMath};

#[const_trait]
pub trait MatrixMath<T, const M: usize, const N: usize>: ~const Array2dOps<T, M, N>
{
    fn identity_matrix() -> Self
    where
        T: Zero + One;
        
    fn eye_matrix(k: isize) -> Self
    where
        T: Zero + One;
        
    fn transpose_conj(self) -> [[T; M]; N]
    where
        T: ComplexFloat;
        
    fn solve_matrix(&self, y: &[T; M]) -> [T; N]
    where
        T: ComplexFloat + AddAssign + SubAssign + DivAssign + Div<T::Real, Output = T>,
        [(); max_len(N, N)]:,
        [(); max_len(M, M)]:;

    fn covariance_matrix(&self, expected: Option<&Self>) -> [[T; N]; N]
    where
        T: Div<Output = T> + Sub<Output = T> + AddAssign + SubAssign + One + Zero + NumCast + Copy;

    /// Performs two-dimensional direct convolution on two matrices.
    /// 
    /// # Example
    /// ```rust
    /// #![feature(generic_const_exprs)]
    /// 
    /// use array_math::*;
    /// 
    /// let x = [
    ///     [1, 0, 0],
    ///     [0, 0, 0],
    ///     [0, 0, 2]
    /// ];
    /// let h = [
    ///     [1, 1],
    ///     [1, 1]
    /// ];
    /// 
    /// let y = x.convolve_2d_direct(&h);
    /// 
    /// assert_eq!(y, [
    ///     [1, 1, 0, 0],
    ///     [1, 1, 0, 0],
    ///     [0, 0, 2, 2],
    ///     [0, 0, 2, 2]
    /// ]);
    /// ```
    fn convolve_2d_direct<Rhs, const H: usize, const W: usize>(&self, rhs: &[[Rhs; W]; H]) -> [[<T as Mul<Rhs>>::Output; N + W - 1]; M + H - 1]
    where
        T: Mul<Rhs, Output: AddAssign + Zero> + Copy,
        Rhs: Copy;
        
    /// Performs two-dimensional convolution using FFT on two matrices.
    /// 
    /// # Example
    /// ```rust
    /// #![feature(generic_arg_infer)]
    /// #![feature(generic_const_exprs)]
    /// 
    /// use array_math::*;
    /// 
    /// let x = [
    ///     [1.0, 0.0, 0.0],
    ///     [0.0, 0.0, 0.0],
    ///     [0.0, 0.0, 2.0]
    /// ];
    /// let h = [
    ///     [1.0, 1.0],
    ///     [1.0, 1.0]
    /// ];
    /// 
    /// let y_fft = x.convolve_2d_real_fft(&h);
    /// let y_direct = x.convolve_2d_direct(&h);
    /// 
    /// let avg_error = y_fft.comap(y_direct, |y_fft, y_direct| y_fft.comap(y_direct, |y_fft: f64, y_direct: f64| (y_fft - y_direct).abs()).avg()).avg();
    /// assert!(avg_error < 1.0e-16);
    /// ```
    fn convolve_2d_real_fft<Rhs, const W: usize, const H: usize>(&self, rhs: &[[Rhs; W]; H]) -> [[<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real; N + W - 1]; M + H - 1]
    where
        T: Float + Copy,
        Rhs: Float + Copy,
        Complex<T>: MulAssign + AddAssign + ComplexFloat<Real = T> + Mul<Complex<Rhs>, Output: ComplexFloat<Real: Float>>,
        Complex<Rhs>: MulAssign + AddAssign + ComplexFloat<Real = Rhs>,
        <Complex<T> as Mul<Complex<Rhs>>>::Output: ComplexFloat<Real: Float> + Into<Complex<<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>>,
        Complex<<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>: MulAssign + AddAssign + ComplexFloat<Real = <<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>,
        [(); (M + H - 1).next_power_of_two()]:,
        [(); (M + H - 1).next_power_of_two() - M]:,
        [(); (M + H - 1).next_power_of_two() - H]:,
        [(); (M + H - 1).next_power_of_two() - (M + H - 1)]:,
        [(); (M + H - 1).next_power_of_two()/2 + 1]:,
        [(); (N + W - 1).next_power_of_two()]:,
        [(); (N + W - 1).next_power_of_two() - N]:,
        [(); (N + W - 1).next_power_of_two() - W]:,
        [(); (N + W - 1).next_power_of_two() - (N + W - 1)]:,
        [(); (N + W - 1).next_power_of_two()/2 + 1]:;
        
    fn convolve_2d_fft<Rhs, const W: usize, const H: usize>(&self, rhs: &[[Rhs; W]; H]) -> [[<T as Mul<Rhs>>::Output; N + W - 1]; M + H - 1]
    where
        T: ComplexFloat + Mul<Rhs, Output: ComplexFloat + From<<<T as Mul<Rhs>>::Output as ComplexFloat>::Real> + 'static>,
        Rhs: ComplexFloat,
        Complex<T::Real>: From<T> + AddAssign + MulAssign + Mul<Complex<Rhs::Real>, Output: ComplexFloat<Real = <<T as Mul<Rhs>>::Output as ComplexFloat>::Real> + MulAssign + AddAssign + From<Complex<<<T as Mul<Rhs>>::Output as ComplexFloat>::Real>> + Sum + 'static>,
        Complex<Rhs::Real>: From<Rhs> + AddAssign + MulAssign,
        [(); (M + H - 1).next_power_of_two()]:,
        [(); (M + H - 1).next_power_of_two() - M]:,
        [(); (M + H - 1).next_power_of_two() - H]:,
        [(); (M + H - 1).next_power_of_two() - (M + H - 1)]:,
        [(); (M + H - 1).next_power_of_two()/2 + 1]:,
        [(); (N + W - 1).next_power_of_two()]:,
        [(); (N + W - 1).next_power_of_two() - N]:,
        [(); (N + W - 1).next_power_of_two() - W]:,
        [(); (N + W - 1).next_power_of_two() - (N + W - 1)]:,
        [(); (N + W - 1).next_power_of_two()/2 + 1]:;

    /// Performs the 2D Cooley-Tukey FFT on a given matrix
    /// 
    /// # Examples
    /// ```rust
    /// #![feature(generic_const_exprs)]
    /// 
    /// use num::Complex;
    /// use array_math::*;
    /// 
    /// let x = [
    ///     [1.0, 0.0],
    ///     [0.0, 1.0]
    /// ].map(|r| r.map(|x| Complex::from(x)));
    /// let mut y = x;
    /// 
    /// y.fft_2d();
    /// y.ifft_2d();
    /// 
    /// assert_eq!(x, y);
    /// ```
    fn fft_2d(&mut self)
    where
        T: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<T::Real>> + Sum;
        
    /// Performs the 2D Cooley-Tukey IFFT on a given matrix
    /// 
    /// # Examples
    /// ```rust
    /// #![feature(generic_const_exprs)]
    /// 
    /// use num::Complex;
    /// use array_math::*;
    /// 
    /// let x = [
    ///     [1.0, 0.0],
    ///     [0.0, 1.0]
    /// ].map(|r| r.map(|x| Complex::from(x)));
    /// let mut y = x;
    /// 
    /// y.fft_2d();
    /// y.ifft_2d();
    /// 
    /// assert_eq!(x, y);
    /// ```
    fn ifft_2d(&mut self)
    where
        T: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<T::Real>> + Sum;

    fn fwht_2d(&mut self)
    where
        T: ComplexFloat + MulAssign<T::Real>,
        [(); N.is_power_of_two() as usize - 1]:,
        [(); M.is_power_of_two() as usize - 1]:;
        
    fn ifwht_2d(&mut self)
    where
        T: ComplexFloat + MulAssign<T::Real>,
        [(); N.is_power_of_two() as usize - 1]:,
        [(); M.is_power_of_two() as usize - 1]:;
        
    fn dst_i_2d(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + DivAssign<T::Real>;
    fn dst_ii_2d(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign;
    fn dst_iii_2d(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + Mul<T, Output = Complex<T::Real>>;
    fn dst_iv_2d(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + Mul<T, Output = Complex<T::Real>>;
        
    fn dct_i_2d(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + DivAssign<T::Real> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + DivAssign<T::Real>;
    fn dct_ii_2d(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign;
    fn dct_iii_2d(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + Mul<T, Output = Complex<T::Real>> + DivAssign<T::Real>;
    fn dct_iv_2d(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + Mul<T, Output = Complex<T::Real>>;
        
    fn real_fft_2d_tall(&self, y: &mut [[Complex<T>; N]; M/2 + 1])
    where
        T: Float,
        Complex<T>: ComplexFloat<Real = T> + MulAssign + AddAssign;
    fn real_fft_2d_wide(&self, y: &mut [[Complex<T>; N/2 + 1]; M])
    where
        T: Float,
        Complex<T>: ComplexFloat<Real = T> + MulAssign + AddAssign;
    fn real_ifft_2d_tall(&mut self, y: &[[Complex<T>; N]; M/2 + 1])
    where
        T: Float,
        Complex<T>: ComplexFloat<Real = T> + MulAssign + AddAssign;
    fn real_ifft_2d_wide(&mut self, y: &[[Complex<T>; N/2 + 1]; M])
    where
        T: Float,
        Complex<T>: ComplexFloat<Real = T> + MulAssign + AddAssign;

    fn mul_matrix<Rhs, const P: usize>(&self, rhs: &[[Rhs; P]; N]) -> [[<T as Mul<Rhs>>::Output; P]; M]
    where
        T: Mul<Rhs, Output: AddAssign + Zero> + Copy,
        Rhs: Copy;
    fn mul_matrix_assign<Rhs>(&mut self, rhs: &[[Rhs; N]; N])
    where
        T: Mul<Rhs> + Copy + AddAssign<<T as Mul<Rhs>>::Output> + Zero,
        Rhs: Copy;
    fn rmul_matrix_assign<Rhs>(&mut self, rhs: &[[Rhs; M]; M])
    where
        T: Copy + AddAssign<<Rhs as Mul<T>>::Output> + Zero,
        Rhs: Copy + Mul<T>;

    fn add_matrix<Rhs>(self, rhs: [[Rhs; N]; M]) -> [[<T as Add<Rhs>>::Output; N]; M]
    where
        T: Add<Rhs>;
    fn add_matrix_assign<Rhs>(&mut self, rhs: [[Rhs; N]; M])
    where
        T: AddAssign<Rhs>;
        
    fn sub_matrix<Rhs>(self, rhs: [[Rhs; N]; M]) -> [[<T as Sub<Rhs>>::Output; N]; M]
    where
        T: Sub<Rhs>;
    fn sub_matrix_assign<Rhs>(&mut self, rhs: [[Rhs; N]; M])
    where
        T: SubAssign<Rhs>;
        
    fn mul_matrix_all<Rhs>(self, rhs: Rhs) -> [[<T as Mul<Rhs>>::Output; N]; M]
    where
        T: Mul<Rhs>,
        Rhs: Copy;
    fn mul_matrix_assign_all<Rhs>(&mut self, rhs: Rhs)
    where
        T: MulAssign<Rhs>,
        Rhs: Copy;
        
    fn div_matrix_all<Rhs>(self, rhs: Rhs) -> [[<T as Div<Rhs>>::Output; N]; M]
    where
        T: Div<Rhs>,
        Rhs: Copy;
    fn div_matrix_assign_all<Rhs>(&mut self, rhs: Rhs)
    where
        T: DivAssign<Rhs>,
        Rhs: Copy;
    
    fn rpivot_matrix(&self) -> [[T; M]; M]
    where
        T: Neg<Output = T> + Zero + One + PartialOrd + Copy;
    fn cpivot_matrix(&self) -> [[T; N]; N]
    where
        T: Neg<Output = T> + Zero + One + PartialOrd + Copy;
    fn rpivot_matrix_complex(&self) -> [[T; M]; M]
    where
        T: ComplexFloat + Copy;
    fn cpivot_matrix_complex(&self) -> [[T; N]; N]
    where
        T: ComplexFloat + Copy;
    
    fn lup_matrix(&self) -> ([[T; max_len(M, N)]; M], [[T; N]; max_len(M, N)], [[T; M]; M])
    where
        T: Neg<Output = T> + Zero + One + PartialOrd + Mul<Output = T> + AddAssign + Copy + Sub<Output = T> + Div<Output = T>;
    fn lup_matrix_complex(&self) -> ([[T; max_len(M, N)]; M], [[T; N]; max_len(M, N)], [[T; M]; M])
    where
        T: ComplexFloat + AddAssign + Copy;
        
    fn lupq_matrix(&self) -> ([[T; max_len(M, N)]; M], [[T; N]; max_len(M, N)], [[T; M]; M], [[T; N]; N])
    where
        T: Neg<Output = T> + Zero + One + PartialOrd + Mul<Output = T> + AddAssign + Copy + Sub<Output = T> + Div<Output = T>;
    fn lupq_matrix_complex(&self) -> ([[T; max_len(M, N)]; M], [[T; N]; max_len(M, N)], [[T; M]; M], [[T; N]; N])
    where
        T: ComplexFloat + AddAssign + Copy;
            
    fn lu_matrix(&self) -> ([[T; max_len(M, N)]; M], [[T; N]; max_len(M, N)])
    where
        T: Zero + One + Mul<Output = T> + Sub<Output = T> + Div<Output = T> + AddAssign + Copy;

    fn det_matrix(&self) -> T
    where
        T: Zero + Neg<Output = T> + One + Copy + AddAssign + PartialOrd + Zero + Mul<Output = T> + MulAssign + Sub<Output = T> + Div<Output = T>,
        [(); max_len(M, N)]:;
    fn det_matrix_complex(&self) -> T
    where
        T: ComplexFloat + AddAssign + MulAssign + Copy,
        [(); max_len(M, N)]:;
        
    fn qrp_matrix(&self) -> ([[T; M]; M], [[T; N]; M], [[T; N]; N])
    where
        T: ComplexFloat + SubAssign + AddAssign + Copy + DivAssign<T::Real>;
    fn qr_matrix(&self) -> ([[T; M]; M], [[T; N]; M])
    where
        T: ComplexFloat + SubAssign + AddAssign + Copy + DivAssign<T::Real>;
    
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

    fn eye_matrix(k: isize) -> Self
    where
        T: Zero + One
    {
        ArrayOps::fill(|m| ArrayOps::fill(|n| if if k >= 0
        {
            n == k as usize + m
        }
        else
        {
            n + (-k) as usize == m
        }
        {
            One::one()
        }
        else
        {
            Zero::zero()
        }))
    }
    
    fn transpose_conj(self) -> [[T; M]; N]
    where
        T: ComplexFloat
    {
        let mut t = self.transpose();
        for t in t.iter_mut()
        {
            t.conj_assign_all()
        }
        t
    }
    
    fn solve_matrix(&self, y: &[T; M]) -> [T; N]
    where
        T: ComplexFloat + AddAssign + SubAssign + DivAssign + Div<T::Real, Output = T>,
        [(); max_len(N, N)]:,
        [(); max_len(M, M)]:
    {
        if N == 0
        {
            return [T::zero(); N]
        }

        let tol = T::Real::epsilon();

        let at = self.transpose();

        let mut beta_hat = at.mul_matrix(self)
            .inv_matrix_complex()
            .map(|atainv| atainv.mul_matrix(&at.mul_matrix(y.as_collumn()))
                .into_uncollumn()
            ).unwrap_or([T::zero(); N]);

        loop
        {
            let epsilon = y.sub_each(self.mul_matrix(beta_hat.as_collumn()).into_uncollumn());

            let mut is_done = true;
            for e in epsilon.iter()
            {
                if e.abs() > tol
                {
                    is_done = false;
                    break
                }
            }
            if is_done
            {
                break
            }
            let sigma = (<T::Real as NumCast>::from(M).unwrap() - NumCast::from(N).unwrap()).max(One::one());
            let omega = epsilon.mul_outer(&epsilon)
                .div_matrix_all(sigma);
            let omega_inv = omega.inv_matrix_complex();
            let omega_inv = if let Some(omega_inv) = omega_inv
            {
                omega_inv
            }
            else
            {
                break
            };
            let atoinvinv = at.mul_matrix(&omega_inv)
                .mul_matrix(self)
                .inv_matrix_complex();
            let atoinvinv = if let Some(atoinvinv) = atoinvinv
            {
                atoinvinv
            }
            else
            {
                break
            };
            beta_hat = atoinvinv
                .mul_matrix(&at.mul_matrix(&omega_inv).mul_matrix(y.as_collumn()))
                .into_uncollumn()
        }

        beta_hat
    }
    
    fn covariance_matrix(&self, expected: Option<&Self>) -> [[T; N]; N]
    where
        T: Div<Output = T> + Sub<Output = T> + AddAssign + SubAssign + One + Zero + NumCast + Copy
    {
        let s = expected.is_some();
        let expected = expected.unwrap_or(self);
        let mean = expected.transpose()
            .map(|col| col.avg());
        let mut d = T::from(M).unwrap();
        if !s
        {
            d -= T::one()
        }

        ArrayOps::fill(|j| ArrayOps::fill(|k| {
            let mut c = T::zero();
            for i in 0..M
            {
                c += (self[i][j] - mean[j])*(self[i][k] - mean[k])
            }
            c/d
        }))
    }
    
    fn convolve_2d_direct<Rhs, const H: usize, const W: usize>(&self, rhs: &[[Rhs; W]; H]) -> [[<T as Mul<Rhs>>::Output; N + W - 1]; M + H - 1]
    where
        T: Mul<Rhs, Output: AddAssign + Zero> + Copy,
        Rhs: Copy
    {
        ArrayOps::fill(|r| ArrayOps::fill(|c| {
            let mut y = Zero::zero();
            for k in (r + 1).saturating_sub(M)..H.min(r + 1)
            {
                for j in (c + 1).saturating_sub(N)..W.min(c + 1)
                {
                    y += self[r - k][c - j]*rhs[k][j];
                }
            }
            y
        }))
    }
    
    fn convolve_2d_real_fft<Rhs, const W: usize, const H: usize>(&self, rhs: &[[Rhs; W]; H]) -> [[<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real; N + W - 1]; M + H - 1]
    where
        T: Float + Copy,
        Rhs: Float + Copy,
        Complex<T>: MulAssign + AddAssign + ComplexFloat<Real = T> + Mul<Complex<Rhs>, Output: ComplexFloat<Real: Float>>,
        Complex<Rhs>: MulAssign + AddAssign + ComplexFloat<Real = Rhs>,
        <Complex<T> as Mul<Complex<Rhs>>>::Output: ComplexFloat<Real: Float> + Into<Complex<<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>>,
        Complex<<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>: MulAssign + AddAssign + ComplexFloat<Real = <<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>,
        [(); (M + H - 1).next_power_of_two()]:,
        [(); (M + H - 1).next_power_of_two() - M]:,
        [(); (M + H - 1).next_power_of_two() - H]:,
        [(); (M + H - 1).next_power_of_two() - (M + H - 1)]:,
        [(); (M + H - 1).next_power_of_two()/2 + 1]:,
        [(); (N + W - 1).next_power_of_two()]:,
        [(); (N + W - 1).next_power_of_two() - N]:,
        [(); (N + W - 1).next_power_of_two() - W]:,
        [(); (N + W - 1).next_power_of_two() - (N + W - 1)]:,
        [(); (N + W - 1).next_power_of_two()/2 + 1]:
    {
        let x: [_; (M + H - 1).next_power_of_two()] = self
            .map(|x| x.resize(|_| Zero::zero()))
            .resize(|_| [Zero::zero(); (N + W - 1).next_power_of_two()]);
        let h: [_; (M + H - 1).next_power_of_two()] = rhs
            .map(|h| h.resize(|_| Zero::zero()))
            .resize(|_| [Zero::zero(); (N + W - 1).next_power_of_two()]);

        let mut y = [[Zero::zero(); (N + W - 1).next_power_of_two()]; (M + H - 1).next_power_of_two()];
        if N + W >= M + H
        {
            let mut x_f = [[Complex::zero(); _]; _];
            x.real_fft_2d_tall(&mut x_f);
            
            let mut h_f = [[Complex::zero(); _]; _];
            h.real_fft_2d_tall(&mut h_f);
    
            let y_f = x_f.comap(h_f, |x_f, h_f| x_f.comap(h_f, |x_f, h_f| (x_f*h_f).into()));
            y.real_ifft_2d_tall(&y_f);
        }
        else
        {
            let mut x_f = [[Complex::zero(); _]; _];
            x.real_fft_2d_wide(&mut x_f);
            
            let mut h_f = [[Complex::zero(); _]; _];
            h.real_fft_2d_wide(&mut h_f);
    
            let y_f = x_f.comap(h_f, |x_f, h_f| x_f.comap(h_f, |x_f, h_f| (x_f*h_f).into()));
            y.real_ifft_2d_wide(&y_f);
        }

        y.truncate()
            .map(|y| y.truncate())
    }
    
    fn convolve_2d_fft<Rhs, const W: usize, const H: usize>(&self, rhs: &[[Rhs; W]; H]) -> [[<T as Mul<Rhs>>::Output; N + W - 1]; M + H - 1]
    where
        T: ComplexFloat + Mul<Rhs, Output: ComplexFloat + From<<<T as Mul<Rhs>>::Output as ComplexFloat>::Real> + 'static>,
        Rhs: ComplexFloat,
        Complex<T::Real>: From<T> + AddAssign + MulAssign + Mul<Complex<Rhs::Real>, Output: ComplexFloat<Real = <<T as Mul<Rhs>>::Output as ComplexFloat>::Real> + MulAssign + AddAssign + From<Complex<<<T as Mul<Rhs>>::Output as ComplexFloat>::Real>> + Sum + 'static>,
        Complex<Rhs::Real>: From<Rhs> + AddAssign + MulAssign,
        [(); (M + H - 1).next_power_of_two()]:,
        [(); (M + H - 1).next_power_of_two() - M]:,
        [(); (M + H - 1).next_power_of_two() - H]:,
        [(); (M + H - 1).next_power_of_two() - (M + H - 1)]:,
        [(); (M + H - 1).next_power_of_two()/2 + 1]:,
        [(); (N + W - 1).next_power_of_two()]:,
        [(); (N + W - 1).next_power_of_two() - N]:,
        [(); (N + W - 1).next_power_of_two() - W]:,
        [(); (N + W - 1).next_power_of_two() - (N + W - 1)]:,
        [(); (N + W - 1).next_power_of_two()/2 + 1]:
    {
        let mut x: [[Complex<T::Real>; (N + W - 1).next_power_of_two()]; (M + H - 1).next_power_of_two()] = self
            .map(|x| x.map(|x| x.into()).resize(|_| Zero::zero()))
            .resize(|_| [Zero::zero(); (N + W - 1).next_power_of_two()]);
        let mut h: [[Complex<Rhs::Real>; (N + W - 1).next_power_of_two()]; (M + H - 1).next_power_of_two()] = rhs
            .map(|h| h.map(|h| h.into()).resize(|_| Zero::zero()))
            .resize(|_| [Zero::zero(); (N + W - 1).next_power_of_two()]);

        x.fft_2d();
        h.fft_2d();
        
        let mut y = x.comap(h, |x_f, h_f| x_f.comap(h_f, |x_f, h_f| x_f*h_f));

        y.ifft_2d();

        y.truncate()
            .map(|y| y.map(|y| {
                if let Some(y) = <dyn Any>::downcast_ref::<<T as Mul<Rhs>>::Output>(&y as &dyn Any)
                {
                    *y
                }
                else
                {
                    y.re().into()
                }
            }).truncate())
    }

    fn fft_2d(&mut self)
    where
        T: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<T::Real>> + Sum
    {
        let mut t = self.transpose();

        for r in t.iter_mut()
        {
            r.fft();
        }

        *self = t.transpose();

        for r in self.iter_mut()
        {
            r.fft();
        }
    }
    fn ifft_2d(&mut self)
    where
        T: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<T::Real>> + Sum
    {
        let mut t = self.transpose();

        for r in t.iter_mut()
        {
            r.ifft();
        }

        *self = t.transpose();

        for r in self.iter_mut()
        {
            r.ifft();
        }
    }
    
    fn fwht_2d(&mut self)
    where
        T: ComplexFloat + MulAssign<T::Real>,
        [(); N.is_power_of_two() as usize - 1]:,
        [(); M.is_power_of_two() as usize - 1]:
    {
        let mut t = self.transpose();

        for r in t.iter_mut()
        {
            r.fwht();
        }

        *self = t.transpose();

        for r in self.iter_mut()
        {
            r.fwht();
        }
    }
        
    fn ifwht_2d(&mut self)
    where
        T: ComplexFloat + MulAssign<T::Real>,
        [(); N.is_power_of_two() as usize - 1]:,
        [(); M.is_power_of_two() as usize - 1]:
    {
        let mut t = self.transpose();

        for r in t.iter_mut()
        {
            r.ifwht();
        }

        *self = t.transpose();

        for r in self.iter_mut()
        {
            r.ifwht();
        }
    }
    
    fn dst_i_2d(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + DivAssign<T::Real>
    {
        let mut t = self.transpose();

        for r in t.iter_mut()
        {
            r.dst_i();
        }

        *self = t.transpose();

        for r in self.iter_mut()
        {
            r.dst_i();
        }
    }
    fn dst_ii_2d(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign
    {
        let mut t = self.transpose();

        for r in t.iter_mut()
        {
            r.dst_ii();
        }

        *self = t.transpose();

        for r in self.iter_mut()
        {
            r.dst_ii();
        }
    }
    fn dst_iii_2d(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + Mul<T, Output = Complex<T::Real>>
    {
        let mut t = self.transpose();

        for r in t.iter_mut()
        {
            r.dst_iii();
        }

        *self = t.transpose();

        for r in self.iter_mut()
        {
            r.dst_iii();
        }
    }
    fn dst_iv_2d(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + Mul<T, Output = Complex<T::Real>>
    {
        let mut t = self.transpose();

        for r in t.iter_mut()
        {
            r.dst_iv();
        }

        *self = t.transpose();

        for r in self.iter_mut()
        {
            r.dst_iv();
        }
    }
        
    fn dct_i_2d(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + DivAssign<T::Real> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + DivAssign<T::Real>
    {
        let mut t = self.transpose();

        for r in t.iter_mut()
        {
            r.dct_i();
        }

        *self = t.transpose();

        for r in self.iter_mut()
        {
            r.dct_i();
        }
    }
    fn dct_ii_2d(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign
    {
        let mut t = self.transpose();

        for r in t.iter_mut()
        {
            r.dct_ii();
        }

        *self = t.transpose();

        for r in self.iter_mut()
        {
            r.dct_ii();
        }
    }
    fn dct_iii_2d(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + Mul<T, Output = Complex<T::Real>> + DivAssign<T::Real>
    {
        let mut t = self.transpose();

        for r in t.iter_mut()
        {
            r.dct_iii();
        }

        *self = t.transpose();

        for r in self.iter_mut()
        {
            r.dct_iii();
        }
    }
    fn dct_iv_2d(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + Mul<T, Output = Complex<T::Real>>
    {
        let mut t = self.transpose();

        for r in t.iter_mut()
        {
            r.dct_iv();
        }

        *self = t.transpose();

        for r in self.iter_mut()
        {
            r.dct_iv();
        }
    }
    
    fn real_fft_2d_tall(&self, y: &mut [[Complex<T>; N]; M/2 + 1])
    where
        T: Float,
        Complex<T>: ComplexFloat<Real = T> + MulAssign + AddAssign
    {
        let t = self.transpose();
        let mut t_f = [[Complex::zero(); _]; _];

        for (r, r_f) in t.iter()
            .zip(t_f.iter_mut())
        {
            r.real_fft(r_f);
        }

        *y = t_f.transpose();

        for r in y.iter_mut()
        {
            r.fft();
        }
    }
    fn real_fft_2d_wide(&self, y: &mut [[Complex<T>; N/2 + 1]; M])
    where
        T: Float,
        Complex<T>: ComplexFloat<Real = T> + MulAssign + AddAssign
    {
        let mut t_f = [[Complex::zero(); _]; _];

        for (r, r_f) in self.iter()
            .zip(t_f.iter_mut())
        {
            r.real_fft(r_f);
        }

        let mut t_f = t_f.transpose();

        for r in t_f.iter_mut()
        {
            r.fft();
        }

        *y = t_f.transpose()
    }
    fn real_ifft_2d_tall(&mut self, y: &[[Complex<T>; N]; M/2 + 1])
    where
        T: Float,
        Complex<T>: ComplexFloat<Real = T> + MulAssign + AddAssign
    {
        let mut t = *y;

        for r in t.iter_mut()
        {
            r.ifft();
        }

        let t_f = t.transpose();
        let mut t = [[T::zero(); _]; _];

        for (r, r_f) in t.iter_mut()
            .zip(t_f.iter())
        {
            r.real_ifft(r_f);
        }

        *self = t.transpose();
    }
    fn real_ifft_2d_wide(&mut self, y: &[[Complex<T>; N/2 + 1]; M])
    where
        T: Float,
        Complex<T>: ComplexFloat<Real = T> + MulAssign + AddAssign
    {
        let mut t = y.transpose();

        for r in t.iter_mut()
        {
            r.ifft();
        }

        let t_f = t.transpose();

        for (r, r_f) in self.iter_mut()
            .zip(t_f.iter())
        {
            r.real_ifft(r_f);
        }
    }

    fn mul_matrix<Rhs, const P: usize>(&self, rhs: &[[Rhs; P]; N]) -> [[<T as Mul<Rhs>>::Output; P]; M]
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
    fn mul_matrix_assign<Rhs>(&mut self, rhs: &[[Rhs; N]; N])
    where
        T: Mul<Rhs> + Copy + AddAssign<<T as Mul<Rhs>>::Output> + Zero,
        Rhs: Copy
    {
        let mut m = 0;
        while m != M
        {
            let mut buf = [Zero::zero(); _];
            core::mem::swap(&mut self[m], &mut buf);
            let mut p = 0;
            while p != N
            {
                let mut n = 0;
                while n != N
                {
                    self[m][p] += buf[n]*rhs[n][p];
                    n += 1;
                }
                p += 1;
            }
            m += 1;
        }
    }
    fn rmul_matrix_assign<Rhs>(&mut self, rhs: &[[Rhs; M]; M])
    where
        T: Copy + AddAssign<<Rhs as Mul<T>>::Output> + Zero,
        Rhs: Copy + Mul<T>
    {
        let mut buf = [[Zero::zero(); _]; _];
        core::mem::swap(self, &mut buf);

        let mut m = 0;
        while m != M
        {
            let mut p = 0;
            while p != N
            {
                let mut n = 0;
                while n != M
                {
                    self[m][p] += rhs[m][n]*buf[n][p];
                    n += 1;
                }
                p += 1;
            }
            m += 1;
        }
    }
    fn add_matrix<Rhs>(self, rhs: [[Rhs; N]; M]) -> [[<T as Add<Rhs>>::Output; N]; M]
    where
        T: Add<Rhs>
    {
        self.comap(rhs, |lhs, rhs| lhs.comap(rhs, |lhs, rhs| lhs + rhs))
    }
    fn add_matrix_assign<Rhs>(&mut self, rhs: [[Rhs; N]; M])
    where
        T: AddAssign<Rhs>
    {
        for (i, rhs) in rhs.into_iter()
            .enumerate()
        {
            self[i].add_assign_each(rhs)
        }
    }
        
    fn sub_matrix<Rhs>(self, rhs: [[Rhs; N]; M]) -> [[<T as Sub<Rhs>>::Output; N]; M]
    where
        T: Sub<Rhs>
    {
        self.comap(rhs, |lhs, rhs| lhs.comap(rhs, |lhs, rhs| lhs - rhs))
    }
    fn sub_matrix_assign<Rhs>(&mut self, rhs: [[Rhs; N]; M])
    where
        T: SubAssign<Rhs>
    {
        for (i, rhs) in rhs.into_iter()
            .enumerate()
        {
            self[i].sub_assign_each(rhs)
        }
    }
        
    fn mul_matrix_all<Rhs>(self, rhs: Rhs) -> [[<T as Mul<Rhs>>::Output; N]; M]
    where
        T: Mul<Rhs>,
        Rhs: Copy
    {
        self.map(|lhs| lhs.mul_all(rhs))
    }
    fn mul_matrix_assign_all<Rhs>(&mut self, rhs: Rhs)
    where
        T: MulAssign<Rhs>,
        Rhs: Copy
    {
        for lhs in self.iter_mut()
        {
            lhs.mul_assign_all(rhs)
        }
    }
        
    fn div_matrix_all<Rhs>(self, rhs: Rhs) -> [[<T as Div<Rhs>>::Output; N]; M]
    where
        T: Div<Rhs>,
        Rhs: Copy
    {
        self.map(|lhs| lhs.div_all(rhs))
    }
    fn div_matrix_assign_all<Rhs>(&mut self, rhs: Rhs)
    where
        T: DivAssign<Rhs>,
        Rhs: Copy
    {
        for lhs in self.iter_mut()
        {
            lhs.div_assign_all(rhs)
        }
    }
    
    fn rpivot_matrix(&self) -> [[T; M]; M]
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
                let p = p.each_mut()    
                    .map(|p| p as *mut [T; M]);
                unsafe {core::ptr::swap(p[n], p[row_max])};
            }
    
            n += 1;
        }
    
        p
    }
    fn cpivot_matrix(&self) -> [[T; N]; N]
    where
        T: Neg<Output = T> + Zero + One + PartialOrd + Copy
    {
        self.transpose().rpivot_matrix().transpose()
    }
    
    fn rpivot_matrix_complex(&self) -> [[T; M]; M]
    where
        T: ComplexFloat + Copy
    {
        let mut p = <[[T; M]; M]>::identity_matrix();
        
        let mut n = 0;
        while n < M.min(N)
        {
            let mut row_max = n;
            let mut e_abs_max = self[n][n].abs();
            
            let mut m = n + 1;
            while m < M
            {
                let e_abs = self[m][m].abs();
                if e_abs > e_abs_max
                {
                    row_max = m;
                    e_abs_max = e_abs;
                }
                m += 1;
            }
    
            if row_max != n
            {
                let p = p.each_mut()    
                    .map(|p| p as *mut [T; M]);
                unsafe {core::ptr::swap(p[n], p[row_max])};
            }
    
            n += 1;
        }
    
        p
    }
    
    fn cpivot_matrix_complex(&self) -> [[T; N]; N]
    where
        T: ComplexFloat + Copy
    {
        self.transpose().rpivot_matrix_complex().transpose()
    }
    
    fn lup_matrix(&self) -> ([[T; max_len(M, N)]; M], [[T; N]; max_len(M, N)], [[T; M]; M])
    where
        T: Neg<Output = T> + Zero + One + PartialOrd + Mul<Output = T> + AddAssign + Copy + Sub<Output = T> + Div<Output = T>
    {
        let p = self.rpivot_matrix();
        let pa = p.mul_matrix(self);
    
        let (l, u) = pa.lu_matrix();   
    
        (l, u, p)
    }
    fn lup_matrix_complex(&self) -> ([[T; max_len(M, N)]; M], [[T; N]; max_len(M, N)], [[T; M]; M])
    where
        T: ComplexFloat + AddAssign + Copy
    {
        let p = self.rpivot_matrix_complex();
        let pa = p.mul_matrix(self);
    
        let (l, u) = pa.lu_matrix();   
    
        (l, u, p)
    }
    
    fn lupq_matrix(&self) -> ([[T; max_len(M, N)]; M], [[T; N]; max_len(M, N)], [[T; M]; M], [[T; N]; N])
    where
        T: Neg<Output = T> + Zero + One + PartialOrd + Mul<Output = T> + AddAssign + Copy + Sub<Output = T> + Div<Output = T>
    {
        let p = self.rpivot_matrix();
        let pa = p.mul_matrix(self);
        let q = pa.cpivot_matrix();
        let paq = pa.mul_matrix(&q);
    
        let (l, u) = paq.lu_matrix();   
    
        (l, u, p, q)
    }
    fn lupq_matrix_complex(&self) -> ([[T; max_len(M, N)]; M], [[T; N]; max_len(M, N)], [[T; M]; M], [[T; N]; N])
    where
        T: ComplexFloat + AddAssign + Copy
    {
        let p = self.rpivot_matrix_complex();
        let pa = p.mul_matrix(self);
        let q = pa.cpivot_matrix_complex();
        let paq = pa.mul_matrix(&q);
    
        let (l, u) = paq.lu_matrix();   
    
        (l, u, p, q)
    }
            
    fn lu_matrix(&self) -> ([[T; max_len(M, N)]; M], [[T; N]; max_len(M, N)])
    where
        T: Zero + One + Mul<Output = T> + Sub<Output = T> + Div<Output = T> + AddAssign + Copy
    {
        let mut l = [[Zero::zero(); _]; _];
        let mut u = [[Zero::zero(); _]; _];
    
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
                    if s.is_zero()
                    {
                        l[i][n] = self[i][n]/u[n][n];
                    }
                    else
                    {
                        l[i][n] = (self[i][n] - s)/u[n][n];
                    }
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
        let (l, u, p, q) = self.lupq_matrix();
    
        let mut det_abs = One::one();
    
        let mut n = 0;
        while n < min_len(M, N)
        {
            det_abs *= l[n][n];
            det_abs *= u[n][n];
            n += 1;
        }
    
        let mut s = false;
    
        let mut m = 0;
        while m < M
        {
            let mut n = 0;
            while n < M && p[m][n].is_zero()
            {
                n += 1;
            };
            if n != m
            {
                s = !s;
            }
            m += 1;
        }
        
        let mut m = 0;
        while m < N
        {
            let mut n = 0;
            while n < N && q[m][n].is_zero()
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
    fn det_matrix_complex(&self) -> T
    where
        T: ComplexFloat + AddAssign + MulAssign + Copy,
        [(); max_len(M, N)]:
    {
        let (l, u, p, q) = self.lupq_matrix_complex();
    
        let mut det_abs = One::one();
    
        let mut n = 0;
        while n < M
        {
            det_abs *= l[n][n];
            n += 1;
        }
        let mut n = 0;
        while n < N
        {
            det_abs *= u[n][n];
            n += 1;
        }
    
        let mut s = false;
    
        let mut m = 0;
        while m < M
        {
            let mut n = 0;
            while n < M && p[m][n].is_zero()
            {
                n += 1;
            };
            if n != m
            {
                s = !s;
            }
            m += 1;
        }
        
        let mut m = 0;
        while m < N
        {
            let mut n = 0;
            while n < N && q[m][n].is_zero()
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
    
    fn qrp_matrix(&self) -> ([[T; M]; M], [[T; N]; M], [[T; N]; N])
    where
        T: ComplexFloat + SubAssign + AddAssign + Copy + DivAssign<T::Real>
    {
        let p = self.cpivot_matrix_complex();
        let ap = self.mul_matrix(&p);
    
        let (q, r) = ap.qr_matrix();   
    
        (q, r, p)
    }
    fn qr_matrix(&self) -> ([[T; M]; M], [[T; N]; M])
    where
        T: ComplexFloat + SubAssign + AddAssign + Copy + DivAssign<T::Real>
    {
        let mut q = <[[T; M]; M]>::identity_matrix();
        let mut r = *self;
        let mut w;
        for j in 0..N.min(M)
        {
            let mut x_norm_sqr = T::zero();
            for k in j..M
            {
                let x = r[k][j];
                x_norm_sqr += x.conj()*x;
            }
            let x_norm = x_norm_sqr.sqrt();
            if x_norm.is_zero()
            {
                continue
            }
            let mut s = -r[j][j];
            if !s.is_zero()
            {
                s /= s.abs()
            }
            else
            {
                s = T::one()
            }
            let u1 = r[j][j] - s*x_norm;
            w = [T::zero(); M];
            for k in (j + 1)..M
            {
                w[k] = r[k][j]/u1;
            }
            w[j] = T::one();
            let tau = -u1/s/x_norm;
            let w_tau = w.mul_all(tau);

            for i in 0..N
            {
                let m = w.conj_all().mul_dot(r.each_ref().map(|r| r[i]));
                for k in j..M
                {
                    r[k][i] -= w_tau[k]*m
                }
            }
            for i in 0..M
            {
                let m = w.mul_dot(q[i]);
                for k in j..M
                {
                    q[i][k] -= m*w_tau[k].conj()
                }
            }
        }
        
        for k in 1..N
        {
            for i in k..M
            {
                r[i][k - 1] = T::zero()
            }
        }

        /*let a = q.mul_matrix(&r);
        for k in 0..N
        {
            for i in 0..N
            {
                let d = (a[k][i] - self[k][i]).abs();
                if d > T::from(0.1).unwrap().re()
                {
                    panic!("QR not equal")
                }
            }
        }*/

        (q, r)
    }
    
    fn is_matrix_invertible(&self) -> bool
    where
        T: Zero + Neg<Output = T> + One + Copy + AddAssign + PartialOrd + MulAssign + Sub<Output = T> + Div<Output = T>,
        [(); max_len(M, N)]:
    {
        let (l, u, _, _) = self.lupq_matrix();
        
        let mut n = 0;
        while n < M
        {
            if l[n][n].is_zero()
            {
                return false
            }
            n += 1;
        }
        let mut n = 0;
        while n < N
        {
            if u[n][n].is_zero()
            {
                return false
            }
            n += 1;
        }
    
        true
    }
}

#[cfg(test)]
mod test
{
    use std::f64::EPSILON;

    use array__ops::ArrayOps;
    use num::Complex;

    use crate::{MatrixMath, SquareMatrixMath};

    #[test]
    fn test()
    {
        let n = 0.01;
        let a = [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, n, 0.0],
            [0.0, -n, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0]
        ];
    
        let a = a.map(|a| a.map(|a| Complex::new(a, 0.0)));
    
        let (e, v) = a.eigen();
    
        println!("{:?}", e);
        println!("{:?}", v)
    }
}