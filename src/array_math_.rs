use core::any::Any;
use std::{iter::Sum, ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign}};

use array__ops::{max_len, Array2dOps, ArrayOps, CollumnArrayOps, SliceOps};
use num::{complex::ComplexFloat, traits::{FloatConst, Inv, Pow}, Complex, Float, NumCast, One, Zero};

use crate::{MatrixMath, SquareMatrixMath, SliceMath};

const NEWTON_POLYNOMIAL_ROOTS: usize = 16;

#[const_trait]
pub trait ArrayMath<T, const N: usize>: ~const ArrayOps<T, N>
{
    fn sum(self) -> T
    where
        T: AddAssign + Zero;

    fn product(self) -> T
    where
        T: MulAssign + One;

    fn variance(self) -> <T as Mul>::Output
    where
        Self: Copy,
        T: Div<Output: Mul<Output: Neg<Output = <T as Mul>::Output>> + Copy> + Mul<Output: AddAssign> + AddAssign + Zero + NumCast;
    
    fn avg(self) -> <T as Div>::Output
    where
        T: Div + AddAssign + Zero + NumCast;

    fn geometric_mean(self) -> <T as Pow<<T as Inv>::Output>>::Output
    where
        T: MulAssign + One + Pow<<T as Inv>::Output> + Inv + NumCast;

    fn mul_dot<Rhs>(self, rhs: [Rhs; N]) -> <T as Mul<Rhs>>::Output
    where
        T: Mul<Rhs, Output: AddAssign + Zero>;

    fn magnitude_squared(self) -> <T as Mul<T>>::Output
    where
        T: Mul<T, Output: AddAssign + Zero> + Copy;
    fn magnitude_squared_complex(self) -> T::Real
    where
        T: ComplexFloat + AddAssign + Copy;

    fn magnitude(self) -> <T as Mul<T>>::Output
    where
        T: Mul<T, Output: AddAssign + Zero + Float> + Copy;
    fn magnitude_complex(self) -> T::Real
    where
        T: ComplexFloat + AddAssign + Copy;
    
    fn magnitude_inv(self) -> <T as Mul<T>>::Output
    where
        T: Mul<T, Output: AddAssign + Zero + Float> + Copy;
    fn magnitude_inv_complex(self) -> T::Real
    where
        T: ComplexFloat + AddAssign + Copy;

    fn normalize(self) -> [<T as Mul<<T as Mul<T>>::Output>>::Output; N]
    where
        T: Mul<T, Output: AddAssign + Zero + Float + Copy> + Mul<<T as Mul<T>>::Output> + Copy;
    fn normalize_complex(self) -> [<T as Mul<T::Real>>::Output; N]
    where
        T: ComplexFloat + AddAssign + Mul<T::Real> + Copy;

    fn normalize_to<Rhs>(self, magnitude: Rhs) -> [<T as Mul<<<T as Mul<T>>::Output as Mul<Rhs>>::Output>>::Output; N]
    where
        T: Mul<T, Output: AddAssign + Zero + Float + Mul<Rhs, Output: Copy>> + Mul<<<T as Mul<T>>::Output as Mul<Rhs>>::Output> + Copy;
    fn normalize_complex_to<Rhs>(self, magnitude: Rhs) -> [<T as Mul<<T::Real as Mul<Rhs>>::Output>>::Output; N]
    where
        T: ComplexFloat<Real: Mul<Rhs, Output: Copy>> + AddAssign + Mul<<T::Real as Mul<Rhs>>::Output>;

    fn normalize_assign(&mut self)
    where
        T: Mul<T, Output: AddAssign + Zero + Float + Copy> + MulAssign<<T as Mul<T>>::Output> + Copy;
    fn normalize_assign_complex(&mut self)
    where
        T: ComplexFloat + AddAssign + MulAssign<T::Real> + Copy;

    fn normalize_assign_to<Rhs>(&mut self, magnitude: Rhs)
    where
        T: Mul<T, Output: AddAssign + Zero + Float + Mul<Rhs, Output: Copy>> + MulAssign<<<T as Mul<T>>::Output as Mul<Rhs>>::Output> + Copy;
    fn normalize_assign_complex_to<Rhs>(&mut self, magnitude: Rhs)
    where
        T: ComplexFloat + AddAssign + MulAssign<<T::Real as Mul<Rhs>>::Output>,
        T::Real: Mul<Rhs, Output: Copy>;
        
    fn ellipke(self, tol: Option<T>) -> Option<([T; N], [T; N])>
    where
        T: Float + FloatConst + AddAssign + MulAssign;

    fn polynomial<Rhs>(self, rhs: Rhs) -> T
    where
        T: AddAssign + MulAssign<Rhs> + Zero,
        Rhs: Copy;
    fn rpolynomial<Rhs>(self, rhs: Rhs) -> T
    where
        T: AddAssign + MulAssign<Rhs> + Zero,
        Rhs: Copy;
        
    fn derivate_polynomial(self) -> [<T as Mul>::Output; N - 1]
    where
        T: NumCast + Zero + Mul;
    fn derivate_rpolynomial(self) -> [<T as Mul>::Output; N - 1]
    where
        T: NumCast + Zero + Mul;
        
    fn integrate_polynomial(self, c: <T as Div>::Output) -> [<T as Div>::Output; N + 1]
    where
        T: NumCast + Zero + Div;
    fn integrate_rpolynomial(self, c: <T as Div>::Output) -> [<T as Div>::Output; N + 1]
    where
        T: NumCast + Zero + Div;

    fn companion_matrix(&self) -> [[<T as Neg>::Output; N - 1]; N - 1]
    where
        T: Copy + Neg + Zero,
        <T as Neg>::Output: One + Zero + DivAssign<T>;
    fn rcompanion_matrix(&self) -> [[<T as Neg>::Output; N - 1]; N - 1]
    where
        T: Copy + Neg + Zero,
        <T as Neg>::Output: One + Zero + DivAssign<T>;
    fn vandermonde_matrix<const M: usize>(&self) -> [[T; M]; N]
    where
        T: One + Copy + Mul;
        
    fn polynomial_roots(&self) -> [Complex<T::Real>; N - 1]
    where
        Complex<T::Real>: AddAssign + SubAssign + MulAssign + DivAssign + DivAssign<T::Real>,
        T: ComplexFloat + AddAssign + DivAssign + Into<Complex<T::Real>>,
        [(); N - 1]:;
    fn rpolynomial_roots(&self) -> [Complex<T::Real>; N - 1]
    where
        Complex<T::Real>: AddAssign + SubAssign + MulAssign + DivAssign + DivAssign<T::Real>,
        T: ComplexFloat + AddAssign + DivAssign + Into<Complex<T::Real>>,
        [(); N - 1]:;

    fn polyfit<Y, Z, const M: usize>(&self, y: &[Y; N]) -> [Z; M]
    where
        Z: ComplexFloat + AddAssign + SubAssign + DivAssign + Div<Z::Real, Output = Z>,
        T: ComplexFloat + AddAssign + SubAssign + DivAssign + DivAssign<T::Real> + Mul<Y, Output = Z> + Into<Z>,
        Y: Copy,
        [(); max_len(M, M)]:,
        [(); max_len(N, N)]:;
    fn rpolyfit<Y, Z, const M: usize>(&self, y: &[Y; N]) -> [Z; M]
    where
        Z: ComplexFloat + AddAssign + SubAssign + DivAssign + Div<Z::Real, Output = Z>,
        T: ComplexFloat + AddAssign + SubAssign + DivAssign + DivAssign<T::Real> + Mul<Y, Output = Z> + Into<Z>,
        Y: Copy,
        [(); max_len(M, M)]:,
        [(); max_len(N, N)]:;
    fn linfit<Y, Z>(&self, y: &[Y; N]) -> [Z; 2]
    where
        Z: ComplexFloat + AddAssign + SubAssign + DivAssign + Div<Z::Real, Output = Z>,
        T: ComplexFloat + AddAssign + SubAssign + DivAssign + DivAssign<T::Real> + Mul<Y, Output = Z> + Into<Z>,
        Y: Copy,
        [(); max_len(N, N)]:;
    fn rlinfit<Y, Z>(&self, y: &[Y; N]) -> [Z; 2]
    where
        Z: ComplexFloat + AddAssign + SubAssign + DivAssign + Div<Z::Real, Output = Z>,
        T: ComplexFloat + AddAssign + SubAssign + DivAssign + DivAssign<T::Real> + Mul<Y, Output = Z> + Into<Z>,
        Y: Copy,
        [(); max_len(N, N)]:;

    fn detrend<const M: usize>(&mut self)
    where
        T: ComplexFloat + AddAssign + SubAssign + MulAssign + DivAssign + Div<T::Real, Output = T> + DivAssign<T::Real> + Mul<Output = T>,
        [(); max_len(M, M)]:,
        [(); max_len(N, N)]:;

    /// Performs direct convolution.
    /// This is equivalent to a polynomial multiplication.
    /// 
    /// # Examples
    /// 
    /// Convolving a unit impulse yields the impulse response.
    /// 
    /// ```rust
    /// #![feature(generic_const_exprs)]
    /// 
    /// use array_math::*;
    /// 
    /// let x = [1.0];
    /// let h = [1.0, 0.6, 0.3];
    /// 
    /// let y = x.convolve_direct(&h);
    /// 
    /// assert_eq!(y, h);
    /// ```
    /// 
    /// Convolution can be done directly `O(n^2)` or using FFT `O(nlog(n))`.
    /// 
    /// ```rust
    /// #![feature(generic_arg_infer)]
    /// #![feature(generic_const_exprs)]
    /// 
    /// use array_math::*;
    /// 
    /// let x = [1.0, 0.0, 1.5, 0.0, 0.0, -1.0];
    /// let h = [1.0, 0.6, 0.3];
    /// 
    /// let y_fft = x.convolve_real_fft(h);
    /// let y_direct = x.convolve_direct(&h);
    /// 
    /// let avg_error = y_fft.comap(y_direct, |y_fft: f64, y_direct: f64| (y_fft - y_direct).abs()).avg();
    /// assert!(avg_error < 1.0e-15);
    /// ```
    fn convolve_direct<Rhs, const M: usize>(&self, rhs: &[Rhs; M]) -> [<T as Mul<Rhs>>::Output; N + M - 1]
    where
        T: Mul<Rhs, Output: AddAssign + Zero> + Copy,
        Rhs: Copy;

    /// Performs convolution using FFT.
    /// 
    /// # Examples
    /// 
    /// Convolution can be done directly `O(n^2)` or using FFT `O(nlog(n))`.
    /// 
    /// ```rust
    /// #![feature(generic_arg_infer)]
    /// #![feature(generic_const_exprs)]
    /// 
    /// use array_math::*;
    /// 
    /// let x = [1.0, 0.0, 1.5, 0.0, 0.0, -1.0];
    /// let h = [1.0, 0.6, 0.3];
    /// 
    /// let y_fft = x.convolve_real_fft(h);
    /// let y_direct = x.convolve_direct(&h);
    /// 
    /// let avg_error = y_fft.comap(y_direct, |y_fft: f64, y_direct: f64| (y_fft - y_direct).abs()).avg();
    /// assert!(avg_error < 1.0e-15);
    /// ```
    fn convolve_real_fft<Rhs, const M: usize>(self, rhs: [Rhs; M]) -> [<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real; N + M - 1]
    where
        T: Float,
        Rhs: Float,
        Complex<T>: MulAssign + AddAssign + ComplexFloat<Real = T> + Mul<Complex<Rhs>, Output: ComplexFloat<Real: Float>>,
        Complex<Rhs>: MulAssign + AddAssign + ComplexFloat<Real = Rhs>,
        <Complex<T> as Mul<Complex<Rhs>>>::Output: ComplexFloat<Real: Float> + Into<Complex<<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>>,
        Complex<<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>: MulAssign + AddAssign + MulAssign<<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real> + ComplexFloat<Real = <<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>,
        [(); (N + M - 1).next_power_of_two() - N]:,
        [(); (N + M - 1).next_power_of_two() - M]:,
        [(); (N + M - 1).next_power_of_two() - (N + M - 1)]:,
        [(); (N + M - 1).next_power_of_two()/2 + 1]:;
        
    fn convolve_fft<Rhs, const M: usize>(self, rhs: [Rhs; M]) -> [<T as Mul<Rhs>>::Output; N + M - 1]
    where
        T: ComplexFloat + Mul<Rhs, Output: ComplexFloat<Real: Into<<T as Mul<Rhs>>::Output>> + 'static> + Into<Complex<T::Real>>,
        Rhs: ComplexFloat + Into<Complex<Rhs::Real>>,
        Complex<<<T as Mul<Rhs>>::Output as ComplexFloat>::Real>: Into<<Complex<T::Real> as Mul<Complex<Rhs::Real>>>::Output>,
        Complex<T::Real>: AddAssign + MulAssign + Mul<Complex<Rhs::Real>, Output: ComplexFloat<Real = <<T as Mul<Rhs>>::Output as ComplexFloat>::Real> + MulAssign + AddAssign + MulAssign<<<T as Mul<Rhs>>::Output as ComplexFloat>::Real> + Sum + 'static>,
        Complex<Rhs::Real>: AddAssign + MulAssign,
        [(); (N + M - 1).next_power_of_two() - N]:,
        [(); (N + M - 1).next_power_of_two() - M]:,
        [(); (N + M - 1).next_power_of_two() - (N + M - 1)]:;
        
    fn cconvolve_direct<Rhs, const M: usize>(&self, rhs: &[Rhs; M]) -> [<T as Mul<Rhs>>::Output; max_len(N, M)]
    where
        T: Mul<Rhs, Output: AddAssign + Zero> + Copy,
        Rhs: Copy;
    fn cconvolve_real_fft<Rhs, const M: usize>(self, rhs: [Rhs; M]) -> [<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real; max_len(N, M)]
    where
        T: Float,
        Rhs: Float,
        Complex<T>: MulAssign + AddAssign + ComplexFloat<Real = T> + Mul<Complex<Rhs>, Output: ComplexFloat<Real: Float>>,
        Complex<Rhs>: MulAssign + AddAssign + ComplexFloat<Real = Rhs>,
        <Complex<T> as Mul<Complex<Rhs>>>::Output: ComplexFloat<Real: Float> + Into<Complex<<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>>,
        Complex<<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>: MulAssign + AddAssign + MulAssign<<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real> + ComplexFloat<Real = <<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>,
        [(); max_len(N, M)/2 + 1]:;
    fn cconvolve_fft<Rhs, const M: usize>(self, rhs: [Rhs; M]) -> [<T as Mul<Rhs>>::Output; max_len(N, M)]
    where
        T: ComplexFloat + Mul<Rhs, Output: ComplexFloat<Real: Into<<T as Mul<Rhs>>::Output>> + 'static> + Into<Complex<T::Real>>,
        Rhs: ComplexFloat + Into<Complex<Rhs::Real>>,
        Complex<<<T as Mul<Rhs>>::Output as ComplexFloat>::Real>: Into<<Complex<T::Real> as Mul<Complex<Rhs::Real>>>::Output>,
        Complex<T::Real>: AddAssign + MulAssign + Mul<Complex<Rhs::Real>, Output: ComplexFloat<Real = <<T as Mul<Rhs>>::Output as ComplexFloat>::Real> + MulAssign + AddAssign + MulAssign<<<T as Mul<Rhs>>::Output as ComplexFloat>::Real> + Sum + 'static>,
        Complex<Rhs::Real>: AddAssign + MulAssign;
        
    fn deconvolve_direct<Rhs, const M: usize>(self, rhs: [Rhs; M]) -> ([<T as Div<Rhs>>::Output; N + 1 - M], [T; M - 1])
    where
        T: Div<Rhs, Output: Zero + Mul<Rhs, Output: Zero + AddAssign + Copy> + Copy> + Sub<<<T as Div<Rhs>>::Output as Mul<Rhs>>::Output, Output = T> + Zero + Copy,
        Rhs: Copy + Zero;

    fn recip_all(self) -> [<T as Inv>::Output; N]
    where
        T: Inv;
    fn recip_assign_all(&mut self)
    where
        T: Inv<Output = T>;
        
    fn conj_all(self) -> Self
    where
        T: ComplexFloat;
    fn conj_assign_all(&mut self)
    where
        T: ComplexFloat;

    fn dtft(&self, omega: T::Real) -> Complex<T::Real>
    where
        T: ComplexFloat + Into<Complex<T::Real>>,
        Complex<T::Real>: ComplexFloat<Real = T::Real> + MulAssign + AddAssign;

    #[doc(hidden)]
    fn fft_unscaled<const I: bool>(&mut self)
    where
        T: ComplexFloat<Real: Float> + MulAssign + AddAssign + Sum,
        Complex<T::Real>: Into<T>;

    /// Performs an iterative, in-place radix-2 FFT algorithm as described in https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm#Data_reordering,_bit_reversal,_and_in-place_algorithms.
    /// If `N` is not a power of two, it uses the DFT, which is a lot slower.
    /// 
    /// # Examples
    /// ```rust
    /// #![feature(generic_const_exprs)]
    /// 
    /// use num::Complex;
    /// use array_math::*;
    /// 
    /// let x = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    ///     .map(|x| <Complex<_> as From<_>>::from(x));
    /// 
    /// let mut y = x;
    /// 
    /// y.fft();
    /// y.ifft();
    /// 
    /// let avg_error = x.comap(y, |x, y| (x - y).norm()).avg();
    /// assert!(avg_error < 1.0e-16);
    /// ```
    fn fft(&mut self)
    where
        T: ComplexFloat<Real: Float> + MulAssign + AddAssign + Sum,
        Complex<T::Real>: Into<T>;
        
    /// Performs an iterative, in-place radix-2 IFFT algorithm as described in https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm#Data_reordering,_bit_reversal,_and_in-place_algorithms.
    /// If `N` is not a power of two, it uses the IDFT, which is a lot slower.
    /// 
    /// # Examples
    /// ```rust
    /// #![feature(generic_const_exprs)]
    /// 
    /// use num::Complex;
    /// use array_math::*;
    /// 
    /// let x = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    ///     .map(|x| <Complex<_> as From<_>>::from(x));
    /// 
    /// let mut y = x;
    /// 
    /// y.fft();
    /// y.ifft();
    /// 
    /// let avg_error = x.comap(y, |x, y| (x - y).norm()).avg();
    /// assert!(avg_error < 1.0e-16);
    /// ```
    fn ifft(&mut self)
    where
        T: ComplexFloat<Real: Float> + MulAssign + AddAssign + Sum + MulAssign<T::Real>,
        Complex<T::Real>: Into<T>;
        
    /// Walsh-Hadamard transform
    fn fwht_unscaled(&mut self)
    where
        T: Add<Output = T> + Sub<Output = T> + Copy,
        [(); N.is_power_of_two() as usize - 1]:;
    /// Normalized Walsh-Hadamard transform
    fn fwht(&mut self)
    where
        T: ComplexFloat + MulAssign<T::Real>,
        [(); N.is_power_of_two() as usize - 1]:;
    /// Normalized inverse Walsh-Hadamard transform
    fn ifwht(&mut self)
    where
        T: ComplexFloat + MulAssign<T::Real>,
        [(); N.is_power_of_two() as usize - 1]:;
        
    fn fht(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign;
    fn ifht(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + MulAssign<T::Real>;
        
    fn dst_i(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + DivAssign<T::Real>;
    fn dst_ii(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign;
    fn dst_iii(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + MulAssign<T::Real> + Mul<T, Output = Complex<T::Real>>;
    fn dst_iv(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + Mul<T, Output = Complex<T::Real>>;
        
    fn dct_i(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + DivAssign<T::Real> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + DivAssign<T::Real>;
    fn dct_ii(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign;
    fn dct_iii(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + Mul<T, Output = Complex<T::Real>> + DivAssign<T::Real>;
    fn dct_iv(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + Mul<T, Output = Complex<T::Real>>;
    
    /// Performs the FFT on an array of real floating-point numbers of length `N`.
    /// The result is an array of complex numbers of length `N/2 + 1`.
    /// This is truncated because the last half of the values are redundant, since they are a conjugate mirror-image of the first half.
    /// if `N` is not a power of two, the naive DFT is used instead, which is a lot slower.
    /// 
    /// # Examples
    /// ```rust
    /// #![feature(generic_arg_infer)]
    /// #![feature(generic_const_exprs)]
    /// 
    /// use num::{Complex, Zero};
    /// use array_math::*;
    /// 
    /// let x = [1.0, 1.0, 0.0, 0.0];
    /// 
    /// let mut z = [Complex::zero(); _];
    /// x.real_fft(&mut z);
    /// 
    /// let mut y = [0.0; _];
    /// y.real_ifft(&z);
    /// 
    /// assert_eq!(x, y);
    /// ```
    fn real_fft(&self, y: &mut [Complex<T>; N/2 + 1])
    where
        T: Float,
        Complex<T>: ComplexFloat<Real = T> + MulAssign + AddAssign;
        
    /// Performs the IFFT on a truncated array of complex floating-point numbers of length `N/2 + 1`.
    /// The result is an array of real numbers of length `N`.
    /// if `N` is not a power of two, the naive IDFT is used instead, which is a lot slower.
    /// 
    /// # Examples
    /// ```rust
    /// #![feature(generic_arg_infer)]
    /// #![feature(generic_const_exprs)]
    /// 
    /// use num::{Complex, Zero};
    /// use array_math::*;
    /// 
    /// let x = [1.0, 1.0, 0.0, 0.0];
    /// 
    /// let mut z = [Complex::zero(); _];
    /// x.real_fft(&mut z);
    /// 
    /// let mut y = [0.0; _];
    /// y.real_ifft(&z);
    /// 
    /// assert_eq!(x, y);
    /// ```
    fn real_ifft(&mut self, x: &[Complex<T>; N/2 + 1])
    where
        T: Float,
        Complex<T>: ComplexFloat<Real = T> + MulAssign + AddAssign + MulAssign<T>;

    fn chebyshev_polynomial(kind: usize, order: usize) -> Option<[T; N]>
    where
        T: Copy + Add<Output = T> + Sub<Output = T> + Neg<Output = T> + AddAssign + Mul<Output = T> + One + Zero;
        
    fn bartlett_window() -> Self
    where
        T: Float;

    fn parzen_window() -> Self
    where
        T: Float;
        
    fn welch_window() -> Self
    where
        T: Float;
        
    fn sine_window() -> Self
    where
        T: Float + FloatConst;
    
    fn power_of_sine_window<A>(alpha: A) -> Self
    where
        T: Float + FloatConst + Pow<A, Output = T>,
        A: Copy;
        
    fn hann_window() -> Self
    where
        T: Float + FloatConst;
        
    fn hamming_window() -> Self
    where
        T: Float + FloatConst;
    
    fn blackman_window() -> Self
    where
        T: Float + FloatConst;

    fn nuttal_window() -> Self
    where
        T: Float + FloatConst;

    fn blackman_nuttal_window() -> Self
    where
        T: Float + FloatConst;

    fn blackman_harris_window() -> Self
    where
        T: Float + FloatConst;

    fn flat_top_window() -> Self
    where
        T: Float + FloatConst;
        
    fn frac_rotate_right(&mut self, shift: T::Real)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<<T>::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + MulAssign<T::Real>;
    fn frac_rotate_left(&mut self, shift: T::Real)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<<T>::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + MulAssign<T::Real>;
}

impl<T, const N: usize> ArrayMath<T, N> for [T; N]
{
    fn sum(self) -> T
    where
        T: AddAssign + Zero
    {
        //self.sum_from(T::ZERO)
        let sum = self.try_sum();
        if sum.is_some()
        {
            sum.unwrap()
        }
        else
        {
            core::mem::forget(sum);
            Zero::zero()
        }
    }

    fn product(self) -> T
    where
        T: MulAssign + One
    {
        //self.product_from(T::ONE)
        let product = self.try_product();
        if product.is_some()
        {
            product.unwrap()
        }
        else
        {
            core::mem::forget(product);
            One::one()
        }
    }
    
    fn variance(self) -> <T as Mul>::Output
    where
        Self: Copy,
        T: Div<Output: Mul<Output: Neg<Output = <T as Mul>::Output>> + Copy> + Mul<Output: AddAssign> + AddAssign + Zero + NumCast
    {
        let mu = self.avg();
        self.mul_dot_bias(self, -(mu*mu))
    }
    
    fn avg(self) -> <T as Div>::Output
    where
        T: Div + AddAssign + Zero + NumCast
    {
        self.sum()/T::from(N).unwrap()
    }
    
    fn geometric_mean(self) -> <T as Pow<<T as Inv>::Output>>::Output
    where
        T: MulAssign + One + Pow<<T as Inv>::Output> + Inv + NumCast
    {
        self.product().pow(T::from(N).unwrap().inv())
    }

    fn mul_dot<Rhs>(self, rhs: [Rhs; N]) -> <T as Mul<Rhs>>::Output
    where
        T: Mul<Rhs, Output: AddAssign + Zero>
    {
        let product = self.try_mul_dot(rhs);
        if product.is_some()
        {
            product.unwrap()
        }
        else
        {
            core::mem::forget(product);
            Zero::zero()
        }
    }

    fn magnitude_squared(self) -> <T as Mul<T>>::Output
    where
        T: Mul<T, Output: AddAssign + Zero> + Copy
    {
        self.mul_dot(self)
    }
    fn magnitude_squared_complex(self) -> T::Real
    where
        T: ComplexFloat + AddAssign + Copy
    {
        self.conj_all().mul_dot(self).abs()
    }

    fn magnitude(self) -> <T as Mul<T>>::Output
    where
        T: Mul<T, Output: AddAssign + Zero + Float> + Copy
    {
        //const N: usize = 3;
        self.magnitude_squared()
            .sqrt()
    }
    fn magnitude_complex(self) -> T::Real
    where
        T: ComplexFloat + AddAssign + Copy
    {
        Float::sqrt(self.magnitude_squared_complex())
    }
    
    fn magnitude_inv(self) -> <T as Mul<T>>::Output
    where
        T: Mul<T, Output: AddAssign + Zero + Float> + Copy
    {
        //const N: usize = 4;
        self.magnitude_squared()
            .sqrt()
            .recip()
    }
    fn magnitude_inv_complex(self) -> T::Real
    where
        T: ComplexFloat + AddAssign + Copy
    {
        Float::recip(Float::sqrt(self.magnitude_squared_complex()))
    }

    fn normalize(self) -> [<T as Mul<<T as Mul<T>>::Output>>::Output; N]
    where
        T: Mul<T, Output: AddAssign + Zero + Float + Copy> + Mul<<T as Mul<T>>::Output> + Copy
    {
        self.mul_all(self.magnitude_inv())
    }
    fn normalize_complex(self) -> [<T as Mul<T::Real>>::Output; N]
    where
        T: ComplexFloat + AddAssign + Mul<T::Real> + Copy
    {
        self.mul_all(self.magnitude_inv_complex())
    }

    fn normalize_to<Rhs>(self, magnitude: Rhs) -> [<T as Mul<<<T as Mul<T>>::Output as Mul<Rhs>>::Output>>::Output; N]
    where
        T: Mul<T, Output: AddAssign + Zero + Float + Mul<Rhs, Output: Copy>> + Mul<<<T as Mul<T>>::Output as Mul<Rhs>>::Output> + Copy
    {
        self.mul_all(self.magnitude_inv()*magnitude)
    }
    fn normalize_complex_to<Rhs>(self, magnitude: Rhs) -> [<T as Mul<<T::Real as Mul<Rhs>>::Output>>::Output; N]
    where
        T: ComplexFloat + AddAssign + Mul<<T::Real as Mul<Rhs>>::Output>,
        T::Real: Mul<Rhs, Output: Copy>
    {
        self.mul_all(<T::Real as Mul<Rhs>>::mul(self.magnitude_inv_complex(), magnitude))
    }
    
    fn normalize_assign(&mut self)
    where
        T: Mul<T, Output: AddAssign + Zero + Float + Copy> + MulAssign<<T as Mul<T>>::Output> + Copy
    {
        self.mul_assign_all(self.magnitude_inv())
    }
    fn normalize_assign_complex(&mut self)
    where
        T: ComplexFloat + AddAssign + MulAssign<T::Real> + Copy
    {
        self.mul_assign_all(self.magnitude_inv_complex())
    }

    fn normalize_assign_to<Rhs>(&mut self, magnitude: Rhs)
    where
        T: Mul<T, Output: AddAssign + Zero + Float + Mul<Rhs, Output: Copy>> + MulAssign<<<T as Mul<T>>::Output as Mul<Rhs>>::Output> + Copy
    {
        self.mul_assign_all(self.magnitude_inv()*magnitude)
    }
    fn normalize_assign_complex_to<Rhs>(&mut self, magnitude: Rhs)
    where
        T: ComplexFloat + AddAssign + MulAssign<<T::Real as Mul<Rhs>>::Output>,
        T::Real: Mul<Rhs, Output: Copy>
    {
        self.mul_assign_all(<T::Real as Mul<Rhs>>::mul(self.magnitude_inv_complex(), magnitude))
    }
    
    fn ellipke(mut self, tol: Option<T>) -> Option<([T; N], [T; N])>
    where
        T: Float + FloatConst + AddAssign + MulAssign
    {
        let one = T::one();
    
        if self.iter()
            .any(|m| m > &one)
        {
            return None
        }
    
        let tol = tol.unwrap_or_else(T::epsilon);
    
        let zero = T::zero();
        let two = one + one;
        let half = two.recip();
    
        let mut k = [zero; N];
        let mut e = [zero; N];
    
        let mut idx = vec![];
        for (i, &m) in self.iter()
            .enumerate()
        {
            if m == one
            {
                k[i] = T::infinity();
                e[i] = one;
            }
            else if m == T::neg_infinity()
            {
                k[i] = zero;
                e[i] = T::infinity();
            }
            else
            {
                idx.push(i);
            }
        }
    
        const N_MAX: usize = 16;
    
        if !idx.is_empty()
        {
            let idx_neg: Vec<_> = idx.iter()
                .filter(|&&i| self[i] < zero)
                .map(|&i| i)
                .collect();
            let mult_e: Vec<_> = idx_neg.iter()
                .map(|&i| (one - self[i]).sqrt())
                .collect();
            let mult_k: Vec<_> = mult_e.iter()
                .map(|&e| e.recip())
                .collect();
            for &i in idx_neg.iter()
            {
                self[i] = -self[i]/(one - self[i])
            }
            let mut b: Vec<_> = idx.iter()
                .map(|&i| (one - self[i]).sqrt())
                .collect();
            let mut a = vec![one; idx.len()];
            let mut c: Vec<_> = idx.iter()
                .map(|&i| self[i].sqrt())
                .collect();
            let mut f = half;
            let mut sum: Vec<_> = c.into_iter()
                .map(|c| f*c*c)
                .collect();
            let mut n = 2;
            while n <= N_MAX
            {
                let t = a.iter()
                    .zip(b.iter())
                    .map(|(&a, &b)| (a + b)*half)
                    .collect();
                c = a.iter()
                    .zip(b.iter())
                    .map(|(&a, &b)| (a - b)*half)
                    .collect();
                b = a.into_iter()
                    .zip(b.into_iter())
                    .map(|(a, b)| (a*b).sqrt())
                    .collect();
                a = t;
                f *= two;
                let mut done = true;
                for ((s, c), &a) in sum.iter_mut()
                    .zip(c.into_iter())
                    .zip(a.iter())
                {
                    *s += f*c*c;
                    if done && c > tol*a
                    {
                        done = false
                    }
                }
                if done
                {
                    break
                }
    
                n += 1;
            }
            if n >= N_MAX
            {
                return None
            }
            for ((i, a), sum) in idx.into_iter()
                .zip(a.into_iter())
                .zip(sum.into_iter())
            {
                k[i] = T::FRAC_PI_2()/a;
                e[i] = k[i]*(one - sum);
            }
            for ((i, mk), me) in idx_neg.into_iter()
                .zip(mult_k.into_iter())
                .zip(mult_e.into_iter())
            {
                k[i] *= mk;
                e[i] *= me;
            }
        }
    
        Some((k, e))
    }

    fn polynomial<Rhs>(self, rhs: Rhs) -> T
    where
        T: AddAssign + MulAssign<Rhs> + Zero,
        Rhs: Copy
    {
        let ptr = self.as_ptr();
        let mut y = T::zero();
        let mut i = N;
        while i > 0
        {
            i -= 1;
            y *= rhs;
            y += unsafe {
                ptr.add(i).read()
            };
        }
        core::mem::forget(self);
        y
    }
    fn rpolynomial<Rhs>(self, rhs: Rhs) -> T
    where
        T: AddAssign + MulAssign<Rhs> + Zero,
        Rhs: Copy
    {
        let ptr = self.as_ptr();
        let mut y = T::zero();
        let mut i = 0;
        while i < N
        {
            y *= rhs;
            y += unsafe {
                ptr.add(i).read()
            };
            i += 1;
        }
        core::mem::forget(self);
        y
    }

    fn derivate_polynomial(self) -> [<T as Mul>::Output; N - 1]
    where
        T: NumCast + Zero + Mul
    {
        let ptr = self.as_ptr();
        let y = ArrayOps::fill(|i| {
            let b = unsafe {
                ptr.add(i + 1).read()
            };
            b*T::from(i + 1).unwrap()
        });
        core::mem::forget(self);
        y
    }
    fn derivate_rpolynomial(self) -> [<T as Mul>::Output; N - 1]
    where
        T: NumCast + Zero + Mul
    {
        let ptr = self.as_ptr();
        let y = ArrayOps::fill(|i| {
            let b = unsafe {
                ptr.add(i).read()
            };
            b*T::from(N - i - 1).unwrap()
        });
        core::mem::forget(self);
        y
    }
    
    fn integrate_polynomial(self, c: <T as Div>::Output) -> [<T as Div>::Output; N + 1]
    where
        T: NumCast + Zero + Div
    {
        let c_ptr = &c as *const <T as Div>::Output;
        let ptr = self.as_ptr();
        let y = ArrayOps::fill(|i| {
            if i == 0
            {
                unsafe {
                    c_ptr.read()
                }
            }
            else
            {
                let b = unsafe {
                    ptr.add(i - 1).read()
                };
                b/T::from(i).unwrap()
            }
        });
        core::mem::forget(self);
        core::mem::forget(c);
        y
    }
    fn integrate_rpolynomial(self, c: <T as Div>::Output) -> [<T as Div>::Output; N + 1]
    where
        T: NumCast + Zero + Div
    {
        let c_ptr = &c as *const <T as Div>::Output;
        let ptr = self.as_ptr();
        let y = ArrayOps::fill(|i| {
            if i == N
            {
                unsafe {
                    c_ptr.read()
                }
            }
            else
            {
                let b = unsafe {
                    ptr.add(i).read()
                };
                b/T::from(N - i).unwrap()
            }
        });
        core::mem::forget(self);
        core::mem::forget(c);
        y
    }

    fn companion_matrix(&self) -> [[<T as Neg>::Output; N - 1]; N - 1]
    where
        T: Copy + Neg + Zero,
        <T as Neg>::Output: One + Zero + DivAssign<T>
    {
        let mut c = <[[_; N - 1]; N - 1]>::fill(|_| ArrayOps::fill(|_| Zero::zero()));
        let mut n = N - 1;
        while n > 0
        {
            if !self[n].is_zero()
            {
                break
            }
            n -= 1;
        }
        if n == 0
        {
            return c
        }
        let mut i = 0;
        while i < n
        {
            if i > 0
            {
                c[i][i - 1] = One::one();
            }
            c[i][n - 1] = -self[i];
            c[i][n - 1] /= self[n];
            i += 1;
        }
        c
    }
    fn rcompanion_matrix(&self) -> [[<T as Neg>::Output; N - 1]; N - 1]
    where
        T: Copy + Neg + Zero,
        <T as Neg>::Output: One + Zero + DivAssign<T>
    {
        let mut c = <[[_; N - 1]; N - 1]>::fill(|_| ArrayOps::fill(|_| Zero::zero()));
        let mut n = N - 1;
        while n > 0
        {
            if !self[N - 1 - n].is_zero()
            {
                break
            }
            n -= 1;
        }
        if n == 0
        {
            return c
        }
        let mut i = n;
        loop
        {
            c[n - i][n - 1] = -self[i];
            c[n - i][n - 1] /= self[N - 1 - n];
            i -= 1;
            if i > 0
            {
                c[i][i - 1] = One::one();
            }
            else
            {
                break
            }
        }
        c
    }
    fn vandermonde_matrix<const M: usize>(&self) -> [[T; M]; N]
    where
        T: One + Copy + Mul
    {
        let mut m = [[T::one(); M]; N];
        for j in (0..M - 1).rev()
        {
            for k in 0..N
            {
                m[k][j] = self[k]*m[k][j + 1]
            }
        }
        m
    }

    fn polynomial_roots(&self) -> [Complex<T::Real>; N - 1]
    where
        Complex<T::Real>: AddAssign + SubAssign + MulAssign + DivAssign + DivAssign<T::Real>,
        T: ComplexFloat + AddAssign + DivAssign + Into<Complex<T::Real>>,
        [(); N - 1]:
    {
        let scale = self.magnitude_complex();
        let c = self.companion_matrix();
        let mut roots = c.eigenvalues();
        // Use newtons method
        let p: [Complex<T::Real>; _] = self.map(|p| p.into());
        let dp = p.derivate_polynomial();
        for k in 0..N - 1
        {
            const NEWTON: usize = NEWTON_POLYNOMIAL_ROOTS;

            for _ in 0..NEWTON
            {
                let df = p.polynomial(roots[k]);
                if df.is_zero()
                {
                    break
                }
                roots[k] -= df/dp.polynomial(roots[k])
            }
        }
        let mut excess = 0;
        while excess < N
        {
            if !self[N - 1 - excess].is_zero()
            {
                break
            }
            excess += 1;
        }
        // Even out duplicate roots
        for k in 0..N - 1
        {
            if !roots[k].is_nan()
            {
                let mut j = 1;
                for i in 0..N - 1
                {
                    if excess > 0 && i != k && !roots[i].is_nan()
                    {
                        if (roots[i] - roots[k]).abs() < scale*T::Real::epsilon()
                        {
                            roots[k] += roots[i];
                            j += 1;
                            roots[i] = From::from(T::Real::nan());
                            excess -= 1;
                        }
                    }
                }
                if j > 1
                {
                    roots[k] /= <T::Real as NumCast>::from(j).unwrap();
                }
            }
        }
        roots
    }
    fn rpolynomial_roots(&self) -> [Complex<T::Real>; N - 1]
    where
        Complex<T::Real>: AddAssign + SubAssign + MulAssign + DivAssign + DivAssign<T::Real>,
        T: ComplexFloat + AddAssign + DivAssign + Into<Complex<T::Real>>,
        [(); N - 1]:
    {
        let scale = self.magnitude_complex();
        let c = self.rcompanion_matrix();
        let mut roots = c.eigenvalues();
        // Use newtons method
        let p: [Complex<T::Real>; _] = self.map(|p| p.into());
        let dp = p.derivate_rpolynomial();
        for k in 0..N - 1
        {
            const NEWTON: usize = NEWTON_POLYNOMIAL_ROOTS;

            for _ in 0..NEWTON
            {
                let df = p.rpolynomial(roots[k]);
                if df.is_zero()
                {
                    break
                }
                roots[k] -= df/dp.rpolynomial(roots[k])
            }
        }
        let mut excess = 0;
        while excess < N
        {
            if !self[excess].is_zero()
            {
                break
            }
            excess += 1;
        }
        // Even out duplicate roots
        for k in 0..N - 1
        {
            if !roots[k].is_nan()
            {
                let mut j = 1;
                for i in 0..N - 1
                {
                    if excess > 0 && i != k && !roots[i].is_nan()
                    {
                        if (roots[i] - roots[k]).abs() < scale*T::Real::epsilon()
                        {
                            roots[k] += roots[i];
                            j += 1;
                            roots[i] = From::from(T::Real::nan());
                            excess -= 1;
                        }
                    }
                }
                if j > 1
                {
                    roots[k] /= <T::Real as NumCast>::from(j).unwrap();
                }
            }
        }
        roots
    }
    
    fn polyfit<Y, Z, const M: usize>(&self, y: &[Y; N]) -> [Z; M]
    where
        Z: ComplexFloat + AddAssign + SubAssign + DivAssign + Div<Z::Real, Output = Z>,
        T: ComplexFloat + AddAssign + SubAssign + DivAssign + DivAssign<T::Real> + Mul<Y, Output = Z> + Into<Z>,
        Y: Copy,
        [(); max_len(M, M)]:,
        [(); max_len(N, N)]:
    {
        let mut p = self.rpolyfit(y);
        p.reverse();
        p
    }
    fn rpolyfit<Y, Z, const M: usize>(&self, y: &[Y; N]) -> [Z; M]
    where
        Z: ComplexFloat + AddAssign + SubAssign + DivAssign + Div<Z::Real, Output = Z>,
        T: ComplexFloat + AddAssign + SubAssign + DivAssign + DivAssign<T::Real> + Mul<Y, Output = Z> + Into<Z>,
        Y: Copy,
        [(); max_len(M, M)]:,
        [(); max_len(N, N)]:
    {
        let v = self.vandermonde_matrix::<M>();

        let (q, r) = v.qr_matrix();
        let qtmy = q.transpose().mul_matrix(y.as_collumn());
        let p = r.map(|r| r.map(|r| r.into())).solve_matrix(qtmy.as_uncollumn());

        p
    }
    fn linfit<Y, Z>(&self, y: &[Y; N]) -> [Z; 2]
    where
        Z: ComplexFloat + AddAssign + SubAssign + DivAssign + Div<Z::Real, Output = Z>,
        T: ComplexFloat + AddAssign + SubAssign + DivAssign + DivAssign<T::Real> + Mul<Y, Output = Z> + Into<Z>,
        Y: Copy,
        [(); max_len(N, N)]:
    {
        self.polyfit::<_, _, 2>(y)
    }
    fn rlinfit<Y, Z>(&self, y: &[Y; N]) -> [Z; 2]
    where
        Z: ComplexFloat + AddAssign + SubAssign + DivAssign + Div<Z::Real, Output = Z>,
        T: ComplexFloat + AddAssign + SubAssign + DivAssign + DivAssign<T::Real> + Mul<Y, Output = Z> + Into<Z>,
        Y: Copy,
        [(); max_len(N, N)]:
    {
        self.rpolyfit::<_, _, 2>(y)
    }
    
    fn detrend<const M: usize>(&mut self)
    where
        T: ComplexFloat + AddAssign + SubAssign + MulAssign + DivAssign + Div<T::Real, Output = T> + DivAssign<T::Real> + Mul<Output = T>,
        [(); max_len(M, M)]:,
        [(); max_len(N, N)]:
    {
        let x = core::array::from_fn(|i| <T as NumCast>::from(i).unwrap());
        let p: [_; M] = x.rpolyfit(self);
        for i in 0..self.len()
        {
            self[i] -= p.rpolynomial(x[i])
        }
    }
    
    fn convolve_direct<Rhs, const M: usize>(&self, rhs: &[Rhs; M]) -> [<T as Mul<Rhs>>::Output; N + M - 1]
    where
        T: Mul<Rhs, Output: AddAssign + Zero> + Copy,
        Rhs: Copy
    {
        ArrayOps::fill(|n| {
            let mut y = Zero::zero();
            for k in (n + 1).saturating_sub(N)..M.min(n + 1)
            {
                y += self[n - k]*rhs[k];
            }
            y
        })
    }
    
    fn convolve_real_fft<Rhs, const M: usize>(self, rhs: [Rhs; M]) -> [<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real; N + M - 1]
    where
        T: Float,
        Rhs: Float,
        Complex<T>: MulAssign + AddAssign + ComplexFloat<Real = T> + Mul<Complex<Rhs>, Output: ComplexFloat<Real: Float>>,
        Complex<Rhs>: MulAssign + AddAssign + ComplexFloat<Real = Rhs>,
        <Complex<T> as Mul<Complex<Rhs>>>::Output: ComplexFloat<Real: Float> + Into<Complex<<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>>,
        Complex<<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>: MulAssign + AddAssign + MulAssign<<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real> + ComplexFloat<Real = <<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>,
        [(); (N + M - 1).next_power_of_two() - N]:,
        [(); (N + M - 1).next_power_of_two() - M]:,
        [(); (N + M - 1).next_power_of_two() - (N + M - 1)]:,
        [(); (N + M - 1).next_power_of_two()/2 + 1]:
    {
        let x: [T; (N + M - 1).next_power_of_two()] = self.resize(|_| T::zero());
        let h: [Rhs; (N + M - 1).next_power_of_two()] = rhs.resize(|_| Rhs::zero());

        let mut x_f = [Complex::zero(); _];
        let mut h_f = [Complex::zero(); _];
        x.real_fft(&mut x_f);
        h.real_fft(&mut h_f);

        let y_f = x_f.comap(h_f, |x_f, h_f| (x_f*h_f).into());
        let mut y = [Zero::zero(); (N + M - 1).next_power_of_two()];
        y.real_ifft(&y_f);

        y.truncate()
    }
    
    fn convolve_fft<Rhs, const M: usize>(self, rhs: [Rhs; M]) -> [<T as Mul<Rhs>>::Output; N + M - 1]
    where
        T: ComplexFloat + Mul<Rhs, Output: ComplexFloat<Real: Into<<T as Mul<Rhs>>::Output>> + 'static> + Into<Complex<T::Real>>,
        Rhs: ComplexFloat + Into<Complex<Rhs::Real>>,
        Complex<<<T as Mul<Rhs>>::Output as ComplexFloat>::Real>: Into<<Complex<T::Real> as Mul<Complex<Rhs::Real>>>::Output>,
        Complex<T::Real>: AddAssign + MulAssign + Mul<Complex<Rhs::Real>, Output: ComplexFloat<Real = <<T as Mul<Rhs>>::Output as ComplexFloat>::Real> + MulAssign + AddAssign + MulAssign<<<T as Mul<Rhs>>::Output as ComplexFloat>::Real> + Sum + 'static>,
        Complex<Rhs::Real>: AddAssign + MulAssign,
        [(); (N + M - 1).next_power_of_two() - N]:,
        [(); (N + M - 1).next_power_of_two() - M]:,
        [(); (N + M - 1).next_power_of_two() - (N + M - 1)]:
    {
        let mut x: [Complex<T::Real>; (N + M - 1).next_power_of_two()] = self.map(Into::into).resize(|_| Zero::zero());
        let mut h: [Complex<Rhs::Real>; (N + M - 1).next_power_of_two()] = rhs.map(Into::into).resize(|_| Zero::zero());
        x.fft();
        h.fft();

        let mut y = x.comap(h, |x, h| x*h);
        y.ifft();

        y.truncate()
            .map(|y| {
                if let Some(y) = <dyn Any>::downcast_ref::<<T as Mul<Rhs>>::Output>(&y as &dyn Any)
                {
                    *y
                }
                else
                {
                    y.re().into()
                }
            })
    }
    
    fn cconvolve_direct<Rhs, const M: usize>(&self, rhs: &[Rhs; M]) -> [<T as Mul<Rhs>>::Output; max_len(N, M)]
    where
        T: Mul<Rhs, Output: AddAssign + Zero> + Copy,
        Rhs: Copy
    {
        let y_len = N.max(M);

        ArrayOps::fill(|n| {
            let mut y = Zero::zero();
            for k in 0..y_len
            {
                let i = k;
                let j = (n + y_len - k) % y_len;
                if i < self.len() && j < rhs.len()
                {
                    y += self[i]*rhs[j]
                }
            }
            y
        })
    }
    fn cconvolve_real_fft<Rhs, const M: usize>(self, rhs: [Rhs; M]) -> [<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real; max_len(N, M)]
    where
        T: Float,
        Rhs: Float,
        Complex<T>: MulAssign + AddAssign + ComplexFloat<Real = T> + Mul<Complex<Rhs>, Output: ComplexFloat<Real: Float>>,
        Complex<Rhs>: MulAssign + AddAssign + ComplexFloat<Real = Rhs>,
        <Complex<T> as Mul<Complex<Rhs>>>::Output: ComplexFloat<Real: Float> + Into<Complex<<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>>,
        Complex<<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>: MulAssign + AddAssign + MulAssign<<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real> + ComplexFloat<Real = <<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>,
        [(); max_len(N, M)/2 + 1]:
    {
        let x: [T; max_len(N, M)] = self.resize(|_| T::zero());
        let h: [Rhs; max_len(N, M)] = rhs.resize(|_| Rhs::zero());

        let mut x_f = [Complex::zero(); _];
        let mut h_f = [Complex::zero(); _];
        x.real_fft(&mut x_f);
        h.real_fft(&mut h_f);

        let y_f = x_f.comap(h_f, |x_f, h_f| (x_f*h_f).into());
        let mut y = [Zero::zero(); max_len(N, M)];
        y.real_ifft(&y_f);

        y
    }
    fn cconvolve_fft<Rhs, const M: usize>(self, rhs: [Rhs; M]) -> [<T as Mul<Rhs>>::Output; max_len(N, M)]
    where
        T: ComplexFloat + Mul<Rhs, Output: ComplexFloat<Real: Into<<T as Mul<Rhs>>::Output>> + 'static> + Into<Complex<T::Real>>,
        Rhs: ComplexFloat + Into<Complex<Rhs::Real>>,
        Complex<<<T as Mul<Rhs>>::Output as ComplexFloat>::Real>: Into<<Complex<T::Real> as Mul<Complex<Rhs::Real>>>::Output>,
        Complex<T::Real>: AddAssign + MulAssign + Mul<Complex<Rhs::Real>, Output: ComplexFloat<Real = <<T as Mul<Rhs>>::Output as ComplexFloat>::Real> + MulAssign + AddAssign + MulAssign<<<T as Mul<Rhs>>::Output as ComplexFloat>::Real> + Sum + 'static>,
        Complex<Rhs::Real>: AddAssign + MulAssign
    {
        let mut x: [Complex<T::Real>; max_len(N, M)] = self.map(Into::into).resize(|_| Zero::zero());
        let mut h: [Complex<Rhs::Real>; max_len(N, M)] = rhs.map(Into::into).resize(|_| Zero::zero());
        x.fft();
        h.fft();

        let mut y = x.comap(h, |x, h| x*h);
        y.ifft();

        y.map(|y| {
            if let Some(y) = <dyn Any>::downcast_ref::<<T as Mul<Rhs>>::Output>(&y as &dyn Any)
            {
                *y
            }
            else
            {
                y.re().into()
            }
        })
    }
    
    fn deconvolve_direct<Rhs, const M: usize>(self, rhs: [Rhs; M]) -> ([<T as Div<Rhs>>::Output; N + 1 - M], [T; M - 1])
    where
        T: Div<Rhs, Output: Zero + Mul<Rhs, Output: Zero + AddAssign + Copy> + Copy> + Sub<<<T as Div<Rhs>>::Output as Mul<Rhs>>::Output, Output = T> + Zero + Copy,
        Rhs: Copy + Zero
    {
        let (q, r): (Vec<_>, Vec<_>) = self.as_slice().deconvolve_direct(&rhs);
        (q.try_into().ok().unwrap(), r.try_into().ok().unwrap())
    }
    
    fn recip_all(self) -> [<T as Inv>::Output; N]
    where
        T: Inv
    {
        self.map(Inv::inv)
    }
    fn recip_assign_all(&mut self)
    where
        T: Inv<Output = T>
    {
        self.map_assign(Inv::inv)
    }

    fn conj_all(mut self) -> Self
    where
        T: ComplexFloat
    {
        self.conj_assign_all();
        self
    }
    fn conj_assign_all(&mut self)
    where
        T: ComplexFloat
    {
        let mut i = 0;
        while i < N
        {
            unsafe {
                let ptr = self.as_mut_ptr().add(i);
                ptr.write(ptr.read().conj());
            }
            i += 1;
        }
    }
    
    fn dtft(&self, omega: T::Real) -> Complex<T::Real>
    where
        T: ComplexFloat + Into<Complex<T::Real>>,
        Complex<T::Real>: ComplexFloat<Real = T::Real> + MulAssign + AddAssign
    {
        let mut y = Complex::zero();
        let z1 = Complex::cis(-omega);
        let mut z = Complex::one();
        for &x in self
        {
            y += x.into()*z;
            z *= z1;
        }
        y
    }
    
    fn fft_unscaled<const I: bool>(&mut self)
    where
        T: ComplexFloat<Real: Float> + MulAssign + AddAssign + Sum,
        Complex<T::Real>: Into<T>
    {
        self.as_mut_slice()
            .fft_unscaled::<I>()
    }
    
    fn fft(&mut self)
    where
        T: ComplexFloat<Real: Float> + MulAssign + AddAssign + Sum,
        Complex<T::Real>: Into<T>
    {
        self.fft_unscaled::<false>()
    }
    fn ifft(&mut self)
    where
        T: ComplexFloat<Real: Float> + MulAssign + AddAssign + Sum + MulAssign<T::Real>,
        Complex<T::Real>: Into<T>
    {
        self.fft_unscaled::<true>();

        self.mul_assign_all(<T::Real as NumCast>::from(1.0/N as f64).unwrap());
    }

    fn fwht_unscaled(&mut self)
    where
        T: Add<Output = T> + Sub<Output = T> + Copy,
        [(); N.is_power_of_two() as usize - 1]:
    {
        let mut h = 1;
        while h < N
        {
            for i in (0..N).step_by(h*2)
            {
                for j in i..i + h
                {
                    let x = self[j];
                    let y = self[j + h];
                    self[j] = x + y;
                    self[j + h] = x - y;
                }
            }
            h *= 2;
        }
    }
    fn fwht(&mut self)
    where
        T: ComplexFloat + MulAssign<T::Real>,
        [(); N.is_power_of_two() as usize - 1]:
    {
        self.fwht_unscaled();
        self.mul_assign_all(Float::powi(T::Real::FRAC_1_SQRT_2(), (self.len().ilog2() + 3).try_into().unwrap()))
    }
    fn ifwht(&mut self)
    where
        T: ComplexFloat + MulAssign<T::Real>,
        [(); N.is_power_of_two() as usize - 1]:
    {
        self.fwht_unscaled();
        self.mul_assign_all(Float::powi(T::Real::FRAC_1_SQRT_2(), TryInto::<i32>::try_into(self.len().ilog2()).unwrap() - 3))
    }
    
    fn fht(&mut self)
    where
        T: ComplexFloat<Real: Into<T> + Into<Complex<T::Real>>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign
    {
        if core::intrinsics::type_id::<T>() == core::intrinsics::type_id::<Complex<T::Real>>()
        {
            let x = unsafe {
                core::mem::transmute::<&mut Self, &mut [Complex<T::Real>; N]>(self)
            };
            x.fft();

            for x in x.iter_mut()
            {
                *x = (x.re - x.im).into()
            }

            return
        }

        let mut y = self.each_ref()
            .map(|&y| y.into());
        y.fft();
        
        for (x, y) in self.iter_mut()
            .zip(y.into_iter())
        {
            *x = (y.re - y.im).into()
        }
    }
    fn ifht(&mut self)
    where
        T: ComplexFloat<Real: Into<T> + Into<Complex<T::Real>>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + MulAssign<T::Real>
    {
        if core::intrinsics::type_id::<T>() == core::intrinsics::type_id::<Complex<T::Real>>()
        {
            let x = unsafe {
                core::mem::transmute::<&mut Self, &mut [Complex<T::Real>; N]>(self)
            };
            x.ifft();

            for x in x.iter_mut()
            {
                *x = (x.re + x.im).into()
            }

            return
        }

        let mut y = self.each_ref()
            .map(|&y| y.into());
        y.ifft();
        
        for (x, y) in self.iter_mut()
            .zip(y.into_iter())
        {
            *x = (y.re + y.im).into()
        }
    }
    
    fn dst_i(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + DivAssign<T::Real>
    {
        self.as_mut_slice()
            .dst_i()
    }
    fn dst_ii(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign
    {
        self.as_mut_slice()
            .dst_ii()
    }
    fn dst_iii(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + MulAssign<T::Real> + Mul<T::Real, Output = Complex<T::Real>> + Mul<T, Output = Complex<T::Real>>
    {
        self.as_mut_slice()
            .dst_iii()
    }
    fn dst_iv(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + Mul<T::Real, Output = Complex<T::Real>> + Mul<T, Output = Complex<T::Real>>
    {
        self.as_mut_slice()
            .dst_iv()
    }

    fn dct_i(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + DivAssign<T::Real> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + DivAssign<T::Real>
    {
        self.as_mut_slice()
            .dct_i()
    }
    fn dct_ii(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign
    {
        self.as_mut_slice()
            .dct_ii()
    }
    fn dct_iii(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + Mul<T, Output = Complex<T::Real>> + Mul<T::Real, Output = Complex<T::Real>> + DivAssign<T::Real>
    {
        self.as_mut_slice()
            .dct_iii()
    }
    fn dct_iv(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + Mul<T, Output = Complex<T::Real>> + Mul<T::Real, Output = Complex<T::Real>>
    {
        self.as_mut_slice()
            .dct_iv()
    }
    
    fn real_fft(&self, y: &mut [Complex<T>; N/2 + 1])
    where
        T: Float,
        Complex<T>: ComplexFloat<Real = T> + MulAssign + AddAssign
    {
        let mut x = self.map(|x| <Complex<_> as From<_>>::from(x));
        x.fft();

        for (x, y) in x.into_iter()
            .zip(y.iter_mut())
        {
            *y = x;
        }
    }
    
    fn real_ifft(&mut self, x: &[Complex<T>; N/2 + 1])
    where
        T: Float,
        Complex<T>: ComplexFloat<Real = T> + MulAssign + AddAssign + MulAssign<T>
    {
        let mut x = <[Complex<T>; N]>::fill(|i| if i < N/2 + 1 {x[i]} else {x[N - i].conj()});
        x.ifft();

        for (x, y) in x.into_iter()
            .zip(self.iter_mut())
        {
            *y = x.re();
        }
    }
    
    fn chebyshev_polynomial(kind: usize, order: usize) -> Option<[T; N]>
    where
        T: Copy + Add<Output = T> + Sub<Output = T> + Neg<Output = T> + AddAssign + Mul<Output = T> + One + Zero
    {
        if order > N
        {
            return None
        }
    
        let two = T::one() + T::one();
        let mut t_prev: Self = [T::zero(); _];
        t_prev[0] = T::one();
        if order == 0
        {
            return Some(t_prev)
        }
        
        let mut kind_c = T::zero();
        let mut k = 0;
        while k < kind
        {
            kind_c += T::one();
            k += 1;
        }
    
        let mut t: Self = ArrayOps::fill(|i| if i == 1 {kind_c} else {T::zero()});
    
        let mut k = 1;
        while k < order
        {
            let mut t_prev_iter = t_prev.into_iter();
            let mut t_iter = t.into_iter();
            let mut first = true;
            
            let t_next = ArrayOps::fill(|_| if first
                {
                    first = false;
                    -t_prev_iter.next().unwrap()
                }
                else
                {
                    two * t_iter.next().unwrap() - t_prev_iter.next().unwrap()
                }
            );
    
            t_prev = t;
            t = t_next;
            k += 1;
        }
    
        Some(t)
    }
    
    fn bartlett_window() -> Self
    where
        T: Float
    {
        let ld2 = T::from(N - 1).unwrap()/T::from(2.0).unwrap();
        ArrayOps::fill(|n| T::one() - (T::from(n).unwrap()/ld2 - T::one()).abs())
    }

    fn parzen_window() -> Self
    where
        T: Float
    {
        let ld2 = T::from(N).unwrap()/T::from(2.0).unwrap();
        let ld4 =ld2/T::from(2.0).unwrap();
        ArrayOps::fill(|n| {
            let m = T::from(n).unwrap() - T::from(N - 1).unwrap()/T::from(2.0).unwrap();
            let z1 = T::one() - m.abs()/ld2;
            if m.abs() <= ld4
            {
                let z2 = m/ld2;
                T::one() - T::from(6.0).unwrap()*z2*z2*z1
            }
            else
            {
                T::from(2.0).unwrap()*z1*z1*z1
            }
        })
    }
    
    fn welch_window() -> Self
    where
        T: Float
    {
        let ld2 = T::from(N - 1).unwrap()/T::from(2.0).unwrap();
        ArrayOps::fill(|n| {
            let z = T::from(n).unwrap()/ld2 - T::one();
            T::one() - z*z
        })
    }
        
    fn sine_window() -> Self
    where
        T: Float + FloatConst
    {
        ArrayOps::fill(|n| (T::PI()*T::from(n).unwrap()/T::from(N - 1).unwrap()).sin())
    }
    
    fn power_of_sine_window<A>(alpha: A) -> Self
    where
        T: Float + FloatConst + Pow<A, Output = T>,
        A: Copy
    {
        ArrayOps::fill(|n| (T::PI()*T::from(n).unwrap()/T::from(N - 1).unwrap()).sin().pow(alpha))
    }

    fn hann_window() -> Self
    where
        T: Float + FloatConst
    {
        ArrayOps::fill(|n| {
            let z = (T::PI()*T::from(n).unwrap()/T::from(N - 1).unwrap()).sin();
            z*z
        })
    }

    fn hamming_window() -> Self
    where
        T: Float + FloatConst
    {
        let a0 = T::from(25.0/46.0).unwrap();
        ArrayOps::fill(|n| {
            let z = (T::TAU()*T::from(n).unwrap()/T::from(N - 1).unwrap()).cos();
            a0 - (T::one() - a0)*z
        })
    }
    
    fn blackman_window() -> Self
    where
        T: Float + FloatConst
    {
        let a0 = T::from(7938.0/18608.0).unwrap();
        let a1 = T::from(9240.0/18608.0).unwrap();
        let a2 = T::from(1430.0/18608.0).unwrap();
        ArrayOps::fill(|n| {
            let z1 = (T::TAU()*T::from(n).unwrap()/T::from(N - 1).unwrap()).cos();
            let z2 = (T::TAU()*T::from(n*2).unwrap()/T::from(N - 1).unwrap()).cos();
            a0 - a1*z1 + a2*z2
        })
    }
    
    fn nuttal_window() -> Self
    where
        T: Float + FloatConst
    {
        let a0 = T::from(0.355768).unwrap();
        let a1 = T::from(0.487396).unwrap();
        let a2 = T::from(0.144232).unwrap();
        let a3 = T::from(0.012604).unwrap();
        ArrayOps::fill(|n| {
            let z1 = (T::TAU()*T::from(n).unwrap()/T::from(N - 1).unwrap()).cos();
            let z2 = (T::TAU()*T::from(n*2).unwrap()/T::from(N - 1).unwrap()).cos();
            let z3 = (T::TAU()*T::from(n*3).unwrap()/T::from(N - 1).unwrap()).cos();
            a0 - a1*z1 + a2*z2 - a3*z3
        })
    }
    
    fn blackman_nuttal_window() -> Self
    where
        T: Float + FloatConst
    {
        let a0 = T::from(0.3635819).unwrap();
        let a1 = T::from(0.4891775).unwrap();
        let a2 = T::from(0.1365995).unwrap();
        let a3 = T::from(0.0106411).unwrap();
        ArrayOps::fill(|n| {
            let z1 = (T::TAU()*T::from(n).unwrap()/T::from(N - 1).unwrap()).cos();
            let z2 = (T::TAU()*T::from(n*2).unwrap()/T::from(N - 1).unwrap()).cos();
            let z3 = (T::TAU()*T::from(n*3).unwrap()/T::from(N - 1).unwrap()).cos();
            a0 - a1*z1 + a2*z2 - a3*z3
        })
    }
    
    fn blackman_harris_window() -> Self
    where
        T: Float + FloatConst
    {
        let a0 = T::from(0.35875).unwrap();
        let a1 = T::from(0.48829).unwrap();
        let a2 = T::from(0.14128).unwrap();
        let a3 = T::from(0.01168).unwrap();
        ArrayOps::fill(|n| {
            let z1 = (T::TAU()*T::from(n).unwrap()/T::from(N - 1).unwrap()).cos();
            let z2 = (T::TAU()*T::from(n*2).unwrap()/T::from(N - 1).unwrap()).cos();
            let z3 = (T::TAU()*T::from(n*3).unwrap()/T::from(N - 1).unwrap()).cos();
            a0 - a1*z1 + a2*z2 - a3*z3
        })
    }
    
    fn flat_top_window() -> Self
    where
        T: Float + FloatConst
    {
        let a0 = T::from(0.21557895).unwrap();
        let a1 = T::from(0.41663158).unwrap();
        let a2 = T::from(0.277263158).unwrap();
        let a3 = T::from(0.083578947).unwrap();
        let a4 = T::from(0.006947368).unwrap();
        ArrayOps::fill(|n| {
            let z1 = (T::TAU()*T::from(n).unwrap()/T::from(N - 1).unwrap()).cos();
            let z2 = (T::TAU()*T::from(n*2).unwrap()/T::from(N - 1).unwrap()).cos();
            let z3 = (T::TAU()*T::from(n*3).unwrap()/T::from(N - 1).unwrap()).cos();
            let z4 = (T::TAU()*T::from(n*4).unwrap()/T::from(N - 1).unwrap()).cos();
            a0 - a1*z1 + a2*z2 - a3*z3 + a4*z4
        })
    }
    
    fn frac_rotate_right(&mut self, shift: T::Real)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<<T>::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + MulAssign<T::Real>
    {
        let (trunc, fract) = if let Some(trunc) = NumCast::from(shift.trunc())
        {
            (trunc, shift.fract())
        }
        else
        {
            (0, shift)
        };
        if !fract.is_zero()
        {
            let mut x: [Complex<T::Real>; N] = self.map(|x| x.into());
            x.fft();
            let n = x.len();
            for (k, x) in x.iter_mut()
                .enumerate()
            {
                let m = <T::Real as NumCast>::from((k + n/2) % n).unwrap() - NumCast::from(n/2).unwrap();
                
                let z = Complex::cis(-T::Real::TAU()*fract*m/NumCast::from(n).unwrap());
                *x *= z
            }
            x.ifft();
            for (y, x) in self.iter_mut()
                .zip(x.into_iter())
            {
                if let Some(y) = <dyn Any>::downcast_mut::<Complex<T::Real>>(y as &mut dyn Any)
                {
                    *y = x
                }
                else
                {
                    *y = x.re.into()
                }
            }
        }

        if shift.is_sign_positive()
        {
            self.rotate_right(trunc)
        }
        else
        {
            self.rotate_left(trunc)
        }
    }
    fn frac_rotate_left(&mut self, shift: T::Real)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<<T>::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + MulAssign<T::Real>
    {
        self.frac_rotate_right(-shift)
    }
}

#[cfg(test)]
mod test
{
    use num::Complex;

    use crate::ArrayMath;

    #[test]
    fn test()
    {
        let p = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let p = p.map(|b| Complex::new(b, 0.0));

        let r = p.rpolynomial_roots();

        println!("x = {:?}", r);

        for r in r
        {
            println!("p = {:?}", p.rpolynomial(r));
        }
    }
}