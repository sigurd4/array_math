use std::{iter::Sum, ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign}};

use array__ops::ArrayOps;
use num::{complex::ComplexFloat, traits::{FloatConst, Inv, Pow}, Complex, Float, NumCast, One, Zero};

use crate::{fft, MatrixMath, SquareMatrixMath};

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
        Self: ArrayOps<T, N> + Copy,
        u8: Into<T>,
        T: Div<Output: Mul<Output: Neg<Output = <T as Mul>::Output>> + Copy> + Mul<Output: AddAssign> + AddAssign + Zero,
        [(); u8::MAX as usize - N]:;
    
    fn variance16(self) -> <T as Mul>::Output
    where
        Self: ArrayOps<T, N> + Copy,
        u16: Into<T>,
        T: Div<Output: Mul<Output: Neg<Output = <T as Mul>::Output>> + Copy> + Mul<Output: AddAssign> + AddAssign + Zero,
        [(); u16::MAX as usize - N]:;
    
    fn variance32(self) -> <T as Mul>::Output
    where
        Self: ArrayOps<T, N> + Copy,
        u32: Into<T>,
        T: Div<Output: Mul<Output: Neg<Output = <T as Mul>::Output>> + Copy> + Mul<Output: AddAssign> + AddAssign + Zero,
        [(); u32::MAX as usize - N]:;
    
    fn variance64(self) -> <T as Mul>::Output
    where
        Self: ArrayOps<T, N> + Copy,
        u64: Into<T>,
        T: Div<Output: Mul<Output: Neg<Output = <T as Mul>::Output>> + Copy> + Mul<Output: AddAssign> + AddAssign + Zero;
    
    fn avg(self) -> <T as Div>::Output
    where
        u8: Into<T>,
        T: Div + AddAssign + Zero,
        [(); u8::MAX as usize - N]:;
    
    fn avg16(self) -> <T as Div>::Output
    where
        u16: Into<T>,
        T: Div + AddAssign + Zero,
        [(); u16::MAX as usize - N]:;

    fn avg32(self) -> <T as Div>::Output
    where
        u32: Into<T>,
        T: Div + AddAssign + Zero,
        [(); u32::MAX as usize - N]:;
    
    fn avg64(self) -> <T as Div>::Output
    where
        u64: Into<T>,
        T: Div + AddAssign + Zero;

    fn geometric_mean(self) -> <T as Pow<<T as Inv>::Output>>::Output
    where
        u8: Into<T>,
        T: MulAssign + One + Pow<<T as Inv>::Output> + Inv,
        [(); u8::MAX as usize - N]:;

    fn geometric_mean16(self) -> <T as Pow<<T as Inv>::Output>>::Output
    where
        u16: Into<T>,
        T: MulAssign + One + Pow<<T as Inv>::Output> + Inv,
        [(); u16::MAX as usize - N]:;
        
    fn geometric_mean32(self) -> <T as Pow<<T as Inv>::Output>>::Output
    where
        u32: Into<T>,
        T: MulAssign + One + Pow<<T as Inv>::Output> + Inv,
        [(); u32::MAX as usize - N]:;
        
    fn geometric_mean64(self) -> <T as Pow<<T as Inv>::Output>>::Output
    where
        u64: Into<T>,
        T: MulAssign + One + Pow<<T as Inv>::Output> + Inv;

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

    fn polynomial<Rhs>(self, rhs: Rhs) -> T
    where
        T: AddAssign + MulAssign<Rhs> + Zero,
        Rhs: Copy;
    fn rpolynomial<Rhs>(self, rhs: Rhs) -> T
    where
        T: AddAssign + MulAssign<Rhs> + Zero,
        Rhs: Copy;
    fn companion_matrix(&self) -> [[<T as Neg>::Output; N - 1]; N - 1]
    where
        T: Copy + Neg,
        <T as Neg>::Output: One + Zero + DivAssign<T>;
    fn rcompanion_matrix(&self) -> [[<T as Neg>::Output; N - 1]; N - 1]
    where
        T: Copy + Neg,
        <T as Neg>::Output: One + Zero + DivAssign<T>;
    fn polynomial_roots(&self) -> [T; N - 1]
    where
        T: ComplexFloat<Real: 'static> + AddAssign + SubAssign + DivAssign + DivAssign<T::Real> + Div<T::Real, Output = T> + Mul<T::Real, Output = T> + Copy + 'static,
        [(); N - 1]:;
    fn rpolynomial_roots(&self) -> [T; N - 1]
    where
        T: ComplexFloat<Real: 'static> + AddAssign + SubAssign + DivAssign + DivAssign<T::Real> + Div<T::Real, Output = T> + Mul<T::Real, Output = T> + Copy + 'static,
        [(); N - 1]:;

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
        Complex<<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>: MulAssign + AddAssign + ComplexFloat<Real = <<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>,
        [(); (N + M - 1).next_power_of_two() - N]:,
        [(); (N + M - 1).next_power_of_two() - M]:,
        [(); (N + M - 1).next_power_of_two() - (N + M - 1)]:,
        [(); (N + M - 1).next_power_of_two()/2 + 1]:;
        
    fn convolve_fft<Rhs, const M: usize>(self, rhs: [Rhs; M]) -> [<T as Mul<Rhs>>::Output; N + M - 1]
    where
        T: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<T::Real>> + Sum + Mul<Rhs>,
        Rhs: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<Rhs::Real>> + Sum,
        <T as Mul<Rhs>>::Output: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<<<T as Mul<Rhs>>::Output as ComplexFloat>::Real>> + Sum,
        [(); (N + M - 1).next_power_of_two() - N]:,
        [(); (N + M - 1).next_power_of_two() - M]:,
        [(); (N + M - 1).next_power_of_two() - (N + M - 1)]:;

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

    fn dtft(&self, omega: T::Real) -> T
    where
        T: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<T::Real>> + Sum;
        
    fn real_dtft(&self, omega: T) -> Complex<T>
    where
        T: Float,
        Complex<T>: ComplexFloat<Real = T> + MulAssign + AddAssign;

    #[doc(hidden)]
    fn fft_unscaled<const I: bool>(&mut self)
    where
        T: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<T::Real>> + Sum;

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
        T: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<T::Real>> + Sum;
        
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
        T: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<T::Real>> + Sum;
    
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
        Complex<T>: ComplexFloat<Real = T> + MulAssign + AddAssign;

    fn chebyshev_polynomial(kind: usize, order: usize) -> Option<[T; N]>
    where
        T: Copy + Add<Output = T> + Sub<Output = T> + Neg<Output = T> + AddAssign + Mul<Output = T> + One + Zero;
        
    fn bartlett_window() -> Self
    where
        T: Float;

    fn parzen_window() -> Self
    where
        T: Float;
        
    fn belch_window() -> Self
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
        Self: ArrayOps<T, N> + Copy,
        u8: Into<T>,
        T: Div<Output: Mul<Output: Neg<Output = <T as Mul>::Output>> + Copy> + Mul<Output: AddAssign> + AddAssign + Zero,
        [(); u8::MAX as usize - N]:
    {
        let mu = self.avg();
        self.mul_dot_bias(self, -(mu*mu))
    }
    
    fn variance16(self) -> <T as Mul>::Output
    where
        Self: ArrayOps<T, N> + Copy,
        u16: Into<T>,
        T: Div<Output: Mul<Output: Neg<Output = <T as Mul>::Output>> + Copy> + Mul<Output: AddAssign> + AddAssign + Zero,
        [(); u16::MAX as usize - N]:
    {
        let mu = self.avg16();
        self.mul_dot_bias(self, -(mu*mu))
    }
    
    fn variance32(self) -> <T as Mul>::Output
    where
        Self: ArrayOps<T, N> + Copy,
        u32: Into<T>,
        T: Div<Output: Mul<Output: Neg<Output = <T as Mul>::Output>> + Copy> + Mul<Output: AddAssign> + AddAssign + Zero,
        [(); u32::MAX as usize - N]:
    {
        let mu = self.avg32();
        self.mul_dot_bias(self, -(mu*mu))
    }
    
    fn variance64(self) -> <T as Mul>::Output
    where
        Self: ArrayOps<T, N> + Copy,
        u64: Into<T>,
        T: Div<Output: Mul<Output: Neg<Output = <T as Mul>::Output>> + Copy> + Mul<Output: AddAssign> + AddAssign + Zero,
    {
        let mu = self.avg64();
        self.mul_dot_bias(self, -(mu*mu))
    }
    
    fn avg(self) -> <T as Div>::Output
    where
        u8: Into<T>,
        T: Div + AddAssign + Zero,
        [(); u8::MAX as usize - N]:
    {
        self.sum()/(N as u8).into()
    }
    
    fn avg16(self) -> <T as Div>::Output
    where
        u16: Into<T>,
        T: Div + AddAssign + Zero,
        [(); u16::MAX as usize - N]:
    {
        self.sum()/(N as u16).into()
    }

    fn avg32(self) -> <T as Div>::Output
    where
        u32: Into<T>,
        T: Div + AddAssign + Zero,
        [(); u32::MAX as usize - N]:
    {
        self.sum()/(N as u32).into()
    }
    
    fn avg64(self) -> <T as Div>::Output
    where
        u64: Into<T>,
        T: Div + AddAssign + Zero
    {
        self.sum()/(N as u64).into()
    }
    
    fn geometric_mean(self) -> <T as Pow<<T as Inv>::Output>>::Output
    where
        u8: Into<T>,
        T: MulAssign + One + Pow<<T as Inv>::Output> + Inv,
        [(); u8::MAX as usize - N]:
    {
        self.product().pow((N as u8).into().inv())
    }

    fn geometric_mean16(self) -> <T as Pow<<T as Inv>::Output>>::Output
    where
        u16: Into<T>,
        T: MulAssign + One + Pow<<T as Inv>::Output> + Inv,
        [(); u16::MAX as usize - N]:
    {
        self.product().pow((N as u16).into().inv())
    }
        
    fn geometric_mean32(self) -> <T as Pow<<T as Inv>::Output>>::Output
    where
        u32: Into<T>,
        T: MulAssign + One + Pow<<T as Inv>::Output> + Inv,
        [(); u32::MAX as usize - N]:
    {
        self.product().pow((N as u32).into().inv())
    }
        
    fn geometric_mean64(self) -> <T as Pow<<T as Inv>::Output>>::Output
    where
        u64: Into<T>,
        T: MulAssign + One + Pow<<T as Inv>::Output> + Inv
    {
        self.product().pow((N as u64).into().inv())
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
    fn companion_matrix(&self) -> [[<T as Neg>::Output; N - 1]; N - 1]
    where
        T: Copy + Neg,
        <T as Neg>::Output: One + Zero + DivAssign<T>
    {
        let mut c = <[[<T as Neg>::Output; N - 1]; N - 1]>::eye_matrix(-1);
        let mut i = 0;
        while i < N - 1
        {
            c[i][N - 2] = -self[i];
            c[i][N - 2] /= self[N - 1];
            i += 1;
        }
        c
    }
    fn rcompanion_matrix(&self) -> [[<T as Neg>::Output; N - 1]; N - 1]
    where
        T: Copy + Neg,
        <T as Neg>::Output: One + Zero + DivAssign<T>
    {
        let mut c = <[[<T as Neg>::Output; N - 1]; N - 1]>::eye_matrix(-1);
        let mut i = N - 1;
        while i > 0
        {
            c[N - 1 - i][N - 2] = -self[i];
            c[N - 1 - i][N - 2] /= self[0];
            i -= 1;
        }
        c
    }
    fn polynomial_roots(&self) -> [T; N - 1]
    where
        T: ComplexFloat<Real: 'static> + AddAssign + SubAssign + DivAssign + DivAssign<T::Real> + Div<T::Real, Output = T> + Mul<T::Real, Output = T> + Copy + 'static,
        [(); N - 1]:
    {
        let c = self.companion_matrix();
        c.eigenvalues()
    }
    fn rpolynomial_roots(&self) -> [T; N - 1]
    where
        T: ComplexFloat<Real: 'static> + AddAssign + SubAssign + DivAssign + DivAssign<T::Real> + Div<T::Real, Output = T> + Mul<T::Real, Output = T> + Copy + 'static,
        [(); N - 1]:
    {
        let c = self.rcompanion_matrix();
        c.eigenvalues()
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
        Complex<<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>: MulAssign + AddAssign + ComplexFloat<Real = <<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>,
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
        T: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<T::Real>> + Sum + Mul<Rhs>,
        Rhs: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<Rhs::Real>> + Sum,
        <T as Mul<Rhs>>::Output: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<<<T as Mul<Rhs>>::Output as ComplexFloat>::Real>> + Sum,
        [(); (N + M - 1).next_power_of_two() - N]:,
        [(); (N + M - 1).next_power_of_two() - M]:,
        [(); (N + M - 1).next_power_of_two() - (N + M - 1)]:
    {
        let mut x: [T; (N + M - 1).next_power_of_two()] = self.resize(|_| T::zero());
        let mut h: [Rhs; (N + M - 1).next_power_of_two()] = rhs.resize(|_| Rhs::zero());
        x.fft();
        h.fft();

        let mut y = x.comap(h, |x, h| (x*h).into());
        y.ifft();

        y.truncate()
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
    
    fn dtft(&self, omega: T::Real) -> T
    where
        T: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<T::Real>> + Sum
    {
        let mut y = T::zero();
        let z1 = <T as From<_>>::from(Complex::cis(-omega));
        let mut z = T::one();
        for &x in self
        {
            y += x*z;
            z *= z1;
        }
        y
    }
        
    fn real_dtft(&self, omega: T) -> Complex<T>
    where
        T: Float,
        Complex<T>: ComplexFloat<Real = T> + MulAssign + AddAssign
    {
        let mut y = Complex::zero();
        let z1 = Complex::cis(-omega);
        let mut z = Complex::one();
        for &x in self
        {
            y += <Complex<_> as From<_>>::from(x)*z;
            z *= z1;
        }
        y
    }
    
    fn fft_unscaled<const I: bool>(&mut self)
    where
        T: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<T::Real>> + Sum
    {
        if N <= 1
        {
            return;
        }
        if !(
            fft::fft_radix2_unscaled::<_, _, I>(self)
            || fft::fft_radix3_unscaled::<_, _, I>(self)
            || fft::fft_radix_p_unscaled::<_, _, 5, I>(self)
            || fft::fft_radix_p_unscaled::<_, _, 7, I>(self)
            || fft::fft_radix_p_unscaled::<_, _, 11, I>(self)
            || fft::fft_radix_p_unscaled::<_, _, 13, I>(self)
            || fft::fft_radix_p_unscaled::<_, _, 17, I>(self)
            || fft::fft_radix_p_unscaled::<_, _, 19, I>(self)
            || fft::fft_radix_p_unscaled::<_, _, 23, I>(self)
            || fft::fft_radix_p_unscaled::<_, _, 29, I>(self)
            || fft::fft_radix_p_unscaled::<_, _, 31, I>(self)
            || fft::fft_radix_p_unscaled::<_, _, 37, I>(self)
            || fft::fft_radix_p_unscaled::<_, _, 41, I>(self)
            || fft::fft_radix_p_unscaled::<_, _, 43, I>(self)
            || fft::fft_radix_p_unscaled::<_, _, 47, I>(self)
            || fft::fft_radix_p_unscaled::<_, _, 53, I>(self)
            || fft::fft_radix_p_unscaled::<_, _, 59, I>(self)
            || fft::fft_radix_p_unscaled::<_, _, 61, I>(self)
            || fft::fft_radix_p_unscaled::<_, _, 67, I>(self)
            || fft::fft_radix_p_unscaled::<_, _, 71, I>(self)
            || fft::fft_radix_p_unscaled::<_, _, 73, I>(self)
            || fft::fft_radix_p_unscaled::<_, _, 79, I>(self)
            || fft::fft_radix_p_unscaled::<_, _, 83, I>(self)
            || fft::fft_radix_p_unscaled::<_, _, 89, I>(self)
            || fft::fft_radix_p_unscaled::<_, _, 97, I>(self)
            || fft::fft_radix_n_sqrt_unscaled::<_, _, I>(self)
        )
        {
            fft::dft_unscaled::<_, _, I>(self)
        }
    }
    
    fn fft(&mut self)
    where
        T: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<T::Real>> + Sum
    {
        self.fft_unscaled::<false>()
    }
    fn ifft(&mut self)
    where
        T: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<T::Real>> + Sum
    {
        self.fft_unscaled::<true>();

        self.mul_assign_all(<T as From<_>>::from(<Complex<_> as From<_>>::from(<T::Real as NumCast>::from(1.0/N as f64).unwrap())));
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
        Complex<T>: ComplexFloat<Real = T> + MulAssign + AddAssign
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
    
    fn belch_window() -> Self
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
            let z3 = (T::TAU()*T::from(n*6).unwrap()/T::from(N - 1).unwrap()).cos();
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
            let z3 = (T::TAU()*T::from(n*6).unwrap()/T::from(N - 1).unwrap()).cos();
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
            let z3 = (T::TAU()*T::from(n*6).unwrap()/T::from(N - 1).unwrap()).cos();
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
            let z3 = (T::TAU()*T::from(n*6).unwrap()/T::from(N - 1).unwrap()).cos();
            let z4 = (T::TAU()*T::from(n*8).unwrap()/T::from(N - 1).unwrap()).cos();
            a0 - a1*z1 + a2*z2 - a3*z3 + a4*z4
        })
    }
}

#[test]
fn test()
{
    let x = [1.0f64, 2.0, 1.0];

    let x = x.map(|x| Complex::new(x, 0.0));
    
    let a = x.rpolynomial_roots();

    let [a1, a2] = a;

    println!("{:?}", (a1 + a2)/2.0)
}