use std::{f64::consts::TAU, ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, SubAssign}};

use array__ops::{ArrayOps, SliceOps};
use num::{complex::ComplexFloat, traits::Inv, Complex, Float, NumCast, One, Zero};

use crate::fft;

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

    fn mul_dot<Rhs>(self, rhs: [Rhs; N]) -> <T as Mul<Rhs>>::Output
    where
        T: Mul<Rhs, Output: AddAssign + Zero>;

    fn magnitude_squared(self) -> <T as Mul<T>>::Output
    where
        T: Mul<T, Output: AddAssign + Zero> + Copy;

    fn magnitude(self) -> <T as Mul<T>>::Output
    where
        T: Mul<T, Output: AddAssign + Zero + Float> + Copy;
    
    fn magnitude_inv(self) -> <T as Mul<T>>::Output
    where
        T: Mul<T, Output: AddAssign + Zero + Float> + Copy;

    fn normalize(self) -> [<T as Mul<<T as Mul<T>>::Output>>::Output; N]
    where
        T: Mul<T, Output: AddAssign + Zero + Float + Copy> + Mul<<T as Mul<T>>::Output> + Copy;

    fn normalize_to<Rhs>(self, magnitude: Rhs) -> [<T as Mul<<<T as Mul<T>>::Output as Mul<Rhs>>::Output>>::Output; N]
    where
        T: Mul<T, Output: AddAssign + Zero + Float + Mul<Rhs, Output: Copy>> + Mul<<<T as Mul<T>>::Output as Mul<Rhs>>::Output> + Copy;
        
    fn normalize_assign(&mut self)
    where
        T: Mul<T, Output: AddAssign + Zero + Float + Copy> + MulAssign<<T as Mul<T>>::Output> + Copy;

    fn normalize_assign_to<Rhs>(&mut self, magnitude: Rhs)
    where
        T: Mul<T, Output: AddAssign + Zero + Float + Mul<Rhs, Output: Copy>> + MulAssign<<<T as Mul<T>>::Output as Mul<Rhs>>::Output> + Copy;

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
    /// let y_fft = x.convolve_fft(&h);
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
    /// let y_fft = x.convolve_fft(&h);
    /// let y_direct = x.convolve_direct(&h);
    /// 
    /// let avg_error = y_fft.comap(y_direct, |y_fft: f64, y_direct: f64| (y_fft - y_direct).abs()).avg();
    /// assert!(avg_error < 1.0e-15);
    /// ```
    fn convolve_fft<Rhs, const M: usize>(&self, rhs: &[Rhs; M]) -> [<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real; N + M - 1]
    where
        T: Float + Copy,
        Rhs: Float + Copy,
        Complex<T>: MulAssign + AddAssign + ComplexFloat<Real = T> + Mul<Complex<Rhs>, Output: ComplexFloat<Real: Float>>,
        Complex<Rhs>: MulAssign + AddAssign + ComplexFloat<Real = Rhs>,
        <Complex<T> as Mul<Complex<Rhs>>>::Output: ComplexFloat<Real: Float> + Into<Complex<<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>>,
        Complex<<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>: MulAssign + AddAssign + ComplexFloat<Real = <<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>,
        [(); (N + M - 1).next_power_of_two() - N]:,
        [(); (N + M - 1).next_power_of_two() - M]:,
        [(); (N + M - 1).next_power_of_two() - (N + M - 1)]:,
        [(); (N + M - 1).next_power_of_two()/2 + 1]:;

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

    /// Performs an iterative, in-place radix-2 FFT algorithm as described in https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm#Data_reordering,_bit_reversal,_and_in-place_algorithms.
    /// If `N` is not a power of two, it uses the DFT, which is a lot slower.
    /// 
    /// # Examples
    /// ```rust
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
        T: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<T::Real>>;
        
    /// Performs an iterative, in-place radix-2 IFFT algorithm as described in https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm#Data_reordering,_bit_reversal,_and_in-place_algorithms.
    /// If `N` is not a power of two, it uses the IDFT, which is a lot slower.
    /// 
    /// # Examples
    /// ```rust
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
        T: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<T::Real>>;
    
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
}

impl<T, const N: usize> /*const*/ ArrayMath<T, N> for [T; N]
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
    
    fn magnitude(self) -> <T as Mul<T>>::Output
    where
        T: Mul<T, Output: AddAssign + Zero + Float> + Copy
    {
        const N: usize = 3;
        self.magnitude_squared()
            .sqrt()
    }
    
    fn magnitude_inv(self) -> <T as Mul<T>>::Output
    where
        T: Mul<T, Output: AddAssign + Zero + Float> + Copy
    {
        const N: usize = 4;
        self.magnitude_squared()
            .sqrt()
            .recip()
    }

    fn normalize(self) -> [<T as Mul<<T as Mul<T>>::Output>>::Output; N]
    where
        T: Mul<T, Output: AddAssign + Zero + Float + Copy> + Mul<<T as Mul<T>>::Output> + Copy
    {
        self.mul_all(self.magnitude_inv())
    }

    fn normalize_to<Rhs>(self, magnitude: Rhs) -> [<T as Mul<<<T as Mul<T>>::Output as Mul<Rhs>>::Output>>::Output; N]
    where
        T: Mul<T, Output: AddAssign + Zero + Float + Mul<Rhs, Output: Copy>> + Mul<<<T as Mul<T>>::Output as Mul<Rhs>>::Output> + Copy
    {
        self.mul_all(self.magnitude_inv()*magnitude)
    }
    
    fn normalize_assign(&mut self)
    where
        T: Mul<T, Output: AddAssign + Zero + Float + Copy> + MulAssign<<T as Mul<T>>::Output> + Copy
    {
        self.mul_assign_all(self.magnitude_inv())
    }

    fn normalize_assign_to<Rhs>(&mut self, magnitude: Rhs)
    where
        T: Mul<T, Output: AddAssign + Zero + Float + Mul<Rhs, Output: Copy>> + MulAssign<<<T as Mul<T>>::Output as Mul<Rhs>>::Output> + Copy
    {
        self.mul_assign_all(self.magnitude_inv()*magnitude)
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
    
    fn convolve_fft<Rhs, const M: usize>(&self, rhs: &[Rhs; M]) -> [<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real; N + M - 1]
    where
        T: Float + Copy,
        Rhs: Float + Copy,
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
    
    fn fft(&mut self)
    where
        T: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<T::Real>>
    {
        if !fft::fft_radix2_unscaled::<_, _, false>(self) || !fft::fft_radix3_unscaled::<_, _, false>(self)
        {
            fft::dft_unscaled::<_, _, false>(self)
        }
    }
    fn ifft(&mut self)
    where
        T: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<T::Real>>
    {
        if !fft::fft_radix2_unscaled::<_, _, true>(self) || !fft::fft_radix3_unscaled::<_, _, true>(self)
        {
            fft::dft_unscaled::<_, _, true>(self)
        }

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
}

#[test]
fn test()
{
    let x = [1.0, 1.0, 0.0, 0.0, 0.0];

    let mut z = [Complex::zero(); _];
    x.real_fft(&mut z);
    
    let mut y = [0.0; _];
    y.real_ifft(&z);

    //assert_eq!(x, y);

    println!("{:?}", y)
}