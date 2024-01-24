use std::{f64::consts::TAU, ops::{AddAssign, MulAssign, Mul, Neg, Div}};

use array__ops::ArrayOps;
use num::{complex::ComplexFloat, traits::Inv, Complex, Float, NumCast, One, Zero};

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

    fn mul_polynomial<Rhs, const M: usize>(&self, rhs: &[Rhs; M]) -> [<T as Mul<Rhs>>::Output; N + M - 1]
    where
        T: Mul<Rhs, Output: AddAssign + Zero> + Copy,
        Rhs: Copy;

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
    /// y.fft_cooley_tukey();
    /// y.ifft_cooley_tukey();
    /// 
    /// let avg_error = x.comap(y, |x, y| (x - y).norm()).avg();
    /// assert!(avg_error < 1.0e-16);
    /// ```
    fn fft_cooley_tukey(&mut self)
    where
        T: ComplexFloat<Real: Float> + MulAssign + From<Complex<T::Real>>,
        [(); N.is_power_of_two() as usize - 1]:;
    fn ifft_cooley_tukey(&mut self)
    where
        T: ComplexFloat<Real: Float> + MulAssign + From<Complex<T::Real>>,
        [(); N.is_power_of_two() as usize - 1]:;
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
    
    fn mul_polynomial<Rhs, const M: usize>(&self, rhs: &[Rhs; M]) -> [<T as Mul<Rhs>>::Output; N + M - 1]
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
        let mut i = 0;
        while i < N
        {
            unsafe {
                let ptr = self.as_mut_ptr().add(i);
                ptr.write(ptr.read().inv());
            }
            i += 1;
        }
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
    
    fn fft_cooley_tukey(&mut self)
    where
        T: ComplexFloat<Real: Float> + MulAssign + From<Complex<T::Real>>,
        [(); N.is_power_of_two() as usize - 1]:
    {
        self.bit_reverse_permutation();
        
        for s in 0..N.ilog2()
        {
            let m = 2usize << s;
            let wm = <T as From<_>>::from(Complex::cis(<T::Real as NumCast>::from(-TAU/m as f64).unwrap()));
            for k in (0..N).step_by(m)
            {
                let mut w = T::one();
                for j in 0..m/2
                {
                    let t = w*self[k + j + m/2];
                    let u = self[k + j];
                    self[k + j] = u + t;
                    self[k + j + m/2] = u - t;
                    w *= wm;
                }
            }
        }
    }
    fn ifft_cooley_tukey(&mut self)
    where
        T: ComplexFloat<Real: Float> + MulAssign + From<Complex<T::Real>>,
        [(); N.is_power_of_two() as usize - 1]:
    {
        self.bit_reverse_permutation();
        
        for s in 0..N.ilog2()
        {
            let m = 2usize << s;
            let wm = <T as From<_>>::from(Complex::cis(<T::Real as NumCast>::from(TAU/m as f64).unwrap()));
            for k in (0..N).step_by(m)
            {
                let mut w = T::one();
                for j in 0..m/2
                {
                    let t = w*self[k + j + m/2];
                    let u = self[k + j];
                    self[k + j] = u + t;
                    self[k + j + m/2] = u - t;
                    w *= wm;
                }
            }
        }

        self.mul_assign_all(<T as From<_>>::from(<Complex<_> as From<_>>::from(<T::Real as NumCast>::from(1.0/N as f64).unwrap())));
    }
}

#[test]
fn test_conv()
{
    let x = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        .map(|x| <Complex<_> as From<_>>::from(x));
    let mut y = x;

    y.fft_cooley_tukey();
    y.ifft_cooley_tukey();

    let avg_error = x.comap(y, |x, y| (x - y).norm()).avg();
    assert!(avg_error < 1.0e-16);
}