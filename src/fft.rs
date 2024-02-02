use std::{f64::consts::TAU, ops::{AddAssign, MulAssign}};

use array__ops::SliceOps;
use num::{complex::ComplexFloat, Complex, Float, NumCast};

pub fn fft_radix2_unscaled<T, const N: usize, const I: bool>(array: &mut [T; N]) -> bool
where
    T: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<T::Real>>
{
    if N.is_power_of_two()
    {
        array.as_mut_slice()
            .bit_reverse_permutation();
        
        for s in 0..N.ilog2()
        {
            let m = 2usize << s;
            let wm = <T as From<_>>::from(Complex::cis(<T::Real as NumCast>::from(if I {TAU} else {-TAU}/m as f64).unwrap()));
            for k in (0..N).step_by(m)
            {
                let mut w = T::one();
                for j in 0..m/2
                {
                    let t = w*array[k + j + m/2];
                    let u = array[k + j];
                    array[k + j] = u + t;
                    array[k + j + m/2] = u - t;
                    w *= wm;
                }
            }
        }
        return true
    }
    false
}

pub fn dft_unscaled<T, const N: usize, const I: bool>(array: &mut [T; N])
where
    T: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<T::Real>>
{
    let wn = <T as From<_>>::from(Complex::cis(<T::Real as NumCast>::from(if I {TAU} else {-TAU}/N as f64).unwrap()));
    let mut wnk = T::one();

    let mut buf = [T::zero(); N];
    std::mem::swap(&mut buf, array);
    for k in 0..N
    {
        let mut wnki = T::one();
        for i in 0..N
        {
            array[k] += buf[i]*wnki;
            wnki *= wnk;
        }

        wnk *= wn;
    }
}