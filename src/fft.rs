use std::{f64::consts::TAU, ops::{AddAssign, MulAssign}};

use array__ops::SliceOps;
use num::{complex::ComplexFloat, Complex, Float, NumCast, Zero};
use slice_math::SliceMath;

use crate::util;

pub fn fft_radix2_unscaled<T, const N: usize, const I: bool>(array: &mut [T; N]) -> bool
where
    T: ComplexFloat<Real: Float> + MulAssign + From<Complex<T::Real>>
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

pub fn fft_radix3_unscaled<T, const N: usize, const I: bool>(array: &mut [T; N]) -> bool
where
    T: ComplexFloat<Real: Float> + MulAssign + From<Complex<T::Real>>
{
    const P: usize = 3;

    if util::is_power_of(N, P)
    {
        array.as_mut_slice()
            .bit_reverse_permutation();
        
        let w3 = <T as From<_>>::from(Complex::cis(<T::Real as NumCast>::from(if I {TAU} else {-TAU}/P as f64).unwrap()));
        let w3_p2 = w3*w3;
        let mut m = P;
        for _ in 0..N.ilog(P)
        {
            let wm = <T as From<_>>::from(Complex::cis(<T::Real as NumCast>::from(if I {TAU} else {-TAU}/m as f64).unwrap()));
            let wm_d3 = <T as From<_>>::from(Complex::cis(<T::Real as NumCast>::from(if I {TAU} else {-TAU}/m as f64*P as f64).unwrap()));
            let mut wm_pn = T::one();
            for n in (0..N).step_by(m)
            {
                let mut wm_d3_pnk = T::one();
                for k in 0..m/P
                {
                    let p = array[n + k] + array[n + k + m/P] + array[n + k + m/P*2];
                    let q = wm_pn*(array[n + k] + array[n + k + m/P]*w3 + array[n + k + m/P*2]*w3_p2);
                    let r = wm_pn*wm_pn*(array[n + k] + array[n + k + m/P]*w3_p2 + array[n + k + m/P*2]*w3);

                    array[n + k] = wm_d3_pnk*p;
                    array[n + k + m/P] = wm_d3_pnk*q;
                    array[n + k + m/P*2] = wm_d3_pnk*r;
                    wm_d3_pnk *= wm_d3;
                }
                wm_pn *= wm;
            }
            m *= P;
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

#[test]
fn test()
{
    use crate::ArrayMath;

    let mut x = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let mut y = [Complex::zero(); 5];

    x.real_fft(&mut y);
    x.real_ifft(&y);

    println!("{:?}", x)
}