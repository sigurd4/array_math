use std::{f64::consts::TAU, iter::Sum, ops::{AddAssign, MulAssign}};

use array__ops::SliceOps;
use num::{complex::ComplexFloat, Complex, Float, NumCast};
use slice_math::SliceMath;

use crate::{util, ArrayMath};

pub fn partial_fft_unscaled<T, const N: usize, const I: bool, const M: usize>(array: &mut [T; N]) -> [Vec<T>; M]
where
    T: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<T::Real>> + Sum,
    [(); M - 1]:
{
    let spread = array.as_slice().spread_ref();

    spread.map(|spread| {
        let mut spread: Vec<_> = spread.into_iter()
            .map(|x| **x)
            .collect();
        spread.fft_unscaled::<I>();
        spread
    })
}

pub fn partial_fft_unscaled_vec<T, const N: usize, const I: bool>(array: &mut [T; N], m: usize) -> Vec<Vec<T>>
where
    T: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<T::Real>> + Sum
{
    (0..m).map(|k| {
        let mut spread: Vec<_> = array[k..].into_iter()
            .step_by(m)
            .map(|&x| x)
            .collect();
        spread.fft_unscaled::<I>();
        spread
    }).collect()
}

pub fn fft_radix2_unscaled<T, const N: usize, const I: bool>(array: &mut [T; N]) -> bool
where
    T: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<T::Real>> + Sum
{
    if N.is_power_of_two()
    {
        // In-place FFT

        array.as_mut_slice()
            .bit_rev_permutation();
        
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
    if N % 2 == 0 
    {
        // Recursive FFT

        let [even, odd] = partial_fft_unscaled::<_, _, I, _>(array);

        let wn = <T as From<_>>::from(Complex::cis(<T::Real as NumCast>::from(if I {TAU} else {-TAU}/N as f64).unwrap()));
        let mut wn_pk = T::one();
        for k in 0..N/2
        {
            let p = even[k];
            let q = wn_pk*odd[k];

            array[k] = p + q;
            array[k + N/2] = p - q;

            wn_pk *= wn;
        }
        return true;
    }
    false
}

pub fn fft_radix3_unscaled<T, const N: usize, const I: bool>(array: &mut [T; N]) -> bool
where
    T: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<T::Real>> + Sum
{
    const P: usize = 3;

    if N % P == 0
    {
        // Recursive FFT

        let [x1, x2, x3] = partial_fft_unscaled::<_, _, I, _>(array);

        let w3 = <T as From<_>>::from(Complex::cis(<T::Real as NumCast>::from(if I {TAU} else {-TAU}/P as f64).unwrap()));
        let w3_p2 = w3*w3;
        let wn = <T as From<_>>::from(Complex::cis(<T::Real as NumCast>::from(if I {TAU} else {-TAU}/N as f64).unwrap()));
        let mut wn_pn = T::one();
        for k in 0..N/P
        {
            let p = x1[k] + x2[k] + x3[k];
            let q = wn_pn*(x1[k] + x2[k]*w3 + x3[k]*w3_p2);
            let r = wn_pn*wn_pn*(x1[k] + x2[k]*w3_p2 + x3[k]*w3);

            array[k] = p;
            array[k + N/P] = q;
            array[k + N/P*2] = r;
            wn_pn *= wn;
        }
        return true;
    }
    false
}

pub fn fft_radix_p_unscaled<T, const N: usize, const P: usize, const I: bool>(array: &mut [T; N]) -> bool
where
    T: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<T::Real>> + Sum,
    [(); P - 1]:
{
    if N % P == 0
    {
        // Recursive FFT

        let x: [_; P] = partial_fft_unscaled::<_, _, I, _>(array);

        let wn = <T as From<_>>::from(Complex::cis(<T::Real as NumCast>::from(if I {TAU} else {-TAU}/N as f64).unwrap()));
        let mut wn_pk = T::one();
        let m = N/P;
        for k in 0..N
        {
            let mut e = T::one();
            array[k] = x.iter()
                .map(|x| {
                    let x = x[k % m];
                    let y = x*e;
                    e *= wn_pk;
                    y
                }).sum::<T>();
            wn_pk *= wn;
        }
        return true;
    }
    false
}

pub fn fft_radix_n_sqrt_unscaled<T, const N: usize, const I: bool>(array: &mut [T; N]) -> bool
where
    T: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<T::Real>> + Sum
{
    let p = const {
        util::closest_prime(1 << ((N.ilog2() + 1) / 2))
    };
    if let Some(p) = p && N % p == 0
    {
        // Recursive FFT

        let x = partial_fft_unscaled_vec::<_, _, I>(array, p);

        let wn = <T as From<_>>::from(Complex::cis(<T::Real as NumCast>::from(if I {TAU} else {-TAU}/N as f64).unwrap()));
        let mut wn_pk = T::one();
        let m = N/p;
        for k in 0..N
        {
            let mut e = T::one();
            array[k] = x.iter()
                .map(|x| {
                    let x = x[k % m];
                    let y = x*e;
                    e *= wn_pk;
                    y
                }).sum::<T>();
            wn_pk *= wn;
        }
        return true;
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
    use num::Zero;

    let mut x = [0.0; 256];
    x[0] = 1.0;
    let mut y = [Complex::zero(); 129];

    x.real_fft(&mut y);
    x.real_ifft(&y);

    println!("{:?}", x)
}