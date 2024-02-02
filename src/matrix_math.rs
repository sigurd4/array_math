use std::{iter::Sum, mem::MaybeUninit, ops::{AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign}};

use array__ops::{Array2dOps, ArrayOps, min_len, max_len};
use num::{complex::ComplexFloat, Complex, Float, One, Zero};

use crate::{ArrayMath};

#[const_trait]
pub trait MatrixMath<T, const M: usize, const N: usize>: ~const Array2dOps<T, M, N>
{
    fn identity_matrix() -> Self
    where
        T: Zero + One;

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
    /// let y_fft = x.convolve_2d_fft(&h);
    /// let y_direct = x.convolve_2d_direct(&h);
    /// 
    /// let avg_error = y_fft.comap(y_direct, |y_fft, y_direct| y_fft.comap(y_direct, |y_fft: f64, y_direct: f64| (y_fft - y_direct).abs()).avg()).avg();
    /// assert!(avg_error < 1.0e-16);
    /// ```
    fn convolve_2d_fft<Rhs, const W: usize, const H: usize>(&self, rhs: &[[Rhs; W]; H]) -> [[<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real; N + W - 1]; M + H - 1]
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
    
    fn pivot_matrix(&self) -> [[T; M]; M]
    where
        T: Neg<Output = T> + Zero + One + PartialOrd + Copy;
    
    fn lup_matrix(&self) -> ([[T; max_len(M, N)]; M], [[T; N]; max_len(M, N)], [[T; M]; M])
    where
        T: Neg<Output = T> + Zero + One + PartialOrd + Mul<Output = T> + AddAssign + Copy + Sub<Output = T> + Div<Output = T>;
            
    fn lu_matrix(&self) -> ([[T; max_len(M, N)]; M], [[T; N]; max_len(M, N)])
    where
        T: Zero + One + PartialOrd + Mul<Output = T> + Sub<Output = T> + Div<Output = T> + AddAssign + Copy;

    fn det_matrix(&self) -> T
    where
        T: Zero + Neg<Output = T> + One + Copy + AddAssign + PartialOrd + Zero + Mul<Output = T> + MulAssign + Sub<Output = T> + Div<Output = T>,
        [(); max_len(M, N)]:;
    
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
    
    fn convolve_2d_fft<Rhs, const W: usize, const H: usize>(&self, rhs: &[[Rhs; W]; H]) -> [[<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real; N + W - 1]; M + H - 1]
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
    
    fn pivot_matrix(&self) -> [[T; M]; M]
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
    
    fn lup_matrix(&self) -> ([[T; max_len(M, N)]; M], [[T; N]; max_len(M, N)], [[T; M]; M])
    where
        T: Neg<Output = T> + Zero + One + PartialOrd + Mul<Output = T> + AddAssign + Copy + Sub<Output = T> + Div<Output = T>
    {
        let mut l = [[Zero::zero(); _]; _];
        let mut u = [[Zero::zero(); _]; _];
    
        let p = self.pivot_matrix();
        let pa = p.mul_matrix(self);
    
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
                u[m][n] = pa[m][n] - s;
            }
            for i in n..M
            {
                let mut s = Zero::zero();
                for k in 0..n
                {
                    s += u[k][n]*l[i][k];
                }
                if !(pa[i][n] - s).is_zero()
                {
                    l[i][n] = (pa[i][n] - s)/u[n][n];
                }
            }
        }
    
        (l, u, p)
    }
            
    fn lu_matrix(&self) -> ([[T; max_len(M, N)]; M], [[T; N]; max_len(M, N)])
    where
        T: Zero + One + PartialOrd + Mul<Output = T> + Sub<Output = T> + Div<Output = T> + AddAssign + Copy
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
                    l[i][n] = (self[i][n] - s)/u[n][n];
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
        let (l, u, p) = self.lup_matrix();
    
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
            while n < N && p[m][n].is_zero()
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
    
    fn is_matrix_invertible(&self) -> bool
    where
        T: Zero + Neg<Output = T> + One + Copy + AddAssign + PartialOrd + MulAssign + Sub<Output = T> + Div<Output = T>,
        [(); max_len(M, N)]:
    {
        let (l, u, _) = self.lup_matrix();
        
        let mut n = 0;
        while n != N
        {
            if l[n][n].is_zero()
            {
                return false
            }
            if u[n][n].is_zero()
            {
                return false
            }
            n += 1;
        }
    
        true
    }
}