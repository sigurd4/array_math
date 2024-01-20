use std::ops::{AddAssign, MulAssign, Mul, Neg, Div};

use array__ops::ArrayOps;
use num::{Float, One, Zero};

#[const_trait]
pub trait ArrayMath<T, const N: usize>: ~const ArrayOps<T, N>
{
    fn sum(self) -> T
    where
        T: /*~const*/ AddAssign + Zero;

    fn product(self) -> T
    where
        T: /*~const*/ MulAssign + One;

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
        u8: /*~const*/ Into<T>,
        T: /*~const*/ Div + /*~const*/ AddAssign + Zero,
        [(); u8::MAX as usize - N]:;
    
    fn avg16(self) -> <T as Div>::Output
    where
        u16: /*~const*/ Into<T>,
        T: /*~const*/ Div + /*~const*/ AddAssign + Zero,
        [(); u16::MAX as usize - N]:;

    fn avg32(self) -> <T as Div>::Output
    where
        u32: /*~const*/ Into<T>,
        T: /*~const*/ Div + /*~const*/ AddAssign + Zero,
        [(); u32::MAX as usize - N]:;
    
    fn avg64(self) -> <T as Div>::Output
    where
        u64: /*~const*/ Into<T>,
        T: /*~const*/ Div + /*~const*/ AddAssign + Zero;

    fn mul_dot<Rhs>(self, rhs: [Rhs; N]) -> <T as Mul<Rhs>>::Output
    where
        T: /*~const*/ Mul<Rhs, Output: /*~const*/ AddAssign + Zero>;

    fn magnitude_squared(self) -> <T as Mul<T>>::Output
    where
        T: /*~const*/ Mul<T, Output: /*~const*/ AddAssign + Zero> + Copy;

    fn magnitude(self) -> <T as Mul<T>>::Output
    where
        T: /*~const*/ Mul<T, Output: /*~const*/ AddAssign + Zero + Float> + Copy;
    
    fn magnitude_inv(self) -> <T as Mul<T>>::Output
    where
        T: /*~const*/ Mul<T, Output: /*~const*/ AddAssign + Zero + Float> + Copy;

    fn normalize(self) -> [<T as Mul<<T as Mul<T>>::Output>>::Output; N]
    where
        T: /*~const*/ Mul<T, Output: /*~const*/ AddAssign + Zero + Float + Copy> + /*~const*/ Mul<<T as Mul<T>>::Output> + Copy;

    fn normalize_to<Rhs>(self, magnitude: Rhs) -> [<T as Mul<<<T as Mul<T>>::Output as Mul<Rhs>>::Output>>::Output; N]
    where
        T: /*~const*/ Mul<T, Output: /*~const*/ AddAssign + Zero + Float + /*~const*/ Mul<Rhs, Output: Copy>> + /*~const*/ Mul<<<T as Mul<T>>::Output as Mul<Rhs>>::Output> + Copy;
}

impl<T, const N: usize> /*const*/ ArrayMath<T, N> for [T; N]
{
    fn sum(self) -> T
    where
        T: /*~const*/ AddAssign + Zero
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
        T: /*~const*/ MulAssign + One
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
        u8: /*~const*/ Into<T>,
        T: /*~const*/ Div + /*~const*/ AddAssign + Zero,
        [(); u8::MAX as usize - N]:
    {
        self.sum()/(N as u8).into()
    }
    
    fn avg16(self) -> <T as Div>::Output
    where
        u16: /*~const*/ Into<T>,
        T: /*~const*/ Div + /*~const*/ AddAssign + Zero,
        [(); u16::MAX as usize - N]:
    {
        self.sum()/(N as u16).into()
    }

    fn avg32(self) -> <T as Div>::Output
    where
        u32: /*~const*/ Into<T>,
        T: /*~const*/ Div + /*~const*/ AddAssign + Zero,
        [(); u32::MAX as usize - N]:
    {
        self.sum()/(N as u32).into()
    }
    
    fn avg64(self) -> <T as Div>::Output
    where
        u64: /*~const*/ Into<T>,
        T: /*~const*/ Div + /*~const*/ AddAssign + Zero
    {
        self.sum()/(N as u64).into()
    }

    fn mul_dot<Rhs>(self, rhs: [Rhs; N]) -> <T as Mul<Rhs>>::Output
    where
        T: /*~const*/ Mul<Rhs, Output: /*~const*/ AddAssign + Zero>
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
        T: /*~const*/ Mul<T, Output: /*~const*/ AddAssign + Zero> + Copy
    {
        self.mul_dot(self)
    }
    
    fn magnitude(self) -> <T as Mul<T>>::Output
    where
        T: /*~const*/ Mul<T, Output: /*~const*/ AddAssign + Zero + /*~const*/ Float> + Copy
    {
        const N: usize = 3;
        self.magnitude_squared()
            .sqrt()
    }
    
    fn magnitude_inv(self) -> <T as Mul<T>>::Output
    where
        T: /*~const*/ Mul<T, Output: /*~const*/ AddAssign + Zero + /*~const*/ Float> + Copy
    {
        const N: usize = 4;
        self.magnitude_squared()
            .sqrt()
            .recip()
    }

    fn normalize(self) -> [<T as Mul<<T as Mul<T>>::Output>>::Output; N]
    where
        T: /*~const*/ Mul<T, Output: /*~const*/ AddAssign + Zero + /*~const*/ Float + Copy> + /*~const*/ Mul<<T as Mul<T>>::Output> + Copy
    {
        self.mul_all(self.magnitude_inv())
    }

    fn normalize_to<Rhs>(self, magnitude: Rhs) -> [<T as Mul<<<T as Mul<T>>::Output as Mul<Rhs>>::Output>>::Output; N]
    where
        T: /*~const*/ Mul<T, Output: /*~const*/ AddAssign + Zero + /*~const*/ Float + /*~const*/ Mul<Rhs, Output: Copy>> + /*~const*/ Mul<<<T as Mul<T>>::Output as Mul<Rhs>>::Output> + Copy
    {
        self.mul_all(self.magnitude_inv()*magnitude)
    }
}