#![feature(const_trait_impl)]
#![feature(associated_type_bounds)]
#![feature(const_option_ext)]

#![feature(generic_const_exprs)]
#![feature(const_closures)]

pub use array_trait::*;

use std::{ops::{Mul, AddAssign, MulAssign, Div, Neg}, process::Output};

use float_approx_math::{ApproxSqrt, ApproxInvSqrt};
use num_identities_const::{OneConst, ZeroConst};

//#[const_trait]
pub trait ArrayMath<T, const N: usize>: /*~const*/ ArrayOps<T, N>
{
    fn sum(self) -> T
    where
        T: /*~const*/ AddAssign + ZeroConst
    {
        //self.sum_from(T::ZERO)
        self.try_sum()
            .unwrap_or_else(const || T::ZERO)
    }

    fn product(self) -> T
    where
        T: /*~const*/ MulAssign + OneConst
    {
        //self.product_from(T::ONE)
        self.try_product()
            .unwrap_or_else(const || T::ONE)
    }

    fn variance(self) -> <T as Mul>::Output
    where
        Self: ArrayOps<T, N, MappedTo<T> = Self> + Copy,
        u8: Into<T>,
        T: Div<Output: Mul<Output: Neg<Output = <T as Mul>::Output>> + Copy> + Mul<Output: AddAssign> + AddAssign + ZeroConst,
        [(); u8::MAX as usize - N]:
    {
        let mu = self.avg();
        self.mul_dot_bias(self, -(mu*mu))
    }
    
    fn variance16(self) -> <T as Mul>::Output
    where
        Self: ArrayOps<T, N, MappedTo<T> = Self> + Copy,
        u16: Into<T>,
        T: Div<Output: Mul<Output: Neg<Output = <T as Mul>::Output>> + Copy> + Mul<Output: AddAssign> + AddAssign + ZeroConst,
        [(); u16::MAX as usize - N]:
    {
        let mu = self.avg16();
        self.mul_dot_bias(self, -(mu*mu))
    }
    
    fn variance32(self) -> <T as Mul>::Output
    where
        Self: ArrayOps<T, N, MappedTo<T> = Self> + Copy,
        u32: Into<T>,
        T: Div<Output: Mul<Output: Neg<Output = <T as Mul>::Output>> + Copy> + Mul<Output: AddAssign> + AddAssign + ZeroConst,
        [(); u32::MAX as usize - N]:
    {
        let mu = self.avg32();
        self.mul_dot_bias(self, -(mu*mu))
    }
    
    fn variance64(self) -> <T as Mul>::Output
    where
        Self: ArrayOps<T, N, MappedTo<T> = Self> + Copy,
        u64: Into<T>,
        T: Div<Output: Mul<Output: Neg<Output = <T as Mul>::Output>> + Copy> + Mul<Output: AddAssign> + AddAssign + ZeroConst,
    {
        let mu = self.avg64();
        self.mul_dot_bias(self, -(mu*mu))
    }
    
    fn avg(self) -> <T as Div>::Output
    where
        u8: /*~const*/ Into<T>,
        T: /*~const*/ Div + /*~const*/ AddAssign + ZeroConst,
        [(); u8::MAX as usize - N]:
    {
        self.sum()/(N as u8).into()
    }
    
    fn avg16(self) -> <T as Div>::Output
    where
        u16: /*~const*/ Into<T>,
        T: /*~const*/ Div + /*~const*/ AddAssign + ZeroConst,
        [(); u16::MAX as usize - N]:
    {
        self.sum()/(N as u16).into()
    }

    fn avg32(self) -> <T as Div>::Output
    where
        u32: /*~const*/ Into<T>,
        T: /*~const*/ Div + /*~const*/ AddAssign + ZeroConst,
        [(); u32::MAX as usize - N]:
    {
        self.sum()/(N as u32).into()
    }
    
    fn avg64(self) -> <T as Div>::Output
    where
        u64: /*~const*/ Into<T>,
        T: /*~const*/ Div + /*~const*/ AddAssign + ZeroConst
    {
        self.sum()/(N as u64).into()
    }

    fn mul_dot<Rhs>(self, rhs: Self::MappedTo<Rhs>) -> <T as Mul<Rhs>>::Output
    where
        T: /*~const*/ Mul<Rhs, Output: /*~const*/ AddAssign + ZeroConst>
    {
        self.try_mul_dot(rhs)
            .unwrap_or_else(const || ZeroConst::ZERO)
    }

    fn magnitude_squared(self) -> <T as Mul<T>>::Output
    where
        T: /*~const*/ Mul<T, Output: /*~const*/ AddAssign + ZeroConst> + Copy;

    fn magnitude(self) -> <T as Mul<T>>::Output
    where
        T: /*~const*/ Mul<T, Output: /*~const*/ AddAssign + ZeroConst + /*~const*/ ApproxSqrt> + Copy
    {
        const N: usize = 3;
        self.magnitude_squared()
            .approx_sqrt::<{N}>()
    }
    
    fn magnitude_inv(self) -> <T as Mul<T>>::Output
    where
        T: /*~const*/ Mul<T, Output: /*~const*/ AddAssign + ZeroConst + /*~const*/ ApproxInvSqrt> + Copy
    {
        const N: usize = 4;
        self.magnitude_squared()
            .approx_inv_sqrt::<{N}>()
    }

    fn normalize(self) -> Self::MappedTo<<T as Mul<<T as Mul<T>>::Output>>::Output>
    where
        T: /*~const*/ Mul<T, Output: /*~const*/ AddAssign + ZeroConst + /*~const*/ ApproxInvSqrt + Copy> + /*~const*/ Mul<<T as Mul<T>>::Output> + Copy;

    fn normalize_to<Rhs>(self, magnitude: Rhs) -> Self::MappedTo<<T as Mul<<<T as Mul<T>>::Output as Mul<Rhs>>::Output>>::Output>
    where
        T: /*~const*/ Mul<T, Output: /*~const*/ AddAssign + ZeroConst + /*~const*/ ApproxInvSqrt + /*~const*/ Mul<Rhs, Output: Copy>> + /*~const*/ Mul<<<T as Mul<T>>::Output as Mul<Rhs>>::Output> + Copy;
}

impl<T, const N: usize> /*const*/ ArrayMath<T, N> for [T; N]
{
    fn magnitude_squared(self) -> <T as Mul<T>>::Output
    where
        T: /*~const*/ Mul<T, Output: /*~const*/ AddAssign + ZeroConst> + Copy
    {
        self.mul_dot(self)
    }

    fn normalize(self) -> Self::MappedTo<<T as Mul<<T as Mul<T>>::Output>>::Output>
    where
        T: /*~const*/ Mul<T, Output: /*~const*/ AddAssign + ZeroConst + /*~const*/ ApproxInvSqrt + Copy> + /*~const*/ Mul<<T as Mul<T>>::Output> + Copy
    {
        self.mul_all(self.magnitude_inv())
    }

    fn normalize_to<Rhs>(self, magnitude: Rhs) -> Self::MappedTo<<T as Mul<<<T as Mul<T>>::Output as Mul<Rhs>>::Output>>::Output>
    where
        T: /*~const*/ Mul<T, Output: /*~const*/ AddAssign + ZeroConst + /*~const*/ ApproxInvSqrt + /*~const*/ Mul<Rhs, Output: Copy>> + /*~const*/ Mul<<<T as Mul<T>>::Output as Mul<Rhs>>::Output> + Copy
    {
        self.mul_all(self.magnitude_inv()*magnitude)
    }
}

#[cfg(test)]
mod test
{
    use super::*;

    #[test]
    fn test()
    {
        type T = u8;
        let a: [T; 3] = [1, 2, 3];
    
        let avg: T = a.avg();
    
        println!("{}", avg)
    }
}