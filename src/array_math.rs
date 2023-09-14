use std::ops::{Mul, AddAssign};

use array_trait::ArrayOps;

use float_approx_math::{ApproxSqrt, ApproxInvSqrt};

#[const_trait]
pub trait ArrayMath<T, const N: usize>: ArrayOps<T, N>
{
    fn magnitude(self) -> <T as Mul<T>>::Output
    where
        T: ~const Mul<T, Output: ~const AddAssign + ~const Default + ~const ApproxSqrt> + Copy;

    fn magnitude_inv(self) -> <T as Mul<T>>::Output
    where
        T: ~const Mul<T, Output: ~const AddAssign + ~const Default + ~const ApproxInvSqrt> + Copy;

    fn normalize(self) -> Self::MappedTo<<T as Mul<<T as Mul<T>>::Output>>::Output>
    where
        T: ~const Mul<T, Output: ~const AddAssign + ~const Default + ~const ApproxInvSqrt + Copy> + ~const Mul<<T as Mul<T>>::Output> + Copy;
        
    fn normalize_to<Rhs>(self, magnitude: Rhs) -> Self::MappedTo<<T as Mul<<<T as Mul<T>>::Output as Mul<Rhs>>::Output>>::Output>
    where
        T: ~const Mul<T, Output: ~const AddAssign + ~const Default + ~const ApproxInvSqrt + ~const Mul<Rhs, Output: Copy>> + ~const Mul<<<T as Mul<T>>::Output as Mul<Rhs>>::Output> + Copy;
}

impl<T, const N: usize> const ArrayMath<T, N> for [T; N]
{
    fn magnitude(self) -> <T as Mul<T>>::Output
    where
        T: ~const Mul<T, Output: ~const AddAssign + ~const Default + ~const ApproxSqrt> + Copy
    {
        const N: usize = 3;
        self.magnitude_squared().approx_sqrt::<{N}>()
    }
    
    fn magnitude_inv(self) -> <T as Mul<T>>::Output
    where
        T: ~const Mul<T, Output: ~const AddAssign + ~const Default + ~const ApproxInvSqrt> + Copy
    {
        const N: usize = 4;
        self.magnitude_squared().approx_inv_sqrt::<{N}>()
    }

    fn normalize(self) -> Self::MappedTo<<T as Mul<<T as Mul<T>>::Output>>::Output>
    where
        T: ~const Mul<T, Output: ~const AddAssign + ~const Default + ~const ApproxInvSqrt + Copy> + ~const Mul<<T as Mul<T>>::Output> + Copy
    {
        self.mul_all(self.magnitude_inv())
    }

    fn normalize_to<Rhs>(self, magnitude: Rhs) -> Self::MappedTo<<T as Mul<<<T as Mul<T>>::Output as Mul<Rhs>>::Output>>::Output>
    where
        T: ~const Mul<T, Output: ~const AddAssign + ~const Default + ~const ApproxInvSqrt + ~const Mul<Rhs, Output: Copy>> + ~const Mul<<<T as Mul<T>>::Output as Mul<Rhs>>::Output> + Copy
    {
        self.mul_all(self.magnitude_inv()*magnitude)
    }
}