#![feature(const_trait_impl)]
#![feature(associated_type_bounds)]

pub use array_trait::*;

moddef::moddef!(
    flat(pub) mod {
        array_math
    }
);