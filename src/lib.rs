#![feature(const_trait_impl)]
#![feature(associated_type_bounds)]
#![feature(const_option_ext)]
//#![feature(effects)]
#![feature(const_option)]
#![feature(const_refs_to_cell)]
#![feature(const_mut_refs)]
#![feature(generic_arg_infer)]

#![feature(generic_const_exprs)]
#![feature(const_closures)]

pub use array__ops::*;

moddef::moddef!(
    flat(pub) mod {
        array_math_,
        matrix_math,
        square_matrix_math
    }
);

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
    
    #[test]
    fn test_matrix_solve()
    {
        let a = [
            [1.0, 3.0],
            [2.0, 8.0]
        ];
        let b = [
            6.0,
            -12.0
        ];

        let x = a.solve_matrix(&b);
        println!("x = {:?}", x);
    }

    #[test]
    fn test_lu()
    {
        use array__ops::ArrayNdOps;

        let a = [
            [0, 3, -1, 2],
            [3, 8, 1, -4],
            [-1, 1, 4, -1],
            [-1, 1, 4, -2],
        ].map_nd(|a: i8| a as f32);

        let (l, u, p) = a.lup_matrix();
        
        println!("P = {:?}", p);
        println!("L = {:?}", l);
        println!("U = {:?}", u);

        let lu = l.mul_matrix(&u);
        let pa = p.mul_matrix(&a);

        println!("LU = {:?}", lu);
        println!("PA = {:?}", pa);

        let det_a = a.det_matrix();
        
        println!("det(A) = {:?}", det_a);

        if let Some(inv_a) = a.inv_matrix()
        {
            println!("A^(-1) = {:?}", inv_a);
            
            println!("A^(-1)*A = {:?}", inv_a.mul_matrix(&a));
            println!("A*A^(-1) = {:?}", a.mul_matrix(&inv_a));
        }
    }
}