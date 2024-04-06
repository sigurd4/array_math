#![feature(const_trait_impl)]
#![feature(associated_type_bounds)]
#![feature(const_option_ext)]
//#![feature(effects)]
#![feature(const_option)]
#![feature(const_refs_to_cell)]
#![feature(const_mut_refs)]
#![feature(generic_arg_infer)]
#![feature(array_methods)]
#![feature(inline_const)]
#![feature(let_chains)]
#![feature(more_float_constants)]

#![feature(core_intrinsics)]

#![feature(generic_const_exprs)]
#![feature(const_closures)]

pub use array__ops::*;
pub use slice_math::SliceMath;

moddef::moddef!(
    flat(pub) mod {
        array_math_,
        matrix_math,
        square_matrix_math
    },
    mod {
        plot for cfg(test),
        util
    }
);

#[cfg(test)]
mod test
{
    use std::time::{Duration, SystemTime};

    use num::Complex;
    use rustfft::FftPlanner;

    use super::*;

    #[test]
    fn wht()
    {
        let mut a = [19, -1, 11, -9, -7, 13, -15, 5].map(|a| a as f32);
        a.fwht();
        println!("{:?}", a)
    }
    
    #[test]
    fn wht_2d()
    {
        let mut a = [
            [3, 6, -1, -4],
            [-12, 15, 1, 7],
            [0, 5, -2, 3],
            [1, 12, -13, 4]
        ].map_nd(|a: i8| a as f32);
        a.fwht_2d();
        a.ifwht_2d();
        println!("{:?}", a)
    }

    #[test]
    fn test()
    {
        let a = [1.0, 2.0, 1.0];
            //.map(|a| Complex::new(a, 0.0));
    
        let b = a.convolve_fft([1.0, 1.0]);
    
        println!("{:?}", b)
    }

    #[test]
    fn test_matrix_cov()
    {
        let a = [
            [1.0, 2.0],
            [4.0, 4.0]
        ];

        let c = a.covariance_matrix(None);

        println!("{:?}", c)
    }

    #[test]
    fn test_polyfit()
    {
        let p = [2.0, -4.0, 1.0];

        let x = [4.0, -1.0, 6.0, 7.0];
        let y = x.map(|x| p.rpolynomial(x));

        let p: [_; 3] = x.rpolyfit(&y);

        println!("{:?}", p);
    }
    
    #[test]
    fn test_matrix_solve()
    {
        let a = [
            [1.0, 3.0],
            [2.0, 8.0]
        ];
        let b = [
            10.0,
            -12.0
        ];

        let x = a.solve_square_matrix(&b);
        println!("x = {:?}", x);

        let y = a.mul_matrix(x.as_collumn());
        println!("y = {:?}", y);
        
        let x = a.solve_matrix(&b);
        println!("x = {:?}", x);

        let y = a.mul_matrix(x.as_collumn());
        println!("y = {:?}", y);
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

    const PLOT_TARGET: &str = "plots";

    #[allow(unused)]
    pub fn benchmark<T, R>(x: &[T], f: &dyn Fn(T) -> R) -> Duration
    where
        T: Clone
    {
        let x = x.to_vec();
        let t0 = SystemTime::now();
        x.into_iter().for_each(|x| {f(x);});
        t0.elapsed().unwrap()
    }

    #[test]
    #[ignore]
    fn bench_fft()
    {
        let fn_name = "FFT";

        let plot_title: &str = &format!("{fn_name} benchmark");
        let plot_path: &str = &format!("{PLOT_TARGET}/{fn_name}_benchmark.png");

        const N: usize = 9;
        const I: [usize; N] = [
            2,
            4,
            16,
            32,
            64,
            128,
            256,
            512,
            1024
        ];

        fn f1<const N: usize>(array: &mut [Complex<f32>; N])
        {
            array.fft();
            array.ifft();
        }
        
        fn f2<const N: usize>(array: &mut [Complex<f32>; N])
        {
            array.as_mut_slice()
                .fft();
            array.as_mut_slice()
                .ifft();
        }

        fn f3<const N: usize>(array: &mut [Complex<f32>; N])
        {
            let fft = FftPlanner::new()
                .plan_fft_forward(array.len());
            fft.process(array);
            let ifft = FftPlanner::new()
                .plan_fft_inverse(array.len());
            ifft.process(array);
        }

        fn t<const N: usize>(f: impl Fn(&mut [Complex<f32>; N])) -> f32
        {
            const M: usize = 4096;

            let mut x = [Complex::from(1.0); N];
            let t0 = SystemTime::now();
            for _ in 0..M
            {
                f(&mut x);
            }
            let dt = SystemTime::now().duration_since(t0).unwrap();
            println!("Done N = {}", N);
            dt.as_secs_f32()/M as f32
        }

        let t = [
            [
                t::<{I[0]}>(f1),
                t::<{I[1]}>(f1),
                t::<{I[2]}>(f1),
                t::<{I[3]}>(f1),
                t::<{I[4]}>(f1),
                t::<{I[5]}>(f1),
                t::<{I[6]}>(f1),
                t::<{I[7]}>(f1),
                t::<{I[8]}>(f1),
            ],
            /*[
                t::<{I[0]}>(f2),
                t::<{I[1]}>(f2),
                t::<{I[2]}>(f2),
                t::<{I[3]}>(f2),
                t::<{I[4]}>(f2),
                t::<{I[5]}>(f2),
                t::<{I[6]}>(f2),
                t::<{I[7]}>(f2),
                t::<{I[8]}>(f2),
            ],*/
            [
                t::<{I[0]}>(f3),
                t::<{I[1]}>(f3),
                t::<{I[2]}>(f3),
                t::<{I[3]}>(f3),
                t::<{I[4]}>(f3),
                t::<{I[5]}>(f3),
                t::<{I[6]}>(f3),
                t::<{I[7]}>(f3),
                t::<{I[8]}>(f3),
            ]
        ];
        
        let n = I.map(|n| n as f32);

        plot::plot_curves(plot_title, plot_path, [&n; _], t.each_ref()).expect("Plot error")
    }
}