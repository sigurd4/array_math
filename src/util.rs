#![allow(unused)]

pub const fn is_power_of(mut n: usize, p: usize) -> bool
{
    if n <= 1
    {
        return n == 1;
    }

    while n > 1
    {
        if n % p != 0
        {
            return false;
        }

        n /= p;
    }

    return true;
}

pub const fn is_prime(n: usize) -> bool
{
    let n_sqrt = 1 << ((n.ilog2() + 1) / 2);
    let mut m = 2;

    while m < n_sqrt
    {
        if n % m == 0
        {
            return false
        }
        m += 1
    }

    true
}

pub const fn closest_prime(x: usize) -> Option<usize>
{
    if x == 0
    {
        return None;
    }
    let mut n = 2;
    let mut m = 1;
    loop
    {
        if is_prime(n)
        {
            if n > x
            {
                if n - x < x - m
                {
                    return Some(n)
                }
                else
                {
                    return Some(m)
                }
            }
            m = n;
        }
        n += 1;
    }
}

#[test]
fn test()
{
    const P: usize = 4;
    let n: Vec<_> = (0..16).filter(|&n| is_power_of(n, P))
        .collect();
    println!("{:?}", n)
}