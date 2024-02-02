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

#[test]
fn test()
{
    const P: usize = 4;
    let n: Vec<_> = (0..16).filter(|&n| is_power_of(n, P))
        .collect();
    println!("{:?}", n)
}