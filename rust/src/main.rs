mod order_stat;

fn main() {
    let previous = [1, 2, 3, 4];
    let next = order_stat::next_combination(&previous, 2);

    println!("previous = {:?}", previous);
    println!("next = {:?}", next);
}
