mod algorithms;

use algorithms::order_stat;

fn main() {
    let previous = [1, 2, 3, 4];
    let next = order_stat::next_combination(&previous, 2).expect("No more combinations");
    let another_next = order_stat::next_combination(&previous, 3).expect("No more combinations");

    println!("previous = {:?}", previous);
    println!("next = {:?}", next);
    println!("next = {:?}", another_next);
}
