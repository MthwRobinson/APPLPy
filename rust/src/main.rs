mod algorithms;

use algorithms::order_stat;

fn main() {
    let previous = vec![0, 1, 2];
    let next = order_stat::next_combination(&previous, 4).expect("No more combinations");
    let another_next = order_stat::next_combination(&previous, 4).expect("No more combinations");

    println!("previous = {:?}", previous);
    println!("next = {:?}", next);
    println!("next = {:?}", another_next);
}
