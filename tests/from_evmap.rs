// from https://github.com/jonhoo/rust-evmap/blob/440a5850ffb528281d1dafe5175ad0b67cf4348c/tests/lib.rs

extern crate click_map;

use click_map::HashMap;

#[test]
fn busybusybusy_fast() {
    busybusybusy_inner(false);
}
#[test]
fn busybusybusy_slow() {
    busybusybusy_inner(true);
}

fn busybusybusy_inner(slow: bool) {
    use std::sync::Arc;
    use std::time;
    use std::thread;

    let threads = 4;
    let mut n = 1000;
    if !slow {
        n *= 100;
    }
    let map = Arc::new(HashMap::new());

    let rs: Vec<_> = (0..threads)
        .map(|_| {
            let map = map.clone();
            thread::spawn(move || {
                // rustfmt
                for i in 0..n {
                    let i = i.into();
                    loop {
                        match map.get_then(&i, |rs| {
                            if slow {
                                thread::sleep(time::Duration::from_millis(2));
                            }
                            *rs
                        }) {
                            Some(rs) => {
                                assert_eq!(rs, i);
                                break;
                            }
                            None => {
                                thread::yield_now();
                            }
                        }
                    }
                }
            })
        })
        .collect();

    for i in 0..n {
        map.insert(i, i);
    }

    for r in rs {
        r.join().unwrap();
    }
}
