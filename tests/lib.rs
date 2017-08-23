extern crate click_map;

use click_map::HashMap;

#[test]
fn insert() {
    let map = HashMap::with_capacity(32);
    assert_eq!(map.get_then(&1, |&i| i), None);
    assert_eq!(map.get_then(&32, |&i| i), None);
    assert_eq!(map.get_then(&3, |&i| i), None);
    assert_eq!(map.get_then(&2, |&i| i), None);
    map.insert(1, 1);
    assert_eq!(map.get_then(&1, |&i| i), Some(1));
    assert_eq!(map.get_then(&32, |&i| i), None);
    assert_eq!(map.get_then(&3, |&i| i), None);
    assert_eq!(map.get_then(&2, |&i| i), None);
    map.insert(32, 100);
    assert_eq!(map.get_then(&1, |&i| i), Some(1));
    assert_eq!(map.get_then(&32, |&i| i), Some(100));;
    assert_eq!(map.get_then(&3, |&i| i), None);
    assert_eq!(map.get_then(&2, |&i| i), None);
    map.insert(3, 8);
    assert_eq!(map.get_then(&1, |&i| i), Some(1));
    assert_eq!(map.get_then(&32, |&i| i), Some(100));
    assert_eq!(map.get_then(&3, |&i| i), Some(8));
    assert_eq!(map.get_then(&2, |&i| i), None);
    map.insert(1, 20);
    assert_eq!(map.get_then(&1, |&i| i), Some(20));
    assert_eq!(map.get_then(&32, |&i| i), Some(100));
    assert_eq!(map.get_then(&3, |&i| i), Some(8));
    assert_eq!(map.get_then(&2, |&i| i), None);
}

#[test]
fn resize() {
    let map = HashMap::with_capacity(32);
    for i in 0..100 {
        map.insert(i, i);
    }
    for i in 0..100 {
        assert_eq!(map.get_then(&i, |&i| i), Some(i));
    }
    for i in 100..200 {
        assert_eq!(map.get_then(&i, |&i| i), None);
    }
}

#[test]
fn insert_if_new() {
    let map = HashMap::new();
    for i in 0..50 {
        map.insert(i*2, i*2);
    }
    for i in 0..100 {
        map.insert_if_new(i, 100 + i);
    }

    for i in 0..100 {
        if i % 2 == 0 {
            assert_eq!(map.get_then(&i, |&v| v), Some(i))
        } else {
            assert_eq!(map.get_then(&i, |&v| v), Some(100 + i))
        }
    }

    for i in 100..200 {
        assert_eq!(map.get_then(&i, |&v| v), None)
    }
}
