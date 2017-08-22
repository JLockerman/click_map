// modified from std hash/map.rs tests
// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate click_map;

use click_map::HashMap;

#[test]
fn test_insert() {
    let m = HashMap::new();
    m.insert(1, 2);
    m.insert(2, 4);
    m.get_and(&1,|v| assert_eq!(v, &2)).unwrap();
    m.get_and(&2,|v| assert_eq!(v, &4)).unwrap();
}

#[test]
fn test_clone() {
    let m = HashMap::new();
    m.insert(1, 2);
    m.insert(2, 4);
    let m2 = m.clone();
    m2.get_and(&1,|v| assert_eq!(v, &2)).unwrap();
    m2.get_and(&2,|v| assert_eq!(v, &4)).unwrap();
}

#[test]
fn test_lots_of_insertions() {
    let m = HashMap::new();

    // Try this a few times to make sure we never screw up the hashmap's
    // internal state.
    for _ in 0..10 {

        for i in 1..1001 {
            // assert!(m.insert(i, i).is_none());
            m.insert(i, i);

            for j in 1..i + 1 {
                // let r = m.get(&j);
                // assert_eq!(r, Some(&j));
                assert_eq!(m.get_and(&j,|&v| v), Some(j));
            }

            for j in i + 1..1001 {
                // let r = m.get(&j);
                // assert_eq!(r, None);
                assert_eq!(m.get_and(&j,|&v| v), None);
            }
        }

        for i in 1001..2001 {
            // assert!(!m.contains_key(&i));
            assert_eq!(m.get_and(&i, |&v| v), None);
        }

        // remove forwards
        for i in 1..1001 {
            // assert!(m.remove(&i).is_some());
            m.remove(&i);

            for j in 1..i + 1 {
                // assert!(!m.contains_key(&j));
                assert_eq!(m.get_and(&j,|&v| v), None);
            }

            for j in i + 1..1001 {
                // assert!(m.contains_key(&j));
                assert_eq!(m.get_and(&j,|&v| v), Some(j));
            }
        }

        for i in 1..1001 {
            assert_eq!(m.get_and(&i,|&v| v), None);
        }

        for i in 1..1001 {
            // assert!(m.insert(i, i).is_none());
            m.insert(i, i);
        }

        // remove backwards
        for i in (1..1001).rev() {
            // assert!(m.remove(&i).is_some());
            m.remove(&i);

            for j in i..1001 {
                // assert!(!m.contains_key(&j));
                assert_eq!(m.get_and(&j,|&v| v), None);
            }

            for j in 1..i {
                // assert!(m.contains_key(&j));
                assert_eq!(m.get_and(&j,|&v| v), Some(j));
            }
        }
    }
}


mod drop_tests {
    use click_map::{coco, HashMap};

    use std::cell::RefCell;

    thread_local! { static DROP_VECTOR: RefCell<Vec<isize>> = RefCell::new(Vec::new()) }

    #[derive(Hash, PartialEq, Eq, Debug)]
    struct Dropable {
        k: usize,
    }

    impl Dropable {
        fn new(k: usize) -> Dropable {
            DROP_VECTOR.with(|slot| {
                slot.borrow_mut()[k] += 1;
            });

            Dropable { k: k }
        }
    }

    impl Drop for Dropable {
        fn drop(&mut self) {
            DROP_VECTOR.with(|slot| {
                slot.borrow_mut()[self.k] -= 1;
            });
        }
    }

    impl Clone for Dropable {
        fn clone(&self) -> Dropable {
            Dropable::new(self.k)
        }
    }

    #[test]
    fn test_drops() {
        DROP_VECTOR.with(|slot| {
            *slot.borrow_mut() = vec![0; 200];
        });

        {
            let m = HashMap::new();

            DROP_VECTOR.with(|v| {
                for i in 0..200 {
                    assert_eq!(v.borrow()[i], 0);
                }
            });

            for i in 0..100 {
                let d1 = Dropable::new(i);
                let d2 = Dropable::new(i + 100);
                m.insert(d1, d2);
            }
            coco::epoch::pin(|s| s.flush());

            DROP_VECTOR.with(|v| {
                for i in 0..200 {
                    assert_eq!(v.borrow()[i], 1);
                }
            });

            coco::epoch::pin(|s| s.flush());

            for i in 0..50 {
                let k = Dropable::new(i);
                let v = m.remove_and(&k, |_| {
                    DROP_VECTOR.with(|v| {
                        // there are 2 copies of the key, one in the map matching the tombstone
                        // the other we used for the search
                        assert_eq!(v.borrow()[i], 2);
                        assert_eq!(v.borrow()[i+100], 1);
                    });
                });
                assert!(v.is_some());
                assert!(m.get_and(&k, |_| ()).is_none());
                drop(k);
                coco::epoch::pin(|s| s.flush());
            }

            m.gc_tombstones();
            coco::epoch::pin(|s| s.flush());

            DROP_VECTOR.with(|v| {
                for i in 0..50 {
                    assert_eq!(v.borrow()[i], 0, "@ {}", i);
                    assert_eq!(v.borrow()[i+100], 0, "@ {}", i);
                }

                for i in 50..100 {
                    assert_eq!(v.borrow()[i], 1);
                    assert_eq!(v.borrow()[i+100], 1);
                }
            });
            coco::epoch::pin(|s| s.flush());
            //FIXME we don't drop contents on table drop
        }
        coco::epoch::pin(|s| s.flush());

        DROP_VECTOR.with(|v| {
            for i in 0..200 {
                assert_eq!(v.borrow()[i], 0, "@ {}", i);
            }
        });
    }
}