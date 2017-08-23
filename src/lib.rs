//!
//! A lock-free hashtable based on Clif Click's
//! [A Fast Wait-Free Hash Table](https://www.youtube.com/watch?v=WYXgtXWejRM)
//!
//! Note that this implementation table is _not_ wait-free;
//! a thread can race resizing and never reach a cannonical table,
//! and may not linearizable.
//!
//!
//!# Example #
//!
//!```rust
//!use click_map::HashMap;
//!
//!use std::sync::Arc;
//!use std::thread;
//!
//!let map = Arc::new(HashMap::new());
//!
//!
//!let threads_map = map.clone();
//!thread::spawn(move || for i in 0..50 {
//!    threads_map.insert(i*2, i*2);
//!});
//!
//!let threads_map = map.clone();
//!thread::spawn(move || for i in 0..50 {
//!    threads_map.insert(i*2 + 1, i*2 + 101);
//!});
//!
//!for i in 0..100 {
//!    'wait_for_insert: loop {
//!        match map.get_then(&i, |&v| v) {
//!            None => continue 'wait_for_insert,
//!            Some(val) if i % 2 == 0 => {
//!                assert_eq!(val, i);
//!                break 'wait_for_insert
//!            },
//!            Some(val) => {
//!                assert_eq!(val, i + 100);
//!                break 'wait_for_insert
//!            }
//!        }
//!    }
//!}
//!```
//!

// FIXME ensure that boxes have no pointees before they're freed
#![deny(unused_must_use)]

pub extern crate coco;

use std::borrow::Borrow;
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hash, Hasher};
use std::mem::forget;
use std::sync::atomic::{self, AtomicUsize, Ordering};
use Ordering::*;

use coco::epoch::{self, Atomic, Scope, Owned as Box, Ptr};

mod fmt;

pub struct HashMap<K, V, H = RandomState> {
    ptr: Atomic<Table<K, V, H>>,
}

unsafe impl<K, V, H> Send for HashMap<K, V, H>
where K: Send + Sync, V: Send + Sync, H: Send + Sync {}

unsafe impl<K, V, H> Sync for HashMap<K, V, H>
where K: Send + Sync + , V: Send + Sync + , H: Send + Sync {}

impl<K, V> HashMap<K, V, RandomState> {
    pub fn new() -> Self {
        Self::with_capacity(32)
    }

    pub fn with_capacity(cap: usize) -> Self {
        let cap = if cap < 32 { 32 } else { cap };
        let shift = cap.checked_next_power_of_two().unwrap().trailing_zeros();
        let table = alloc_table(shift as u8, Default::default());
        let ptr = Atomic::from_owned(table);
        HashMap { ptr }
    }
}

impl<K, V, H> HashMap<K, V, H>
where H: BuildHasher + Clone, {

    /// Insert a key-value pair into the map.
    /// Return true if the key was not already in the map.
    ///
    /// # Example
    ///
    /// ```rust
    /// use click_map::HashMap;
    ///
    /// let map = HashMap::new();
    /// assert_eq!(map.insert(1, "haha".to_string()), true);
    /// assert_eq!(map.insert(1, "hoho".to_string()), false);
    /// map.remove(&1);
    /// assert_eq!(map.insert(1, "hehe".to_string()), true);
    /// ```
    pub fn insert(&self, key: K, val: V) -> bool
    where K: Clone + PartialEq + Hash {
        self.insert_and(key, val, |_| ()).is_none()
    }

    /// Insert a key-value pair into the map
    /// and call a function f on the previous value, if one existed
    ///
    /// # Example
    ///
    /// ```rust
    /// use click_map::HashMap;
    ///
    /// let map = HashMap::new();
    /// assert_eq!(map.insert_and(1, "haha".to_string(), |old| old.clone()), None);
    /// assert_eq!(map.insert_and(1, "hoho".to_string(), |old| old.clone()), Some("haha".to_string()));
    /// map.remove(&1);
    /// assert_eq!(map.insert_and(1, "hehe".to_string(), |old| old.clone()), None);
    /// ```
    pub fn insert_and<F, T>(&self, key: K, val: V, f: F) -> Option<T>
    where
        K: Clone + PartialEq + Hash,
        F: for<'v >FnOnce(&'v V) -> T, {
        epoch::pin(|pin| {
            let (need_flush, old_val) = self.insert_key_then(key, val, pin, |cell, val, pin| cell.val.swap(val, pin));
            let t = match old_val {
                OwnedVal::Owned(v) => unsafe {
                    let t = v.0.as_ref().map(|v| f(&v.val));
                    v.free(pin);
                    t
                },
                _ => None
            };
            if need_flush { pin.flush() }
            t
        })
    }

    /// Insert a key-value pair into the map
    /// iff the key is not alredy in the map.
    /// Returns true if a new value is insterted
    ///
    /// # Example
    ///
    /// ```rust
    /// use click_map::HashMap;
    ///
    /// let map = HashMap::new();
    /// assert_eq!(map.insert_if_new(0x31, "haha".to_string()), true);
    /// assert_eq!(map.insert_if_new(0x31, "hoho".to_string()), false);
    /// map.remove(&0x31);
    /// assert_eq!(map.insert_if_new(0x31, "hehe".to_string()), true);
    /// ```
    pub fn insert_if_new(&self, key: K, val: V) -> bool
    where
        K: Clone + PartialEq + Hash, {
        self.insert_if_new_then(key, val, |_| ()).is_ok()
    }

    /// Insert a key-value pair into the map
    /// iff the key is not alredy in the map
    /// then call a function on the value in the map.
    pub fn insert_if_new_then<F, T>(&self, key: K, val: V, f: F) -> Result<T, T>
    where
        K: Clone + PartialEq + Hash,
        F: for<'v > FnOnce(&'v V) -> T, {
        epoch::pin(|pin| {
            let (need_flush, t) = self.insert_key_then(key, val, pin, move |cell, val, pin| unsafe {
                let val = cell.val.insert_if_empty(val, pin);
                match val {
                    Ok(v) => Ok(f(&v.deref().val)),
                    Err(v) => Err(f(&v.deref().val)),
                }
            });
            if need_flush { pin.flush() }
            t
        })
    }

    fn insert_key_then<'p, Then, Out>(&self, key: K, val: V, pin: &'p Scope, then: Then) -> (bool, Out)
    where
        K: 'p + Clone + PartialEq + Hash,
        V: 'p,
        H: 'p,
        Then: for<'a> FnOnce(&'a Cell<K, V>, Box<ValBox<V>>, &'p Scope) -> Out, {
        let pre = &self.ptr;
        let table_ptr = pre.load(Acquire, pin);
        let table = unsafe { table_ptr.deref() };
        let hash = table.hash(&key);
        let key = Box::new((hash, key));
        let val = Box::new(ValBox{ val, _align: [] });
        let (new_table, old_val) = table.insert(key, val, pin, then);
        let mut need_flush = false;
        if let Some(new_table) = new_table {
            if let Ok(..) = pre.compare_and_swap(table_ptr, new_table, AcqRel, pin) {
                unsafe {
                    pin.defer_free_array(table.hashes.load(Acquire, pin), table.capacity());
                    pin.defer_free_array(table.cells.load(Acquire, pin), table.capacity());
                    pin.defer_free(Ptr::from_raw(table));
                }
            }
            need_flush = true;
        }
        (need_flush, old_val)
    }

    /// Garbage collect the tombstones in the map
    /// by moving all live values to a fresh allocation.
    pub fn gc_tombstones(&self)
    where K: Clone + PartialEq{
        let pre = &self.ptr;
        epoch::pin(|pin| {
            let table_ptr = pre.load(Acquire, pin);
            let table = unsafe { table_ptr.deref() };
            let new_table = table.gc_tombstones(pin);
            if let Ok(..) = pre.compare_and_swap(table_ptr, new_table, AcqRel, pin) {
                unsafe {
                    pin.defer_free_array(table.hashes.load(Acquire, pin), table.capacity());
                    pin.defer_free_array(table.cells.load(Acquire, pin), table.capacity());
                    pin.defer_free(Ptr::from_raw(table));
                }
            }
            pin.flush();
        });
    }
}

impl<K, V, H> HashMap<K, V, H> {

    /// Call a closure on the value associated with a key, if one exists.
    ///
    /// # Example
    ///
    /// ```rust
    /// use click_map::HashMap;
    ///
    /// let map = HashMap::new();
    /// map.insert(1, "haha".to_string());
    /// assert_eq!(map.get_then(&1, |s| s.len()), Some(4));
    /// assert_eq!(map.get_then(&0, |s| s.len()), None);
    /// ```
    pub fn get_then<Q, F, T>(&self, key: &Q, and: F) -> Option<T>
    where
        H: BuildHasher,
        K: Borrow<Q>,
        Q: Hash + Eq,
        F: for<'v >FnOnce(&'v V) -> T, {
        epoch::pin(|pin| {
            let table = unsafe { self.ptr.load(Acquire, pin).deref() };
            let hash = table.hash(key);
            let v = table.get(hash, key, pin);
            v.map(and)
        })
    }

    /// remove a key-value pair from the map.
    pub fn remove<Q>(&self, key: &Q)
    where
        H: BuildHasher,
        K: Borrow<Q> + Clone + PartialEq,
        Q: Hash + Eq, {
        self.remove_and(key, |_| ());
    }

    /// remove a key-value pair from the map, and if one was removed
    /// call a function on the removed value.
    pub fn remove_and<Q, F, T>(&self, key: &Q, f: F) -> Option<T>
    where
        H: BuildHasher,
        K: Borrow<Q> + Clone + PartialEq,
        Q: Hash + Eq,
        F: FnOnce(&V) -> T {
        epoch::pin(|pin| {
            let table = unsafe { self.ptr.load(Acquire, pin).deref() };
            let hash = table.hash(key);
            let t = table.remove_and(hash, key, pin, f);
            t
        })
    }
}

impl<K, V, H> Drop for HashMap<K, V, H> {
    fn drop(&mut self) {
        use ::std::slice;
        epoch::pin(|pin| unsafe {
            let table = self.ptr.load(Acquire, pin).deref();
            let cells: &CellBlock<K, V> = table.cells.load(Acquire, pin).deref();
            let cells: &[CellBlock<K, V>] = slice::from_raw_parts(cells, table.capacity());
            for cell_block in cells {
                for cell in cell_block {
                    let _ = cell.val.remove_and(pin, |_| ());
                    cell.key.try_free(pin);
                }
            }
            pin.defer_free_array(table.hashes.load(Acquire, pin), table.capacity());
            pin.defer_free_array(table.cells.load(Acquire, pin), table.capacity());
            pin.defer_free(Ptr::from_raw(table));
            pin.flush()
        });
    }
}

const BLOCK_SIZE: usize = 8;

type HashBlock = [AtomicUsize; BLOCK_SIZE];
type CellBlock<K, V> = [Cell<K, V>; BLOCK_SIZE];

struct Table<K, V, H> {
    hashes: Atomic<HashBlock>,
    cells: Atomic<CellBlock<K, V>>,
    hasher: H,
    shift: u8,
    next: Atomic<Table<K, V, H>>,
}

//TODO it may be better to store keys and vals in seperate tables
struct Cell<K, V> {
    key: KeyPtr<K>,
    val: ValPtr<V>,
}

struct KeyPtr<K> {
    //FIXME we need to guarentee that this pointer is alinged enough
    //      to fit the flags.
    //FIXME we no longer store hashes here
    ptr: Atomic<(usize, K)>,
}

struct ValPtr<V> {
    ptr: Atomic<ValBox<V>>,
}

struct ValBox<V> {
    val: V,
    //FIXME replace with something guarenteed to be > 4bit aligned on 32bit
    _align: [&'static u8; 0],
}

impl<K, V> Cell<K, V> {
    fn empty() -> Self {
        Cell { key: KeyPtr{ ptr: Atomic::null() }, val: ValPtr{ ptr: Atomic::null() } }
    }
}

impl<K> KeyPtr<K> {
    fn empty<'a>() -> Ptr<'a, (usize, K)> {
        Ptr::null()
    }

    fn try_fill<'p>(&self, key: Box<(usize, K)>, pin: &'p Scope)
    -> Result<Ptr<'p, (usize, K)>, (Ptr<'p, (usize, K)>, Box<(usize, K)>)> {
        self.ptr.compare_and_swap_owned(KeyPtr::empty(), key, AcqRel, pin)
    }

    fn try_fill_owned<'p>(&self, key: OwnedKey<'p, K>, pin: &'p Scope)
    -> Result<(), (Ptr<'p, (usize, K)>, OwnedKey<'p, K>)> {
        // regular loads can be much faster than cas (on recent intel ~10x)
        // and don't invalidate other core's cache lines like cas might (again on recent intel)
        // so we check if the slot is empty before trying to insert
        let original = self.ptr.load(Acquire, pin);
        if !original.is_null() {
            return Err((original, key))
        }
        match key {
            OwnedKey::Boxed(key) => self.ptr
                .compare_and_swap_owned(KeyPtr::empty(), key, AcqRel, pin)
                .map(|_| ())
                .map_err(|(p, k)| (p, OwnedKey::Boxed(k))),
            OwnedKey::Ptr(key) => {
                let key = key.into_ptr();
                self.ptr
                    .compare_and_swap(KeyPtr::empty(), key, AcqRel, pin)
                    .map_err(|p| (p, OwnedKey::Ptr(Owned(key))))
            },
        }
    }

    fn get<'p>(&self, pin: &'p Scope) -> Option<&'p K> {
        unsafe { self.ptr.load(Acquire, pin).as_ref().map(|&(_, ref k)| k) }
    }

    fn load<'p>(&self, pin: &'p Scope) -> Ptr<'p, (usize, K)> {
        self.ptr.load(Acquire, pin)
    }

    fn start_move<'p>(&self, pin: &'p Scope)
    -> Option<OwnedKey<'p, K>>
    where K: Clone, {
        let original = self.load(pin);
        if original.is_null() { return None }

        fn duplicate_key<'p, K: Clone>(original: Ptr<'p, (usize, K)>) -> OwnedKey<'p, K> {
            let clone = unsafe {<(usize, K) as Clone>::clone(original.deref())};
            OwnedKey::Boxed(Box::new(clone))
        }

        match original.tag() {
            TOMBSTONE | MOVED => return Some(duplicate_key(original)),
            _ => (),
        }
        match self.ptr.compare_and_swap(original, original.with_tag(MOVED), AcqRel, pin) {
            Ok(()) => Some(OwnedKey::Ptr(Owned(original))),
            Err(..) => Some(duplicate_key(original)),
        }
    }

    fn try_free<'p>(&self, pin: &'p Scope) -> bool {
        let original = self.load(pin);
        if original.is_null() { return false }
        match original.tag() {
            TOMBSTONE | MOVED => return false,
            _ => (),
        }
        match self.ptr.compare_and_swap(original, original.with_tag(TOMBSTONE), AcqRel, pin) {
            Err(..) => return false,
            Ok(()) => unsafe{ pin.defer_drop(original); true },
        }
    }
}

enum OwnedKey<'p, K: 'p> {
    Boxed(Box<(usize, K)>),
    Ptr(Owned<'p, (usize, K)>),
}

impl<'p, K: 'p> OwnedKey<'p, K> {
    fn free(self, pin: &'p Scope) {
        match self {
            OwnedKey::Boxed(b) => unsafe { pin.defer_drop(b.into_ptr(pin)) },
            OwnedKey::Ptr(p) => unsafe { pin.defer_drop(p.0) },
        }
    }
}

impl<'p, K: 'p> ::std::ops::Deref for OwnedKey<'p, K> {
    type Target = (usize, K);
    fn deref(&self) -> &Self::Target {
        match self {
            &OwnedKey::Boxed(ref b) => &**b,
            &OwnedKey::Ptr(ref p) => unsafe { p.0.deref() },
        }
    }
}

const MOVING: usize = 1;
const MOVED: usize = 4;
const TOMBSTONE: usize = 2;

#[derive(Debug)]
enum ValPtrGetErr {
    Nothing,
    Tombstone,
    Moved,
}

impl<V> ValPtr<V> {
    fn swap<'p>(&self, val: Box<ValBox<V>>, pin: &'p Scope) -> OwnedVal<'p, V> {
        let new = val.into_ptr(pin);
        let old = self.ptr.swap(new, AcqRel, pin);
        match old.tag() {
            TOMBSTONE => OwnedVal::Empty,
            MOVED => OwnedVal::Moved,
            _ if old.is_null() => OwnedVal::Empty,
            _ => OwnedVal::Owned(Owned(old)),
        }
    }

    fn insert_if_empty<'p>(&self, mut val: Box<ValBox<V>>, pin: &'p Scope) -> Result<Ptr<'p, ValBox<V>>, Ptr<'p, ValBox<V>>> {
        let mut old = Ptr::null();
        loop {
            let res = self.ptr.compare_and_swap_owned(old, val, AcqRel, pin);
            match res {
                Ok(p) => return Ok(p),
                Err((current, new)) => {
                    let current_tag = current.tag();
                    if current_tag == TOMBSTONE || current_tag == MOVED {
                        old = current;
                        val = new;
                        continue
                    }
                    return Err(current)
                },
            }
        }
    }

    fn get<'p>(&self, pin: &'p Scope) -> Result<&'p V, ValPtrGetErr> {
        use ValPtrGetErr::*;
        let ptr = self.ptr.load(Acquire, pin);
        let raw = ptr.as_raw() as usize;
        if raw != TOMBSTONE && raw != MOVED {
            unsafe { ptr.as_ref().map(|v| &v.val).ok_or_else(|| Nothing) }
        } else if raw == TOMBSTONE {
            Err(Tombstone)
        } else {
            Err(Moved)
        }
    }

    fn load_old<'p>(&self, pin: &'p Scope) -> OldVal<'p, V> {
        OldVal(self.ptr.load(Acquire, pin))
    }

    fn start_move<'p>(&self, current: &NewVal<'p, V>, new: &OldVal<'p, V>, pin: &'p Scope)
    -> Result<Option<Owned<'p, ValBox<V>>>, NewVal<'p, V>> {
         match self.ptr.compare_and_swap(current.0, new.moving().0, AcqRel, pin) {
             Ok(()) if current.is_owned() => Ok(Some(Owned(current.0))),
             Ok(()) => Ok(None),
             Err(ptr) => Err(NewVal(ptr)),
         }
    }

    fn finish_move<'p>(&self, val: OldVal<'p, V>, pin: &'p Scope)
    -> Result<(), ()> {
         self.ptr.compare_and_swap(val.moving().0, val.0, AcqRel, pin).map_err(|_| ())
    }

    fn mark_as_moved<'p>(&self, val: &OldVal<'p, V>, pin: &'p Scope)
    -> Result<OldVal<'p, V>, OldVal<'p, V>> {
        match self.ptr.compare_and_swap(val.0, val.0.with_tag(MOVED), AcqRel, pin) {
            Err(ptr) => Err(OldVal(ptr)),
            Ok(()) => Ok(OldVal(val.0)),
        }
    }

    fn remove_and<F, T>(&self, pin: &Scope, f: F) -> Result<T, (ValPtrGetErr, F)>
    where F: FnOnce(&V) -> T, {
        use ValPtrGetErr::*;
        let old = self.ptr.swap(Ptr::null().with_tag(TOMBSTONE), AcqRel, pin);
        match old.tag() {
            TOMBSTONE => Err((Tombstone, f)),
            MOVED => Err((Moved, f)),
            MOVING => Err((Nothing, f)),
            _ if old.is_null() => Err((Nothing, f)),
            //TODO check
            _ => unsafe {
                    match old.as_ref() {
                        None => Err((Nothing, f)),
                        Some(v) => {
                            let t = f(&v.val);
                            pin.defer_drop(old);
                            Ok(t)
                    },
                }
            }
        }
    }

    fn empty<'p>() -> NewVal<'p, V> {
        NewVal(Ptr::null())
    }
}

#[must_use]
enum OwnedVal<'p, V:'p> {
    Empty,
    Moved,
    Owned(Owned<'p, ValBox<V>>)
}

#[must_use]
struct Owned<'p, T: 'p>(Ptr<'p, T>);

impl<'p, T: 'p> Owned<'p, T> {
    fn free(self, pin: &'p Scope) {
        unsafe { pin.defer_drop(self.0) }
        forget(self)
    }

    fn into_ptr(self) -> Ptr<'p, T> {
        let ptr = self.0;
        forget(self);
        ptr
    }
}

impl<'p, T> Drop for Owned<'p, T> {
    fn drop(&mut self) {
        unreachable!("Owned must be used.")
    }
}

#[must_use]
struct NewVal<'p, V:'p>(Ptr<'p, ValBox<V>>);

impl<'p, V: 'p> NewVal<'p, V> {
    fn is_owned(&self) -> bool {
        self.0.tag() == 0 && !self.0.is_null()
    }
}

#[must_use]
struct OldVal<'p, V:'p>(Ptr<'p, ValBox<V>>);

impl<'p, V:'p> OldVal<'p, V> {
    fn needs_move(&self) -> bool {
        self.0.tag() == 0 && !self.0.is_null()
    }

    fn cannot_move(&self) -> bool {
        let tag = self.0.tag();
        tag == MOVED || tag == MOVING || (self.0.is_null() && tag != TOMBSTONE)
    }

    fn is_tombstone(&self) -> bool {
        self.0.tag() == TOMBSTONE
    }

    fn moving(&self) -> Self {
        //we want to be able to move tombstones
        let old_tag = self.0.tag();
        OldVal(self.0.with_tag(MOVING | old_tag))
    }
}

impl<K, V, H> Table<K, V, H>
where H: Clone, {

    fn insert<'p, F, T>(&self, key: Box<(usize, K)>, val: Box<ValBox<V>>, pin: &'p Scope, on_find: F)
    -> (Option<Ptr<'p, Self>>, T)
    where
        K: Clone + PartialEq,
        F: for<'a> FnOnce(&'a Cell<K, V>, Box<ValBox<V>>, &'p Scope) -> T {
        //TODO I think pre-checking breaks linearizability
        //if let Some(table) = self.next.load(pin).as_ref() {
        //    return table.insert(key, val, pin)
        //}
        match self.try_emplace(key, val, pin, on_find) {
            // If a table started resizing after we started inserting we may not have seen it
            // and other threads may not move our entry if we replaced an empty which they
            // have already bypassed.
            // Furthermore, next should be on the same cacheline as table,
            // meaning this will likely only cause a miss if somone started a resize.
            Ok((loc, old)) => {
                let ptr = self.next.load(Acquire, pin);
                if let Some(table) = unsafe { ptr.as_ref() } {
                    self.move_contents_to(table, loc, pin);
                    return (Some(ptr), old)
                }
                return (None, old)
            },
            Err((last, key, val, on_find)) => {
                let new_table = self.alloc_next_table();
                let next = self.next.compare_and_swap_owned(Ptr::null(), new_table, AcqRel, pin);
                let table = match next { Ok(p) | Err((p, _)) => p, };
                let t = unsafe { table.deref() };
                self.move_contents_to(t, last, pin);
                let (new_table, old_val) = t.insert(key, val, pin, on_find);
                (new_table.or(Some(table)), old_val)
            },
        }
    }

    fn try_emplace<'p, F, T>(&self, mut key: Box<(usize, K)>, val: Box<ValBox<V>>, pin: &'p Scope, on_find: F)
    -> Result<(isize, T), (isize, Box<(usize, K)>, Box<ValBox<V>>, F)>
    where
        K: PartialEq,
        F: for<'a> FnOnce(&'a Cell<K, V>, Box<ValBox<V>>, &'p Scope) -> T {
        unsafe {
            let hashes = self.hashes.load(Relaxed, pin).as_raw();
            let cells = self.cells.load(Relaxed, pin).as_raw();
            let hash = key.0;
            let mut last = hash as isize;
            for i in search_path(hash, self.shift).take(self.max_probe_len()) {
                let hash_block = &*hashes.offset(i);
                let cell_block = &*cells.offset(i);
                for j in 0..BLOCK_SIZE {
                    last = i;
                    let found_hash = hash_block[j].load(Relaxed);
                    if !(found_hash == 0 || found_hash == hash) {
                        continue
                    }
                    match cell_block[j].key.try_fill(key, pin) {
                        Ok(..) => {
                            hash_block[j].store(hash, Relaxed);
                            let old = on_find(&cell_block[j], val, pin);
                            return Ok((i, old))
                        }
                        Err((found, k)) => {
                            key = k;
                            if found.as_ref().map(|f| f == &*key).unwrap_or_else(|| false) {
                                hash_block[j].store(hash, Relaxed);
                                let old = on_find(&cell_block[j], val, pin);
                                return Ok((i, old))
                            }
                        },
                    }
                }
            }
            Err((last, key, val, on_find))
        }
    }

    fn gc_tombstones<'p>(&self, pin: &'p Scope) -> Ptr<'p, Self>
    where K: Clone + PartialEq {
        let new_table = alloc_table(self.shift, self.hasher.clone());
        let next = self.next.compare_and_swap_owned(Ptr::null(), new_table, AcqRel, pin);
        let table = match next { Ok(p) | Err((p, _)) => p, };
        let t = unsafe { table.deref() };
        self.move_contents_to(t, 0, pin);
        table
    }

    fn alloc_next_table(&self) -> Box<Self> {
        alloc_table(self.shift + 1, self.hasher.clone())
    }
}

impl<K, V, H> Table<K, V, H> {
    fn get<'p, Q>(&self, hash: usize, q: &Q, pin: &'p Scope) -> Option<&'p V>
    where K: Borrow<Q>, Q: Eq {
        use ValPtrGetErr::*;
        unsafe {
            let hashes = self.hashes.load(Relaxed, pin).as_raw();
            let cells = self.cells.load(Relaxed, pin).as_raw();
            let mut seach_next_table = false;
            for i in search_path(hash, self.shift).take(self.max_probe_len()) {
                let hash_block = &*hashes.offset(i);
                let cell_block = &*cells.offset(i);
                for j in 0..BLOCK_SIZE {
                    if hash_block[j].load(Relaxed) != hash {
                        continue
                    }
                    match cell_block[j].key.get(pin) {
                        Some(k) if q.eq(k.borrow()) => {
                            match cell_block[j].val.get(pin) {
                                Ok(v) => return Some(v),
                                Err(Nothing) | Err(Tombstone) => return None,
                                Err(Moved) => {
                                    seach_next_table = true;
                                    break
                                },
                            }
                        },
                        Some(..) => continue,
                        None => return None,
                    }
                }
            }
            if seach_next_table {
                self.next.load(Acquire, pin).deref().get(hash, q, pin)
            } else {
                None
            }
        }
    }

    fn remove_and<'p, Q, F, T>(&self, hash: usize, q: &Q, pin: &'p Scope, mut f: F) -> Option<T>
    where
        K: Borrow<Q> + Clone + PartialEq,
        Q: Eq,
        F: FnOnce(&V) -> T, {
        use ValPtrGetErr::*;
        unsafe {
            let hashes = self.hashes.load(Relaxed, pin).as_raw();
            let cells = self.cells.load(Relaxed, pin).as_raw();
            for i in search_path(hash, self.shift).take(self.max_probe_len()) {
                let hash_block = &*hashes.offset(i);
                let cell_block = &*cells.offset(i);
                for j in 0..BLOCK_SIZE {
                    if hash_block[j].load(Relaxed) != hash {
                        continue
                    }
                    match cell_block[j].key.get(pin) {
                        // found our key
                        Some(k) if q.eq(k.borrow()) => {
                            //FIXME free should be done after move_contents_to?
                            match cell_block[j].val.remove_and(pin, f) {
                                // If we removed val we need to try and propagate the removal to the net table
                                Ok(t) => {
                                    if let Some(table) = self.next.load(Acquire, pin).as_ref() {
                                        //FIXME handle moving tombstones
                                        self.move_tombstone(table, (i, j), hash, pin);
                                    }
                                    return Some(t)
                                },
                                //if there was no val we're done
                                Err((Tombstone, _)) | Err((Nothing, _)) => return None,

                                // If val was moved we need to check the next table
                                Err((Moved, g)) => {
                                    f = g;
                                    break
                                },
                            }
                        },

                        // not our key, keep looking
                        Some(..) => continue,

                        // key not found in table
                        None => return None,
                    }
                }
            }
            self.next.load(Acquire, pin).deref().remove_and(hash, q, pin, f)
        }
    }

    unsafe fn move_tombstone<'p>(&self, new: &Self, (tomb_i, tomb_j): (isize, usize), hash: usize, pin: &'p Scope)
    where K: Clone + PartialEq {
        let old_cells = self.cells.load(Acquire, pin).as_raw();
        let old_block = &*old_cells.offset(tomb_i);
        let old_cell = &old_block[tomb_j];
        let key = old_cell.key.start_move(pin).unwrap();
        let new_cells = new.cells.load(Relaxed, pin).as_raw();
        let new_hashes = new.hashes.load(Relaxed, pin).as_raw();
        let new_shift = new.shift;
        move_cell(old_cell, new_cells, new_hashes, new_shift, hash, key, pin);
    }

    //TODO only move until we reach a Moved, after that some other thread will move them
    //     but how do we know when the map is finished?
    fn move_contents_to(&self, new: &Self, start_loc: isize, pin: &Scope)
    where K: Clone + PartialEq {
        let old_hashes = self.hashes.load(Acquire, pin).as_raw();
        let old_cells = self.cells.load(Acquire, pin).as_raw();
        let new_hashes = new.hashes.load(Acquire, pin).as_raw();
        let new_cells = new.cells.load(Acquire, pin).as_raw();
        let new_shift = new.shift;
        unsafe {
            for old_off in (start_loc..(1 << self.shift)).chain(0..start_loc) {
                let old_hash_block = &*old_hashes.offset(old_off);
                let old_cell_block = &*old_cells.offset(old_off);
                for j in 0..BLOCK_SIZE {
                    //TODO can we break if we find an empty cell
                    //     based on the assumption that if somone is writing it
                    //     they'll move it to the new table?
                    let old_cell = &old_cell_block[j];
                    let old_val = old_cell.val.load_old(pin);
                    if !old_val.needs_move() {
                        if old_val.is_tombstone() {
                            old_cell.key.try_free(pin);
                        }
                        continue
                    }
                    let key = (*old_cell).key.start_move(pin);
                    let key = match key {
                        None => continue,
                        Some(key) => key,
                    };
                    let hash = old_hash_block[j].load(Relaxed);
                    if hash == 0 { continue }
                    move_cell(old_cell, new_cells, new_hashes, new_shift, hash, key, pin);
                }
            }
        }
        let next = new.next.load(Acquire, pin);
        if let Some(next) = unsafe { next.as_ref() } {
            new.move_contents_to(next, start_loc, pin)
        }
    }

    #[inline(always)]
    fn max_probe_len(&self) -> usize {
        //TODO benchmark
        // (1 << self.shift) / 8
        4
    }

    #[inline(always)]
    fn capacity(&self) -> usize {
        (1 << self.shift)
    }

    //from hashmap, allows 0 to be used as EMPTY_HASH
    fn hash<T>(&self, key: &T) -> usize
    where
        H: BuildHasher,
        T: Hash, {
        let mut hasher = self.hasher.build_hasher();
        key.hash(&mut hasher);
        let hash = hasher.finish() as usize;
        (1 << (std::mem::size_of::<usize>() * 8 - 1) | hash)
    }
}

unsafe fn move_cell<'p, K, V>(
    old_cell: *const Cell<K, V>,
    new_cells: *const CellBlock<K, V>,
    new_hashes: *const HashBlock,
    new_shift: u8,
    hash: usize,
    key: OwnedKey<'p, K>,
    pin: &'p Scope,
) -> bool
where K: PartialEq {
    //FIXME handle running out of space (concurrent writes)
    //      (should not be a problem, we diallow new inserts until the move is done)
    let new_cell = match find_new_cell(new_cells, new_hashes, new_shift, hash, key, pin) {
        Some(new_cell) => new_cell, None => return false,
    };
    let mut expected_new_val = ValPtr::empty();
    'swap: loop {
        //TODO we can remove this get by putting it outside the function
        let old_val = (*old_cell).val.load_old(pin);
        if old_val.cannot_move() {
            return true
        }
        let got = (*new_cell).val.start_move(&expected_new_val, &old_val, pin);
        match got {
            Ok(Some(owned)) => owned.free(pin),
            Ok(None) => {},
            Err(new_val) => {
                expected_new_val = new_val;
                continue 'swap
            },
        }
        if (*old_cell).val.mark_as_moved(&old_val, pin).is_err() {
            continue 'swap
        };
        atomic::fence(Ordering::SeqCst); //TODO which ordering, and is this needed?
        // If this suceeds we're done, if this failed someone raced us to the val
        // and they'll finish the cell
        let _ = (*new_cell).val.finish_move(old_val, pin).is_ok();
        return true
    }
}

unsafe fn find_new_cell<'p, K, V>(
    new_cells: *const CellBlock<K, V>,
    new_hashes: *const HashBlock,
    new_shift: u8,
    hash: usize,
    mut key: OwnedKey<'p, K>,
    pin: &'p Scope,
) -> Option<*const Cell<K, V>>
where K: PartialEq {
    let mut new_cell = None;
    'search: for new_off in search_path(key.0, new_shift) {
        let hash_block = &*new_hashes.offset(new_off);
        let cell_block = &*new_cells.offset(new_off);
        for j in 0..BLOCK_SIZE {
            let found_hash = hash_block[j].load(Relaxed);
            if !(found_hash == 0 || found_hash == hash) {
                continue
            }
            match cell_block[j].key.try_fill_owned(key, pin) {
                Err((found, k)) => {
                    if found.as_ref() == Some(&*k) {
                        hash_block[j].store(hash, Relaxed);
                        new_cell = Some(&cell_block[j] as *const _);
                        k.free(pin);
                        break 'search
                    }
                    key = k
                }
                Ok(..) => {
                    hash_block[j].store(hash, Relaxed);
                    new_cell = Some(&cell_block[j] as *const _);
                    break 'search
                },
            }
        }
    }
    new_cell
}

fn alloc_table<K, V, H>(shift: u8, hasher: H) -> Box<Table<K, V, H>> {
    unsafe {
        let mut vcells: Vec<CellBlock<K, V>> = (0..(1 << shift))
            .map(|_| [
                Cell::empty(), Cell::empty(), Cell::empty(), Cell::empty(),
                Cell::empty(), Cell::empty(), Cell::empty(), Cell::empty()
            ])
            .collect();
        let cells = Atomic::from_ptr(Ptr::from_raw(vcells.as_mut_ptr()));
        forget(vcells);

        let mut vhashes: Vec<HashBlock> = (0..(1 << shift))
            .map(|_| [
                AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0),
                AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0),
            ])
            .collect();
        let hashes = Atomic::from_ptr(Ptr::from_raw(vhashes.as_mut_ptr()));
        forget(vhashes);

        Box::new(Table {
            hashes, cells, shift, hasher, next: Atomic::null(),
        })
    }
}

impl<K, V, H> Drop for Table<K, V, H> {
    fn drop(&mut self) {
        unsafe {
            epoch::unprotected(|pin| {
                let hashes = self.hashes.load(Ordering::Relaxed, pin).as_raw();
                let cells = self.cells.load(Ordering::Relaxed, pin).as_raw();
                let _ = Vec::from_raw_parts(hashes as *mut HashBlock, 0, 1 << self.shift);
                let _ = Vec::from_raw_parts(cells as *mut CellBlock<K, V>, 0, 1 << self.shift);
            })
        }
    }
}

fn search_path(hash: usize, shift: u8) -> SearchPath {
    SearchPath{hash, mask: (1 << shift) - 1}
}

struct SearchPath {
    hash: usize,
    mask: usize,
}

impl Iterator for SearchPath {
    type Item = isize;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let pos = self.hash & self.mask;
        self.hash += 1;
        Some(pos as isize)
    }
}

// struct SearchPath {
//     hash: u64,
//     capacity: usize,
// }
//
// impl Iterator for SearchPath {
//     type Item = isize;
//
//     #[inline(always)]
//     fn next(&mut self) -> Option<Self::Item> {
//         let x = U128::new(self.hash);
//         let n = U128::new(self.capacity as u64);
//         let mul: U128 = x * n;
//         let shft: U128 = mul >> 64;
//         let pos = shft.low64() as isize;
//         self.hash += 1;
//         Some(pos)
//     }
// }
