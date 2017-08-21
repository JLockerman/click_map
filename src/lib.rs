//!
//! A lock-free hashtable based on Clif Click's
//! [A Fast Wait-Free Hash Table](https://www.youtube.com/watch?v=WYXgtXWejRM)
//!
//! Note that this implementation table is __not__ wait-free;
//! a thread can race resizing and never reach a cannonical table,
//! and may not linearizable.
//!

// FIXME ensure that boxes have no pointees before they're freed
#![deny(unused_must_use)]

extern crate coco;

use std::borrow::Borrow;
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hash, Hasher};
use std::mem::{drop, forget};
use std::ptr;
use std::sync::Arc;
use std::sync::atomic::{self, Ordering};
use Ordering::*;

use coco::epoch::{self, Atomic, Scope, Owned as Box, Ptr};

#[derive(Clone)]
pub struct HashMap<K, V, H = RandomState> {
    //TODO we should probably have the Arc be a user responsibility
    ptr: Arc<Atomic<Table<K, V, H>>>,
}

impl<K, V> HashMap<K, V, RandomState> {
    pub fn new() -> Self {
        Self::with_capacity(32)
    }

    pub fn with_capacity(cap: usize) -> Self {
        let cap = if cap < 32 { 32 } else { cap };
        let shift = cap.checked_next_power_of_two().unwrap().trailing_zeros();
        let table = alloc_table(shift as u8, Default::default());
        let atom = Atomic::from_owned(table);
        HashMap {
            ptr: Arc::new(atom),
        }
    }
}

impl<K, V, H> HashMap<K, V, H>
where H: BuildHasher + Clone {
    pub fn insert(&self, key: K, val: V) -> bool
    where K: PartialEq + Hash {
        self.insert_and(key, val, |_| ()).is_some()
    }

    pub fn insert_and<F, T>(&self, key: K, val: V, f: F) -> Option<T>
    where
        K: PartialEq + Hash,
        F: for<'v >FnOnce(&'v V) -> T, {
        let pre = &*self.ptr;
        epoch::pin(|pin| {
            let table_ptr = pre.load(Acquire, pin);
            let table = unsafe { table_ptr.deref() };
            let mut hasher = table.hasher.build_hasher();
            key.hash(&mut hasher);
            let hash = hasher.finish();
            let key = Box::new((hash, key));
            let val = Box::new(ValBox{ val, _align: [] });
            let (new_table, old_val) = table.insert(key, val, pin);
            let mut need_flush = false;
            if let Some(new_table) = new_table {
                let _ = pre.compare_and_swap(table_ptr, new_table, AcqRel, pin);
                need_flush = true;
            }
            let t = match old_val {
                OwnedVal::Owned(v) => unsafe {
                    let t = v.0.as_ref().map(|v| f(&v.val));
                    v.free(pin);
                    need_flush = true;
                    t
                },
                _ => None
            };
            if need_flush { pin.flush() }
            t
        })
    }
}

impl<K, V, H> HashMap<K, V, H> {
    pub fn get_and<Q, F, T>(&self, key: &Q, and: F) -> Option<T>
    where
        H: BuildHasher,
        K: Borrow<Q>,
        Q: Hash + Eq,
        F: for<'v >FnOnce(&'v V) -> T, {
        epoch::pin(|pin| {
            let table = unsafe { self.ptr.load(Acquire, pin).deref() };
            let mut hasher = table.hasher.build_hasher();
            key.hash(&mut hasher);
            let hash = hasher.finish();
            let v = table.get(hash, key, pin);
            v.map(and)
        })
    }

    pub fn remove<Q>(&self, key: &Q)
    where
        H: BuildHasher,
        K: Borrow<Q> + PartialEq,
        Q: Hash + Eq, {
        self.remove_and(key, |_| ());
    }

    pub fn remove_and<Q, F, T>(&self, key: &Q, f: F) -> Option<T>
    where
        H: BuildHasher,
        K: Borrow<Q> + PartialEq,
        Q: Hash + Eq,
        F: FnOnce(&V) -> T {
        epoch::pin(|pin| {
            let table = unsafe { self.ptr.load(Acquire, pin).deref() };
            let mut hasher = table.hasher.build_hasher();
            key.hash(&mut hasher);
            let hash = hasher.finish();
            table.remove_and(hash, key, pin, f)
        })
    }

    //TODO remove_and?
}

struct Table<K, V, H> {
    table: Atomic<Cell<K, V>>,
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
    ptr: Atomic<(u64, K)>,
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
    fn empty<'a>() -> Ptr<'a, (u64, K)> {
        Ptr::null()
    }

    fn try_fill<'p>(&self, key: Box<(u64, K)>, pin: &'p Scope)
    -> Result<Ptr<'p, (u64, K)>, (Ptr<'p, (u64, K)>, Box<(u64, K)>)> {
        self.ptr.compare_and_swap_owned(KeyPtr::empty(), key, AcqRel, pin)
    }

    fn try_fill_owned<'p>(&self, key: OwnedKey<'p, K>, pin: &'p Scope)
    -> Result<(), (Ptr<'p, (u64, K)>, OwnedKey<'p, K>)> {
        match key {
            OwnedKey::Boxed(key) => self.ptr.compare_and_swap_owned(KeyPtr::empty(), key, AcqRel, pin)
                .map(|_| ())
                .map_err(|(p, k)| (p, OwnedKey::Boxed(k))),
            OwnedKey::Ptr(key) => self.ptr.compare_and_swap(KeyPtr::empty(), key.0, AcqRel, pin)
                .map_err(|p| (p, OwnedKey::Ptr(key))),
        }
    }

    fn get<'p>(&self, pin: &'p Scope) -> Option<&'p K> {
        unsafe { self.ptr.load(Acquire, pin).as_ref().map(|&(_, ref k)| k) }
    }

    fn load<'p>(&self, pin: &'p Scope) -> Ptr<'p, (u64, K)> {
        self.ptr.load(Acquire, pin)
    }

    fn start_move<'p>(&self, pin: &'p Scope)
    -> Option<OwnedKey<'p, K>> {
        let original = self.load(pin);
        if original.is_null() { return None }
        match original.tag() {
            TOMBSTONE | MOVED => return Some(OwnedKey::Boxed(Box::new(unsafe { ptr::read(original.as_raw()) }))),
            _ => (),
        }
        match self.ptr.compare_and_swap(original, original.with_tag(MOVED), AcqRel, pin) {
            Ok(()) => Some(OwnedKey::Ptr(Owned(original))),
            Err(..) => Some(OwnedKey::Boxed(Box::new(unsafe { ptr::read(original.as_raw()) }))),
        }
    }

    fn try_free<'p>(&self, pin: &'p Scope) {
        let original = self.load(pin);
        if original.is_null() { return }
        match original.tag() {
            TOMBSTONE | MOVED => return,
            _ => (),
        }
        match self.ptr.compare_and_swap(original, original.with_tag(TOMBSTONE), AcqRel, pin) {
            Err(..) => return,
            Ok(()) => unsafe{ pin.defer_drop(original) },
        }
    }
}

enum OwnedKey<'p, K: 'p> {
    Boxed(Box<(u64, K)>),
    Ptr(Owned<'p, (u64, K)>),
}

impl<'p, K: 'p> OwnedKey<'p, K> {
    fn free(self, pin: &'p Scope) {
        match self {
            //FIXME we never drop keys atm
            OwnedKey::Boxed(b) => unsafe {pin.defer_drop(b.into_ptr(pin))},
            OwnedKey::Ptr(p) => unsafe {pin.defer_drop(p.0)},
        }
    }
}

impl<'p, K: 'p> ::std::ops::Deref for OwnedKey<'p, K> {
    type Target = (u64, K);
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
where H: Clone {

    fn insert<'p>(&self, key: Box<(u64, K)>, val: Box<ValBox<V>>, pin: &'p Scope)
    -> (Option<Ptr<'p, Self>>, OwnedVal<'p, V>)
    where K: PartialEq {
        //TODO I think pre-checking breaks linearizability
        //if let Some(table) = self.next.load(pin).as_ref() {
        //    return table.insert(key, val, pin)
        //}
        match self.try_emplace(key, val, pin) {
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
            Err((last, key, val)) => {
                let new_table = self.alloc_next_table();
                let next = self.next.compare_and_swap_owned(Ptr::null(), new_table, AcqRel, pin);
                let table = match next { Ok(p) | Err((p, _)) => p, };
                let t = unsafe { table.deref() };
                self.move_contents_to(t, last, pin);
                //FIXME do this in caller after CAS
                unsafe {
                    pin.defer_free_array(self.table.load(Acquire, pin), self.capacity());
                    pin.defer_free(Ptr::from_raw(self));
                }
                let (new_table, old_val) = t.insert(key, val, pin);
                (new_table.or(Some(table)), old_val)
            },
        }
    }

    fn try_emplace<'p>(&self, mut key: Box<(u64, K)>, val: Box<ValBox<V>>, pin: &'p Scope)
    -> Result<(isize, OwnedVal<'p, V>), (isize, Box<(u64, K)>, Box<ValBox<V>>)>
    where K: PartialEq {
        //TODO defer_free
        //TODO use new key in new table so we can free old one
        unsafe {
            let table = self.table.load(Acquire, pin).as_raw();
            let mut last = key.0 as isize;
            for i in search_path(key.0, self.shift).take(self.max_probe_len()) {
                last = i;
                let cell = table.offset(i);
                match (*cell).key.try_fill(key, pin) {
                    Ok(..) => {
                        let old = (*cell).val.swap(val, pin);
                        return Ok((i, old))
                    }
                    Err((found, k)) => {
                        key = k;
                        if found.as_ref().map(|f| f == &*key).unwrap_or_else(|| false) {
                            let old = (*cell).val.swap(val, pin);
                            return Ok((i, old))
                        }
                    },
                }
            }
            Err((last, key, val))
        }
    }

    fn alloc_next_table(&self) -> Box<Self> {
        alloc_table(self.shift + 1, self.hasher.clone())
    }
}

impl<K, V, H> Table<K, V, H> {
    fn get<'p, Q>(&self, hash: u64, q: &Q, pin: &'p Scope) -> Option<&'p V>
    where K: Borrow<Q>, Q: Eq {
        use ValPtrGetErr::*;
        unsafe {
            let table = self.table.load(Acquire, pin).as_raw();
            let mut seach_next_table = false;
            for i in search_path(hash, self.shift).take(self.capacity()) {
                let cell = table.offset(i);
                match (*cell).key.get(pin) {
                    Some(k) if q.eq(k.borrow()) => {
                        match (*cell).val.get(pin) {
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
            if seach_next_table {
                self.next.load(Acquire, pin).deref().get(hash, q, pin)
            } else {
                None
            }
        }
    }

    fn remove_and<'p, Q, F, T>(&self, hash: u64, q: &Q, pin: &'p Scope, mut f: F) -> Option<T>
    where
        K: Borrow<Q> + PartialEq,
        Q: Eq,
        F: FnOnce(&V) -> T, {
        use ValPtrGetErr::*;
        unsafe {
            let table = self.table.load(Acquire, pin).as_raw();
            for i in search_path(hash, self.shift).take(self.capacity()) {
                let cell = table.offset(i);
                match (*cell).key.get(pin) {
                    // found our key
                    Some(k) if q.eq(k.borrow()) => {
                        //FIXME free should be done after move_contents_to?
                        match (*cell).val.remove_and(pin, f) {
                            // If we removed val we need to try and propagate the removal to the net table
                            Ok(t) => {
                                if let Some(table) = self.next.load(Acquire, pin).as_ref() {
                                    //FIXME handle moving tombstones
                                    self.move_tombstone(table, i, pin);
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
            self.next.load(Acquire, pin).deref().remove_and(hash, q, pin, f)
        }
    }

    unsafe fn move_tombstone<'p>(&self, new: &Self, tomb_loc: isize, pin: &'p Scope)
    where K: PartialEq {
        let old_table = self.table.load(Acquire, pin).as_raw();
        let old_cell = old_table.offset(tomb_loc);
        let key = (*old_cell).key.start_move(pin).unwrap();
        let new_table = new.table.load(Acquire, pin).as_raw();
        let new_shift = new.shift;
        move_cell(old_cell, new_table, new_shift, key, pin);
    }

    //TODO only move until we reach a Moved, after that some other thread will move them
    //     but how do we know when the map is finished?
    fn move_contents_to(&self, new: &Self, start_loc: isize, pin: &Scope)
    where K: PartialEq {
        let old_table = self.table.load(Acquire, pin).as_raw();
        let new_table = new.table.load(Acquire, pin).as_raw();
        let new_shift = new.shift;
        unsafe {
            for old_off in (start_loc..(1 << self.shift)).chain(0..start_loc) {
                let old_cell = old_table.offset(old_off);
                let old_val = (*old_cell).val.load_old(pin);
                if !old_val.needs_move() {
                    if old_val.is_tombstone() {
                        (*old_cell).key.try_free(pin);
                    }
                    continue
                }
                let key = (*old_cell).key.start_move(pin);
                let key = match key {
                    None => continue,
                    Some(key) => key,
                };
                //FIXME handle running out of space (concurrent writes)
                let moved = move_cell(old_cell, new_table, new_shift, key, pin);
                if !moved { unimplemented!() }
            }
        }
        let next = new.next.load(Acquire, pin);
        if let Some(next) = unsafe { next.as_ref() } {
            new.move_contents_to(next, start_loc, pin)
        }
    }

    #[inline(always)]
    fn max_probe_len(&self) -> usize {
        (1 << self.shift) / 8
    }

    #[inline(always)]
    fn capacity(&self) -> usize {
        (1 << self.shift)
    }
}

unsafe fn move_cell<'p, K, V>(
    old_cell: *const Cell<K, V>,
    new_table: *const Cell<K, V>,
    new_shift: u8,
    key: OwnedKey<'p, K>,
    pin: &'p Scope,
) -> bool
where K: PartialEq {
    //FIXME handle running out of space (concurrent writes)
    //      (should not be a problem, we diallow new inserts until the move is done)
    let new_cell = match find_new_cell(new_table, new_shift, key, pin) {
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
    new_table: *const Cell<K, V>,
    new_shift: u8,
    mut key: OwnedKey<'p, K>,
    pin: &'p Scope,
) -> Option<*const Cell<K, V>>
where K: PartialEq {
    let mut new_cell = None;
    'search: for new_off in search_path(key.0, new_shift) {
        let cell = new_table.offset(new_off);
        //FIXME handle key dealloc
        match (*cell).key.try_fill_owned(key, pin) {
            Err((found, k)) => {
                if found.as_ref() == Some(&*k) {
                    new_cell = Some(cell);
                    k.free(pin);
                    break 'search
                }
                key = k
            }
            Ok(..) => {
                new_cell = Some(cell);
                break 'search
            },
        }
    }
    new_cell
}

fn alloc_table<K, V, H>(shift: u8, hasher: H) -> Box<Table<K, V, H>> {
    unsafe {
        let mut vec: Vec<Cell<K, V>> = (0..(1 << shift)).map(|_| Cell::empty()).collect();
        let table = Atomic::from_ptr(Ptr::from_raw(vec.as_mut_ptr()));
        forget(vec);
        Box::new(Table {
            table, shift, hasher, next: Atomic::null(),
        })
    }
}

impl<K, V, H> Drop for Table<K, V, H> {
    fn drop(&mut self) {
        unsafe {
            epoch::unprotected(|pin| {
                let raw = self.table.load(Ordering::Relaxed, pin).as_raw();
                //TODO drop?
                let vec = Vec::from_raw_parts(raw as *mut Cell<K, V>, 0, 1 << self.shift);
                drop(vec)
            })
        }
    }
}

fn search_path(hash: u64, shift: u8) -> SearchPath {
    SearchPath{hash, mask: (1 << shift) - 1}
}

struct SearchPath {
    hash: u64,
    mask: u64,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert() {
        let map = HashMap::with_capacity(32);
        assert_eq!(map.get_and(&1, |&i| i), None);
        assert_eq!(map.get_and(&32, |&i| i), None);
        assert_eq!(map.get_and(&3, |&i| i), None);
        assert_eq!(map.get_and(&2, |&i| i), None);
        map.insert(1, 1);
        assert_eq!(map.get_and(&1, |&i| i), Some(1));
        assert_eq!(map.get_and(&32, |&i| i), None);
        assert_eq!(map.get_and(&3, |&i| i), None);
        assert_eq!(map.get_and(&2, |&i| i), None);
        map.insert(32, 100);
        assert_eq!(map.get_and(&1, |&i| i), Some(1));
        assert_eq!(map.get_and(&32, |&i| i), Some(100));;
        assert_eq!(map.get_and(&3, |&i| i), None);
        assert_eq!(map.get_and(&2, |&i| i), None);
        map.insert(3, 8);
        assert_eq!(map.get_and(&1, |&i| i), Some(1));
        assert_eq!(map.get_and(&32, |&i| i), Some(100));
        assert_eq!(map.get_and(&3, |&i| i), Some(8));
        assert_eq!(map.get_and(&2, |&i| i), None);
        map.insert(1, 20);
        assert_eq!(map.get_and(&1, |&i| i), Some(20));
        assert_eq!(map.get_and(&32, |&i| i), Some(100));
        assert_eq!(map.get_and(&3, |&i| i), Some(8));
        assert_eq!(map.get_and(&2, |&i| i), None);
    }

    #[test]
    fn resize() {
        let map = HashMap::with_capacity(32);
        for i in 0..100 {
            map.insert(i, i);
        }
        for i in 0..100 {
            assert_eq!(map.get_and(&i, |&i| i), Some(i));
        }
        for i in 100..200 {
            assert_eq!(map.get_and(&i, |&i| i), None);
        }
    }
}
