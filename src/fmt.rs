use std::fmt::{self, Formatter, Result};
use std::slice;
use std::sync::atomic::Ordering;

use coco::epoch;

use super::{HashMap, Table, Cell, KeyPtr, ValPtr};

impl<K, V, H> fmt::Debug for HashMap<K, V, H>
where K: fmt::Debug, V: fmt::Debug {
    fn fmt(&self, f: &mut Formatter) -> Result {
        epoch::pin(|pin| unsafe {
            f.debug_tuple("HashMap")
                .field(self.ptr.load(Ordering::Acquire, pin).as_ref().unwrap())
                .finish()
        })
    }
}

impl<K, V, H> fmt::Debug for Table<K, V, H>
where K: fmt::Debug, V: fmt::Debug {
    fn fmt(&self, f: &mut Formatter) -> Result {
        epoch::pin(|pin| unsafe {
            let mut debug = f.debug_map();
            let ptr = self.table.load(Ordering::Acquire, pin);
            for cell in slice::from_raw_parts(ptr.as_raw(), self.capacity()) {
                if let Some(key) = cell.key.get(pin) {
                // if let Some(key) = cell.key.hash_and_key(pin) {
                    debug.entry(key, &cell.val.get(pin));
                }

            }
            debug.finish()
        })
    }
}

impl<K, V> fmt::Debug for Cell<K, V>
where K: fmt::Debug, V: fmt::Debug {
    fn fmt(&self, f: &mut Formatter) -> Result {
        epoch::pin(|pin|{
            f.debug_struct("ValBox")
                .field("key", &self.key.get(pin))
                .field("val", &self.val.get(pin))
                .finish()
        })
    }
}

impl<K> fmt::Debug for KeyPtr<K>
where K: fmt::Debug {
    fn fmt(&self, f: &mut Formatter) -> Result {
        epoch::pin(|pin|{
            f.debug_tuple("ValBox")
                .field(&self.get(pin))
                .finish()
        })
    }
}

impl<V> fmt::Debug for ValPtr<V>
where V: fmt::Debug {
    fn fmt(&self, f: &mut Formatter) -> Result {
        epoch::pin(|pin|{
            f.debug_tuple("ValBox")
                .field(&self.get(pin))
                .finish()
        })
    }
}