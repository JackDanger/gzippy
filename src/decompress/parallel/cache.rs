#![allow(dead_code)] // vendor-faithful rapidgzip port; many items are pending consumer-port

//! Literal port of `rapidgzip::Cache` + `LeastRecentlyUsed` strategy
//! (vendor/.../core/Cache.hpp).
//!
//! LRU-evicting key/value map with hit/miss/unused-entry statistics.
//! `BlockFetcher` uses two instances (a main cache + a prefetch cache);
//! both share this implementation.

use std::collections::{BTreeMap, HashMap};
use std::hash::Hash;

/// Per-cache statistics. Mirror of `Cache::Statistics` (Cache.hpp:125-132).
#[derive(Debug, Clone, Copy, Default)]
pub struct CacheStatistics {
    pub hits: usize,
    pub misses: usize,
    pub unused_entries: usize,
    pub capacity: usize,
    pub max_size: usize,
}

/// Trait mirroring `CacheStrategy::CacheStrategy<Index>`
/// (Cache.hpp:17-44).
pub trait CacheStrategy<Index: Hash + Eq + Clone + Ord> {
    fn touch(&mut self, index: Index);
    fn next_eviction(&self) -> Option<Index>;
    fn next_nth_eviction(&self, count_to_emplace_hypothetically: usize) -> Option<Index>;
    fn evict(&mut self, index_to_evict: Option<Index>) -> Option<Index>;
}

/// Least-recently-used eviction. Mirror of
/// `CacheStrategy::LeastRecentlyUsed` (Cache.hpp:47-109).
///
/// Maintains two structures:
/// - `last_usage: HashMap<Index, Nonce>` — fast insert + lookup.
/// - `sorted_indexes: BTreeMap<Nonce, Index>` — keys (nonces) sorted
///   ascending, so `.iter().next()` is the LRU entry.
pub struct LeastRecentlyUsed<Index: Hash + Eq + Clone + Ord> {
    last_usage: HashMap<Index, u64>,
    sorted_indexes: BTreeMap<u64, Index>,
    usage_nonce: u64,
}

impl<Index: Hash + Eq + Clone + Ord> Default for LeastRecentlyUsed<Index> {
    fn default() -> Self {
        Self {
            last_usage: HashMap::new(),
            sorted_indexes: BTreeMap::new(),
            usage_nonce: 0,
        }
    }
}

impl<Index: Hash + Eq + Clone + Ord> CacheStrategy<Index> for LeastRecentlyUsed<Index> {
    fn touch(&mut self, index: Index) {
        self.usage_nonce += 1;
        if let Some(prev_nonce) = self.last_usage.get(&index).copied() {
            self.sorted_indexes.remove(&prev_nonce);
        }
        self.last_usage.insert(index.clone(), self.usage_nonce);
        self.sorted_indexes.insert(self.usage_nonce, index);
    }

    fn next_eviction(&self) -> Option<Index> {
        self.sorted_indexes.iter().next().map(|(_, v)| v.clone())
    }

    fn next_nth_eviction(&self, count_to_emplace_hypothetically: usize) -> Option<Index> {
        if count_to_emplace_hypothetically == 0
            || count_to_emplace_hypothetically > self.sorted_indexes.len()
        {
            return None;
        }
        self.sorted_indexes
            .iter()
            .nth(count_to_emplace_hypothetically - 1)
            .map(|(_, v)| v.clone())
    }

    fn evict(&mut self, index_to_evict: Option<Index>) -> Option<Index> {
        let evicted = index_to_evict.or_else(|| self.next_eviction());
        if let Some(ref idx) = evicted {
            if let Some(prev_nonce) = self.last_usage.remove(idx) {
                self.sorted_indexes.remove(&prev_nonce);
            }
        }
        evicted
    }
}

/// Generic key/value cache with pluggable eviction strategy. Mirror of
/// `rapidgzip::Cache<Key, Value, CacheStrategy>` (Cache.hpp:117-296).
pub struct Cache<
    Key: Hash + Eq + Clone + Ord,
    Value,
    Strategy: CacheStrategy<Key> = LeastRecentlyUsed<Key>,
> {
    strategy: Strategy,
    max_cache_size: usize,
    cache: HashMap<Key, Value>,
    statistics: CacheStatistics,
    accesses: HashMap<Key, usize>,
}

impl<Key: Hash + Eq + Clone + Ord, Value> Cache<Key, Value, LeastRecentlyUsed<Key>> {
    pub fn new(max_cache_size: usize) -> Self {
        Self {
            strategy: LeastRecentlyUsed::default(),
            max_cache_size,
            cache: HashMap::new(),
            statistics: CacheStatistics::default(),
            accesses: HashMap::new(),
        }
    }
}

impl<Key: Hash + Eq + Clone + Ord, Value: Clone, Strategy: CacheStrategy<Key>>
    Cache<Key, Value, Strategy>
{
    /// Get a value by key. Increments the hit/miss counter accordingly.
    pub fn get(&mut self, key: &Key) -> Option<Value> {
        if let Some(v) = self.cache.get(key).cloned() {
            self.statistics.hits += 1;
            *self.accesses.entry(key.clone()).or_insert(0) += 1;
            self.strategy.touch(key.clone());
            return Some(v);
        }
        self.statistics.misses += 1;
        None
    }

    pub fn insert(&mut self, key: Key, value: Value) {
        if self.capacity() == 0 {
            return;
        }
        if self.cache.contains_key(&key) {
            self.cache.insert(key.clone(), value);
        } else {
            self.shrink_to(self.capacity() - 1);
            self.cache.insert(key.clone(), value);
            self.statistics.max_size = self.statistics.max_size.max(self.cache.len());
        }
        self.accesses.entry(key.clone()).or_insert(0);
        self.strategy.touch(key);
    }

    pub fn touch(&mut self, key: &Key) {
        if self.test(key) {
            self.strategy.touch(key.clone());
        }
    }

    pub fn test(&self, key: &Key) -> bool {
        self.cache.contains_key(key)
    }

    pub fn clear(&mut self) {
        self.cache.clear();
    }

    pub fn evict(&mut self, key: &Key) {
        self.strategy.evict(Some(key.clone()));
        self.cache.remove(key);
    }

    /// Remove `key` and return its value without counting a cache hit.
    /// Used when resolve-ahead post-process takes sole ownership of a
    /// prefetched entry (consumer must not re-`get` the same Arc).
    pub fn take(&mut self, key: &Key) -> Option<Value> {
        let v = self.cache.remove(key)?;
        self.strategy.evict(Some(key.clone()));
        self.accesses.remove(key);
        Some(v)
    }

    pub fn next_eviction(&self, key: Option<Key>) -> Option<Key> {
        if self.cache.len() < self.capacity()
            || (key.as_ref().is_some_and(|k| !self.cache.contains_key(k)))
        {
            return None;
        }
        self.strategy.next_eviction()
    }

    pub fn next_nth_eviction(&self, count_to_be_inserted: usize) -> Option<Key> {
        let free_capacity = self.capacity().saturating_sub(self.cache.len());
        if count_to_be_inserted <= free_capacity {
            return None;
        }
        self.strategy
            .next_nth_eviction(count_to_be_inserted - free_capacity)
    }

    pub fn shrink_to(&mut self, new_size: usize) {
        while self.cache.len() > new_size {
            let to_evict = self.strategy.evict(None);
            if let Some(key_to_evict) = to_evict {
                self.cache.remove(&key_to_evict);
                if let Some(access_count) = self.accesses.remove(&key_to_evict) {
                    if access_count == 0 {
                        self.statistics.unused_entries += 1;
                    }
                }
            } else {
                break;
            }
        }
    }

    pub fn statistics(&self) -> CacheStatistics {
        let mut stats = self.statistics;
        stats.capacity = self.capacity();
        stats
    }

    pub fn reset_statistics(&mut self) {
        self.statistics = CacheStatistics::default();
        self.accesses.clear();
    }

    pub fn capacity(&self) -> usize {
        self.max_cache_size
    }

    pub fn size(&self) -> usize {
        self.cache.len()
    }

    /// Snapshot of `(key, value)` pairs currently held, WITHOUT touching
    /// the LRU strategy or hit/miss stats. Mirror of vendor's
    /// `prefetchCache().contents()` (used by
    /// `GzipChunkFetcher::queuePrefetchedChunkPostProcessing`,
    /// GzipChunkFetcher.hpp:524). Caller is expected to sort by key.
    #[cfg_attr(not(parallel_sm), allow(dead_code))]
    pub fn contents_snapshot(&self) -> Vec<(Key, Value)> {
        self.cache
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Current keys without cloning values.
    pub fn keys(&self) -> Vec<Key> {
        self.cache.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hit_miss_counted() {
        let mut c = Cache::<u64, String>::new(2);
        c.insert(1, "one".into());
        c.insert(2, "two".into());
        assert_eq!(c.get(&1), Some("one".into()));
        assert_eq!(c.get(&3), None);
        let s = c.statistics();
        assert_eq!(s.hits, 1);
        assert_eq!(s.misses, 1);
    }

    #[test]
    fn lru_eviction_drops_least_recently_used() {
        let mut c = Cache::<u64, String>::new(2);
        c.insert(1, "one".into());
        c.insert(2, "two".into());
        // Touch key 1 → makes 2 the LRU.
        let _ = c.get(&1);
        c.insert(3, "three".into());
        // Key 2 should be evicted.
        assert_eq!(c.get(&2), None);
        assert_eq!(c.get(&1), Some("one".into()));
        assert_eq!(c.get(&3), Some("three".into()));
    }

    #[test]
    fn unused_entries_counted_on_eviction() {
        let mut c = Cache::<u64, String>::new(1);
        c.insert(1, "one".into());
        c.insert(2, "two".into()); // evicts 1 without access → unused
        let s = c.statistics();
        assert_eq!(s.unused_entries, 1);
    }

    #[test]
    fn test_and_touch_do_not_count_as_hit_miss() {
        let mut c = Cache::<u64, String>::new(2);
        c.insert(1, "one".into());
        assert!(c.test(&1));
        c.touch(&1);
        let s = c.statistics();
        assert_eq!(s.hits, 0);
        assert_eq!(s.misses, 0);
    }

    #[test]
    fn capacity_zero_is_noop_insert() {
        let mut c = Cache::<u64, String>::new(0);
        c.insert(1, "one".into());
        assert_eq!(c.size(), 0);
        assert_eq!(c.get(&1), None);
    }

    #[test]
    fn next_nth_eviction_returns_nth_oldest() {
        let mut c = Cache::<u64, String>::new(3);
        c.insert(1, "one".into());
        c.insert(2, "two".into());
        c.insert(3, "three".into());
        // Cache is at capacity. Inserting 2 more would evict 2 oldest.
        // next_nth_eviction(2) → returns the 2nd-oldest = key 2.
        assert_eq!(c.next_nth_eviction(2), Some(2));
    }
}
