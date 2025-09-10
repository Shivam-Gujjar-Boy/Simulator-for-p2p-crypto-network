use std::collections::{HashMap, HashSet};

use crate::chain::BlockchainTree;
use crate::mempool::Mempool;
// use crate::event::Event;

// Node structure
#[derive(Debug)]
pub struct Node {
    pub node_id: u32,
    pub peers: Vec<u32>,
    pub is_fast: bool,
    pub is_high_cpu: bool,
    pub hashing_power_fraction: f64,
    pub blockchain_tree: BlockchainTree,
    pub mempool: Mempool,
    pub orphan_blocks: HashMap<u32, OrphanedTree>, // HashMap<block_id, tree>
    pub confirmed_blocks: HashSet<u32>, // list of blocks in longest chain on this node
    pub seen_transactions: HashMap<u32, f64>, // <tx_id, seen_time>
    pub seen_blocks: HashMap<u32, f64>, // <block_id, seen_time>
}

#[derive(Debug)]
pub struct OrphanedTree {
    pub blocks: HashSet<u32>,
    pub children: HashMap<u32, Vec<u32>>,
    pub tip: u32,
    pub root: u32,
}

impl Node {
    pub fn new(id: usize, is_fast: bool, is_high_cpu: bool, hashing_power_fraction: f64, peers: Vec<u32>) -> Self {
        let mut confirmed_blocks: HashSet<u32> = HashSet::new();
        confirmed_blocks.insert(0);
        Self {
            node_id: id as u32,
            peers,
            is_fast,
            is_high_cpu,
            hashing_power_fraction,
            blockchain_tree: BlockchainTree::new(),
            mempool: Mempool::new(),
            seen_transactions: HashMap::new(),
            seen_blocks: HashMap::new(),
            orphan_blocks: HashMap::new(),
            confirmed_blocks,
        }
    }

}
