use std::collections::{HashMap, HashSet};

use crate::chain::BlockchainTree;
use crate::mempool::Mempool;
// use crate::event::Event;

#[derive(Debug)]
pub struct Node {
    pub node_id: u32,
    pub peers: Vec<u32>,
    pub is_fast: bool,
    pub is_high_cpu: bool,
    pub hashing_power_fraction: f64,
    pub blockchain_tree: BlockchainTree,
    pub mempool: Mempool,
    pub orphan_blocks: HashMap<u32, u32>, // HashMap<block_id, parent_id>
    pub confirmed_blocks: HashSet<u32>,
    pub seen_transactions: HashMap<u32, f64>,
    pub seen_blocks: HashMap<u32, f64>,
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

    // pub fn receive_block(&mut self, block_id: usize, time: SimTime, network: &mut Network) {
    //     self.blockchain_tree.add_block(block_id);
    //     // Example: schedule mining event
    //     let mine_event = Event::new_mine_block(self.id, time + 10);
    //     network.schedule_event(mine_event);
    // }

    // pub fn mine_block(&mut self, block_id: usize, time: SimTime, network: &mut Network) {
    //     self.blockchain_tree.add_block(block_id);
    //     // Example: broadcast block to network
    //     let broadcast_event = Event::new_broadcast_block(self.id, block_id, time + 1);
    //     network.schedule_event(broadcast_event);
    // }
}
