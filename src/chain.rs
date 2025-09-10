use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::BufWriter;
use serde::{Serialize, Deserialize};
use serde_json;

use ordered_float::OrderedFloat;

use crate::{node::Node, Simulation};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Transaction {
    pub id: u32,
    pub from: Option<u32>,
    pub to: u32,
    pub amount: i64,
    pub created_at: OrderedFloat<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {
    pub block_id: u32,
    pub parent_id: Option<u32>,
    pub transactions: Vec<u32>,
    pub timestamp: OrderedFloat<f64>,
    pub block_height: u32,
    pub miner: Option<u32>,
    pub balances: Vec<i64>,
    pub added_to_tree: bool
}

#[derive(Debug)]
pub struct BlockchainTree {
    pub blocks: HashMap<u32, f64>,
    pub children: HashMap<u32, Vec<u32>>,
    pub tip: u32,
    pub genesis: u32,
}


#[derive(Serialize, Deserialize, Clone)]
pub struct TreeNode {
    pub block_id: u32,
    pub parent_id: Option<u32>,
    pub children: Vec<u32>,
    pub block_height: u32,
    pub miner_id: Option<u32>,
    pub num_transactions: usize,
    pub creation_timestamp: f64,  // when block was created by miner
    pub arrival_timestamp: f64,   // when this node received the block
    pub is_tip: bool,
    pub is_genesis: bool,
    pub is_in_longest_chain: bool,
}

#[derive(Serialize, Deserialize)]
pub struct NodeTreeExport {
    pub node_id: u32,
    pub is_fast: bool,
    pub is_high_cpu: bool,
    pub total_blocks: usize,
    pub blocks_created: usize,
    pub self_blocks_in_longest_chain: usize,
    pub longest_chain_length: usize,
    pub num_of_orphaned_blocks: usize,
    pub genesis_block: u32,
    pub current_tip: u32,
    pub tree_nodes: HashMap<u32, TreeNode>,
}

impl BlockchainTree {
    pub fn new() -> Self {
        let mut blocks = HashMap::new();

        blocks.insert(0, 0.0);

        Self {
            blocks,
            children: HashMap::new(),
            tip: 0,
            genesis: 0,
        }
    }

    pub fn export_tree_files(
        &self,
        node: &Node,
        simulation: &Simulation,
        output_dir: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {

        std::fs::create_dir_all(output_dir)?;

        let mut num_of_orphaned_blocks = 0 as usize;
        for (_block_id, tree) in node.orphan_blocks.iter().clone() {
            num_of_orphaned_blocks += tree.blocks.len();
        }

        let tree_export = self.prepare_tree_export(
            node.node_id,
            node.confirmed_blocks.clone(),
            simulation, node.is_fast,
            node.is_high_cpu,
            num_of_orphaned_blocks
        )?;
        self.export_json(&tree_export, output_dir, node.node_id)?;

        Ok(())
    }


    fn prepare_tree_export(
        &self,
        node_id: u32,
        confirmed_blocks: HashSet<u32>,
        simulation: &Simulation,
        is_fast: bool,
        is_high_cpu: bool,
        num_of_orphaned_blocks: usize,
    ) -> Result<NodeTreeExport, Box<dyn std::error::Error>> {
        let mut tree_nodes = HashMap::new();
        let mut blocks_created: usize = 0;
        let mut self_blocks_in_longest_chain: usize = 0;

        for (&block_id, &arrival_time) in &self.blocks {

            if let Some(block) = simulation.blocks.get(&block_id) {

                if let Some(miner) = block.miner {
                    if miner == node_id {
                        blocks_created += 1;
                        if confirmed_blocks.contains(&block_id) {
                            self_blocks_in_longest_chain += 1;
                        }
                    }
                }

                let tree_node = TreeNode {
                    block_id,
                    parent_id: block.parent_id,
                    children: self.children.get(&block_id).cloned().unwrap_or_default(),
                    block_height: block.block_height,
                    miner_id: block.miner,
                    num_transactions: block.transactions.len(),
                    creation_timestamp: block.timestamp.into_inner(),
                    arrival_timestamp: arrival_time,
                    is_tip: block_id == self.tip,
                    is_genesis: block_id == self.genesis,
                    is_in_longest_chain: confirmed_blocks.contains(&block_id),
                };
                tree_nodes.insert(block_id, tree_node);
            }
        }

        Ok(NodeTreeExport {
            node_id,
            is_fast,
            is_high_cpu,
            total_blocks: tree_nodes.len(),
            genesis_block: self.genesis,
            current_tip: self.tip,
            tree_nodes,
            blocks_created,
            self_blocks_in_longest_chain,
            longest_chain_length: confirmed_blocks.len(),
            num_of_orphaned_blocks
        })
    }

    fn export_json(
        &self,
        tree_export: &NodeTreeExport,
        output_dir: &str,
        node_id: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let file_path = format!("{}/node_{}_tree.json", output_dir, node_id);
        let file = File::create(file_path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, tree_export)?;
        Ok(())
    }

}
