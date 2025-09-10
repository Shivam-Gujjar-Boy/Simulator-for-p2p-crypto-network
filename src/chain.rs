use std::collections::HashMap;
use std::fs::File;
use std::io::{Write, BufWriter};
use serde::{Serialize, Deserialize};
use serde_json;

use ordered_float::OrderedFloat;

use crate::Simulation;

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
    pub balances: Vec<i64>
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
}

#[derive(Serialize, Deserialize)]
pub struct NodeTreeExport {
    pub node_id: u32,
    pub total_blocks: usize,
    pub genesis_block: u32,
    pub current_tip: u32,
    pub tree_nodes: HashMap<u32, TreeNode>,
    pub metadata: TreeMetadata,
}

#[derive(Serialize, Deserialize)]
pub struct TreeMetadata {
    pub simulation_end_time: f64,
    pub export_timestamp: String,
    pub node_type: String, // e.g., "honest", "selfish", etc.
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
        node_id: u32,
        simulation: &Simulation,
        output_dir: &str,
        simulation_end_time: f64
    ) -> Result<(), Box<dyn std::error::Error>> {

        std::fs::create_dir_all(output_dir)?;

        let tree_export = self.prepare_tree_export(node_id, simulation, simulation_end_time)?;
        self.export_json(&tree_export, output_dir, node_id)?;
        self.export_csv(&tree_export, output_dir, node_id)?;
        self.export_dot(&tree_export, output_dir, node_id)?;
        self.export_text(&tree_export, output_dir, node_id)?;

        Ok(())
    }


    fn prepare_tree_export(
        &self,
        node_id: u32,
        simulation: &Simulation,
        simulation_end_time: f64
    ) -> Result<NodeTreeExport, Box<dyn std::error::Error>> {
        let mut tree_nodes = HashMap::new();

        for (&block_id, &arrival_time) in &self.blocks {
            if let Some(block) = simulation.blocks.get(&block_id) {
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
                };
                tree_nodes.insert(block_id, tree_node);
            }
        }

        let metadata = TreeMetadata {
            simulation_end_time,
            export_timestamp: chrono::Utc::now().to_rfc3339(),
            node_type: "standard".to_string(), // You can modify this based on node type
        };

        Ok(NodeTreeExport {
            node_id,
            total_blocks: tree_nodes.len(),
            genesis_block: self.genesis,
            current_tip: self.tip,
            tree_nodes,
            metadata,
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

    fn export_csv(
        &self,
        tree_export: &NodeTreeExport,
        output_dir: &str,
        node_id: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let file_path = format!("{}/node_{}_tree.csv", output_dir, node_id);
        let mut file = File::create(file_path)?;
        
        // Write CSV header
        writeln!(
            file,
            "block_id,parent_id,block_height,miner_id,num_transactions,creation_timestamp,arrival_timestamp,delay,is_tip,is_genesis,num_children"
        )?;
        
        // Write data rows
        let mut blocks: Vec<_> = tree_export.tree_nodes.values().collect();
        blocks.sort_by_key(|node| node.block_id);
        
        for node in blocks {
            let delay = node.arrival_timestamp - node.creation_timestamp;
            writeln!(
                file,
                "{},{},{},{},{},{:.6},{:.6},{:.6},{},{},{}",
                node.block_id,
                node.parent_id.map_or("NULL".to_string(), |id| id.to_string()),
                node.block_height,
                node.miner_id.map_or("NULL".to_string(), |id| id.to_string()),
                node.num_transactions,
                node.creation_timestamp,
                node.arrival_timestamp,
                delay,
                node.is_tip,
                node.is_genesis,
                node.children.len()
            )?;
        }
        
        Ok(())
    }

    /// Export to DOT format for graph visualization (Graphviz)
    fn export_dot(
        &self,
        tree_export: &NodeTreeExport,
        output_dir: &str,
        node_id: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let file_path = format!("{}/node_{}_tree.dot", output_dir, node_id);
        let mut file = File::create(file_path)?;
        
        writeln!(file, "digraph BlockchainTree_{} {{", node_id)?;
        writeln!(file, "    rankdir=BT;")?;
        writeln!(file, "    node [shape=box];")?;
        
        // Write nodes with attributes
        for node in tree_export.tree_nodes.values() {
            let color = if node.is_genesis {
                "green"
            } else if node.is_tip {
                "red"
            } else {
                "lightblue"
            };
            
            let delay = node.arrival_timestamp - node.creation_timestamp;
            
            writeln!(
                file,
                "    {} [label=\"Block {}\\nHeight: {}\\nMiner: {}\\nTxns: {}\\nDelay: {:.3}s\" fillcolor={} style=filled];",
                node.block_id,
                node.block_id,
                node.block_height,
                node.miner_id.map_or("?".to_string(), |id| id.to_string()),
                node.num_transactions,
                delay,
                color
            )?;
        }
        
        // Write edges (parent -> child relationships)
        for node in tree_export.tree_nodes.values() {
            if let Some(parent_id) = node.parent_id {
                writeln!(file, "    {} -> {};", parent_id, node.block_id)?;
            }
        }
        
        writeln!(file, "}}")?;
        Ok(())
    }
    
    /// Export to human-readable text format
    fn export_text(
        &self,
        tree_export: &NodeTreeExport,
        output_dir: &str,
        node_id: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let file_path = format!("{}/node_{}_tree.txt", output_dir, node_id);
        let mut file = File::create(file_path)?;
        
        writeln!(file, "BLOCKCHAIN TREE FOR NODE {}", node_id)?;
        writeln!(file, "================================")?;
        writeln!(file, "Total blocks: {}", tree_export.total_blocks)?;
        writeln!(file, "Genesis block: {}", tree_export.genesis_block)?;
        writeln!(file, "Current tip: {}", tree_export.current_tip)?;
        writeln!(file, "Simulation end time: {:.6}s", tree_export.metadata.simulation_end_time)?;
        writeln!(file)?;
        
        // Group blocks by height for better visualization
        let mut blocks_by_height: HashMap<u32, Vec<&TreeNode>> = HashMap::new();
        for node in tree_export.tree_nodes.values() {
            blocks_by_height.entry(node.block_height).or_default().push(node);
        }
        
        let mut heights: Vec<_> = blocks_by_height.keys().cloned().collect();
        heights.sort();
        
        writeln!(file, "BLOCKS BY HEIGHT:")?;
        writeln!(file, "-----------------")?;
        for height in heights {
            writeln!(file, "Height {}:", height)?;
            let mut blocks = blocks_by_height[&height].clone();
            blocks.sort_by_key(|node| node.block_id);
            
            for node in blocks {
                let delay = node.arrival_timestamp - node.creation_timestamp;
                let status = if node.is_genesis {
                    " (GENESIS)"
                } else if node.is_tip {
                    " (TIP)"
                } else {
                    ""
                };
                
                writeln!(
                    file,
                    "  Block {}{}: Parent={}, Miner={}, Txns={}, Created={:.6}s, Arrived={:.6}s, Delay={:.6}s",
                    node.block_id,
                    status,
                    node.parent_id.map_or("None".to_string(), |id| id.to_string()),
                    node.miner_id.map_or("?".to_string(), |id| id.to_string()),
                    node.num_transactions,
                    node.creation_timestamp,
                    node.arrival_timestamp,
                    delay
                )?;
            }
            writeln!(file)?;
        }
        
        Ok(())
    }
}
