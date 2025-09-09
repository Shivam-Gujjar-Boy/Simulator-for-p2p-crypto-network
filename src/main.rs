mod config;
mod event;
mod scheduler;
mod chain;
mod mempool;
mod node;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use std::{collections::HashSet, io};

use ordered_float::OrderedFloat;
use rand::{seq::SliceRandom, thread_rng, Rng};

use crate::chain::{Block, Transaction};
use crate::node::{Node, OrphanedTree};

use crate::config::Config;
use crate::event::{Event, EventType};
use crate::scheduler::Scheduler;

// Global Simulation Struct that stores the state of the whole network
#[derive(Debug)]
pub struct Simulation {
    pub cfg: Config, // simulation parameters
    pub nodes: Vec<node::Node>, // nodes
    pub scheduler: Scheduler, // contains the min-heap (event queue)
    pub transactions: HashMap<u32, Transaction>, // so that no 2 txns have same TxID
    pub blocks: HashMap<u32, Block>, // so that no 2 blocks have same BlkID
}


// Get current system time
fn current_time_millis() -> f64 {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards");
    now.as_micros() as f64 / 1000.0
}


fn select_random_numbers(n: u32, m: u32, element_to_skip: Option<u32>) -> Vec<u32> {
    if m > n {
        panic!("Cannot select more numbers than available in the range");
    }

    let mut numbers: Vec<u32> = (0..n)
        .filter(|&x| Some(x) != element_to_skip)
        .collect();

    if m > numbers.len() as u32 {
        panic!("Cannot select more numbers than available after skipping");
    }

    let mut rng = thread_rng();
    numbers.shuffle(&mut rng);
    numbers.truncate(m as usize);

    numbers
}

// Generate a connected topology
fn generate_connected_topology(n: usize) -> Vec<Vec<u32>> {
    loop {
        let mut topology: Vec<HashSet<u32>> = vec![HashSet::new(); n];
        let mut rng = rand::thread_rng();

        while topology.iter().any(|nbrs| nbrs.len() < 3) {
            let u = rng.gen_range(0..n);
            let v= rng.gen_range(0..n);

            if u != v && topology[u].len() < 6 && topology[v].len() < 6 {
                topology[u].insert(v as u32);
                topology[v].insert(u as u32);
            }
        }

        let adj: Vec<Vec<u32>> = topology
            .into_iter()
            .map(|set| set.into_iter().collect())
            .collect();

        if is_connected(&adj, n) {
            return adj;
        }
    }
}

// check is topology is connected
fn is_connected(topology: &Vec<Vec<u32>>, n: usize) -> bool {
    let mut visited = vec![false; n];
    let mut stack = vec![0];
    visited[0] = true;

    while let Some(node) = stack.pop() {
        for &neighbor in &topology[node] {
            if !visited[neighbor as usize] {
                visited[neighbor as usize] = true;
                stack.push(neighbor as usize);
            }
        }
    }

    visited.into_iter().all(|v| v)
}

// Generate random time using an exponential distribution with a given mean time
fn sample_exponential(mean_ms: f64) -> f64 {
    let mut rng = rand::thread_rng();
    let u: f64 = rng.gen::<f64>();
    -mean_ms * u.ln()
}



fn main() {
    // Helper function to read input from terminal
    fn read_input<T: std::str::FromStr>(prompt: &str) -> T {
        loop {
            println!("{}", prompt);
            let mut input = String::new();
            io::stdin().read_line(&mut input).expect("Failed to read line");

            match input.trim().parse::<T>() {
                Ok(value) => return value,
                Err(_) => {
                    println!("Invalid input, please try again.");
                    continue;
                }
            }
        }
    }

    // let n: u32 = read_input("Enter number of nodes (positive integer):");
    // let z0: u32 = read_input("Enter percentage of slow nodes (0–100):");
    // let z1: u32 = read_input("Enter percentage of low CPU nodes (0–100):");
    let ttx: f64 = read_input("Enter mean time for transaction generation in milliseconds (float):");
    let i: f64 = read_input("Enter average interarrival time between 2 blocks in milliseconds (float):");
    let t_total: f64 = read_input("Enter the total time for simulation to run:");
    // let initiate_mining_delay: f64 = read_input("Enter the amount of time to wait to initiate mining after start of simulation, in milliseconds:");

    let n: u32 = 7;
    let z0: u32 = 90;
    let z1: u32 = 90;
    // let ttx: f64 = 1000.0;
    // let i: f64 = 3000.0;

    // println!("\n--- Input Summary ---");
    // println!("Number of nodes: {}", n);
    // println!("Percentage of slow nodes: {}%", z0);
    // println!("Percentage of low CPU nodes: {}%", z1);
    // println!("Mean transaction generation time (Ttx) milliseconds: {}", ttx);
    // println!("Average interarrival time (I): {} milliseconds", i);
    // println!("Total Simulation time (T_total): {} milliseconds", t_total);
    // println!("Waiting Time for Initial Mining Attempts: {} milliseconds", initiate_mining_delay);


    println!("Generating Topology...");

    // Generate Topology (There should be atleast 7 nodes in the network topology)
    let topology: Vec<Vec<u32>> = generate_connected_topology(n as usize);

    println!("Connected Topology created successfully!");



    println!("Generating Nodes...");

    // Generate Nodes
    let mut nodes: Vec<node::Node> = vec![];

    let num_of_slow_nodes = (n as f64 * (z0 as f64 / 100.0)) as u32;
    // let num_of_fast_nodes = n - num_of_slow_nodes;

    let num_of_low_cpu_nodes = (n as f64 * (z1 as f64 / 100.0)) as u32;
    let num_of_high_cpu_nodes = n - num_of_low_cpu_nodes;
    let low_cpu_hashing_power = 1.0 / (num_of_low_cpu_nodes as f64 + 10.0 * num_of_high_cpu_nodes as f64);
    let high_cpu_hashing_power = 10.0 * low_cpu_hashing_power;

    let slow_node_indices: Vec<u32> = select_random_numbers(n, num_of_slow_nodes, None); // O(n)
    let low_cpu_node_indices: Vec<u32> = select_random_numbers(n, num_of_low_cpu_nodes, None); // O(n)
    // println!("{:?}", slow_node_indices);
    // println!("{:?}", low_cpu_node_indices);

    let mut nodes_propoerties: Vec<(bool, bool)> = vec![(true, true); n as usize];

    for i in slow_node_indices {
        nodes_propoerties[i as usize].0 = false;
    } // O(n)

    for i in low_cpu_node_indices {
        nodes_propoerties[i as usize].1 = false;
    } // O(n)

    // println!("Nodes Properties: {:?}", nodes_propoerties);


    for i in 0..n {
        let mut hashing_power_fraction = low_cpu_hashing_power;
        if nodes_propoerties[i as usize].1 {
            hashing_power_fraction = high_cpu_hashing_power;
        }
        
        let node = Node::new(
            i as usize,
            nodes_propoerties[i as usize].0,
            nodes_propoerties[i as usize].1,
            hashing_power_fraction,
            topology[i as usize].clone(),
        );

        nodes.push(node);
    }

    println!("Nodes Generated according to the Topology!");

    // Calculate initial parameters
    let mut rng = rand::thread_rng();
    let positive_min_latency: f64 = rng.gen_range(10..=500) as f64;


    // Create the Simulation Object
    let config = Config::new(n, t_total, ttx, positive_min_latency, i);
    let mut blocks = HashMap::new();
    blocks.insert(0, Block {
        block_height: 1,
        block_id: 0,
        parent_id: None,
        transactions: vec![],
        timestamp: OrderedFloat(0.0),
        miner: None,
        balances: vec![0i64; n as usize]
    });

    let mut simulation = Simulation {
        cfg: config,
        nodes,
        scheduler: Scheduler::new(),
        transactions: HashMap::new(),
        blocks
    };


    // Schedule GenerateTx Event for each Node in beginning
    for i in 0..n {
        // Find a random peer to send coins to
        let js = select_random_numbers(n, 1, Some(i));
        let mut j = i;
        if js.len() != 0 {
            j = js[0];
        }

        // Schedule GenerateTx Event
        let transaction = Transaction {
            id: simulation.transactions.len() as u32,
            from: Some(i),
            to: j,
            amount: 0,
            created_at: OrderedFloat(0.0),
        };

        let event = Event {
            node_id: i,
            event_type: EventType::GenerateTransaction { transaction_id: transaction.id }
        };

        simulation.scheduler.schedule(event, OrderedFloat(0.0));
        

        simulation.transactions.insert(transaction.id, transaction);
    }

    // Schedule MineBlock Event for each node in beginning
    for i in 0..n {
        // Simulate PoW. Calculate Tk. Assuming no transactions at that time, all nodes are just mining empty blocks on top of genesis block
        let mean_ms = simulation.cfg.mine_interval_ms / simulation.nodes[i as usize].hashing_power_fraction;
        let t_k = sample_exponential(mean_ms);

        // Create Empty Block to Mine on top of Genesis Block
        let block = Block {
            block_id: simulation.blocks.len() as u32,
            parent_id: Some(0),
            transactions: vec![],
            timestamp: OrderedFloat(t_k),
            block_height: 1,
            miner: Some(i),
            balances: vec![0i64; n as usize]
        };

        // Create MineBlock Event at t = Tk
        let event = Event {
            node_id: i,
            event_type: EventType::MineBlock { block_id: block.block_id }
        };

        simulation.scheduler.schedule(event, OrderedFloat(t_k));

        simulation.blocks.insert(block.block_id, block);
    }


    // Start Simulation

    let simulation_start_time = current_time_millis();

    // Run simulation till the event queue goes empty or the required simulation exceedes
    while let Some((event, time)) = simulation.scheduler.next_event() {
        if (current_time_millis() - simulation_start_time) > t_total {
            println!("Simulation Complete!!");
            break;
        }

        simulation.handle_event(event, time);

    }

    // println!("Balances on Each Node =>");

    // simulation.nodes.iter().for_each(|node| {
    //     println!("Node {} -> {:?}", node.node_id, node.balances);
    // });

    println!("Number of transactions created: {}", simulation.transactions.len());
    println!("Number of Blocks created: {}", simulation.blocks.len());
    for i in 0..n {
        println!("Balances: {:?}", simulation.blocks[&simulation.nodes[i as usize].blockchain_tree.tip].balances);
    }


}



impl Simulation {

    // Handle All Events Conditionally
    fn handle_event(&mut self, event: Event, time: f64) {

        match event.event_type {
            EventType::GenerateTransaction { transaction_id } => {
                // println!("Transaction Generated: {}", transaction.id);
                self.handle_generate_transaction(event.node_id, transaction_id, time);
            }

            EventType::ReceiveTransaction { transaction_id, sender } => {
                // println!("Transaction Received: {}", transaction.id);
                self.handle_receive_transaction(event.node_id, transaction_id, time, sender);
            }

            EventType::MineBlock { block_id } => {
                // println!("Block Mined: {}, with {} transactions", block.block_id, block.transactions.len());
                self.handle_mine_block(event.node_id, block_id, time);
            }

            EventType::ReceiveBlock { block_id, sender } => {
                // println!("Block Received: {}", block.block_id);
                self.handle_receive_block(event.node_id, block_id, time, sender);
            }
        }

    }


    // EVENT TYPE 1 -> Handle Generate Transaction Event Execution
    fn handle_generate_transaction(&mut self, node_id: u32, transaction_id: u32, time: f64) {
        let transaction = self.transactions[&transaction_id].clone();

        if let Some(node) = self.nodes.get_mut(node_id as usize) {
            // Verify Transaction
            let sender_balance = self.blocks[&node.blockchain_tree.tip].balances.get(node_id as usize).copied().unwrap_or(0);

            if transaction.amount <= sender_balance {
                // Add txn to mempool
                node.mempool.add_transaction(transaction.id);

                // Add txnID to seen transactions hashset
                node.seen_transactions.insert(transaction.id, time);

                // Schedule Receive Txn Events for neighbor nodes
                let peer_ids: Vec<u32> = self.nodes[node_id as usize].peers.iter().copied().collect();
                for peer_id in peer_ids {
                    self.schedule_receive_tx_event(node_id, peer_id, 1024, time, transaction.id);
                }
            }

            // Now Schedule one more Generate Transaction Event on this node
            self.schedule_generate_tx_event(node_id, time);


        } else {
            eprintln!("Invalid node_id {} in handle_generate_transaction", node_id);
        }

    }


    // EVENT TYPE 2 -> Handle Receive Transaction Event Execution
    fn handle_receive_transaction(&mut self, node_id: u32, transaction_id: u32, time: f64, sender: u32) {
        let transaction = self.transactions[&transaction_id].clone();
        if let Some(node) = self.nodes.get_mut(node_id as usize) {
            // When a transaction is received, no need to verify it
            // Check if this node has already seen this transaction or not
            if !node.seen_transactions.iter().any(|(tx_id, _received_time)| *tx_id == transaction.id) {
                // Mark the transaction as seen
                node.seen_transactions.insert(transaction.id, time);

                // Add transaction to mempool
                node.mempool.add_transaction(transaction.id);

                // Schedule Receive Txn Events for neighbor nodes, excluding the one it received tx from
                let peer_ids: Vec<u32> = self.nodes[node_id as usize]
                    .peers
                    .iter()
                    .copied()
                    .filter(|&peer_id| peer_id != sender)
                    .collect();
                for peer_id in peer_ids {
                    self.schedule_receive_tx_event(node_id, peer_id, 1024, time, transaction.id);
                }

            }
        } else {
            eprintln!("Invalid node_id {} in handle_receive_transaction", node_id);
        }
    }


    // EVENT TYPE 3 -> Handle Mine Block Event Execution
    fn handle_mine_block(&mut self, node_id: u32, block_id: u32, time: f64) {
        // let mut modified_block = self.blocks[&block_id].clone();
        let mut peer_ids = vec![];
        let mut next_block: Option<Block> = None;
        let mut valid_block = false;

        {
            if let Some(node) = self.nodes.get_mut(node_id as usize) {

                // Case 1: Block builds on current tip
                if self.blocks[&block_id].parent_id == Some(node.blockchain_tree.tip) {

                    // iterate through all transactions in the block, modify balances vector in node.balances
                    // After that check is someone's balance is negative, if not, then valid_block = true, else jump out of overall if
                    // Do add 500 coins to node_id, as coinbase txn gives 500 reward
                    let parent_id = self.blocks[&block_id].parent_id;
                    if let Some(_parent_id) = parent_id {
                        // let parent_balances = self.blocks[&parent_id].balances.clone();
                        let mut balances = self.blocks[&block_id].balances.clone();
                        valid_block = true;

                        // for tx_id in &self.blocks[&block_id].transactions {
                        //     if let Some(from) = self.transactions[tx_id].from {
                        //         if let Some(balance) = balances.get_mut(from as usize) {
                        //             *balance -= self.transactions[tx_id].amount;
                        //         } else {
                        //             valid_block = false;
                        //             break;
                        //         }
                        //     }

                        //     balances[self.transactions[tx_id].to as usize] += self.transactions[tx_id].amount;

                        //     if balances.iter().any(|balance| *balance < (0 as i64)) {
                        //         valid_block = false;
                        //         break;
                        //     }
                        // }

                        if valid_block {
                            // Add coinbase transaction
                            let coinbase_tx = Transaction {
                                id: self.transactions.len() as u32,
                                from: None,
                                to: node_id,
                                amount: 500,
                                created_at: OrderedFloat(time)
                            };
                            if let Some(block) = self.blocks.get_mut(&block_id) {
                                block.transactions.insert(0, coinbase_tx.id);
                            } else {
                                panic!("Block with id {} not found", block_id);
                            }
                            self.transactions.insert(coinbase_tx.id, coinbase_tx);

                            balances[node_id as usize] += 500;
                            // self.blocks[&block_id].balances = balances;
                            if let Some(block) = self.blocks.get_mut(&block_id) {
                                block.balances = balances;
                            } else {
                                panic!("Block with id {} not found", block_id);
                            }

                            // Add block to blockchain tree
                            let block_id = self.blocks[&block_id].block_id;
                            node.blockchain_tree.blocks.insert(block_id, time);
                            node.blockchain_tree.children
                                .entry(self.blocks[&block_id].parent_id.unwrap())
                                .or_default()
                                .push(block_id);
                            node.blockchain_tree.tip = block_id;

                            // Store peer ids for broadcasting
                            peer_ids = node.peers.iter().copied().collect();

                            // Add this block_id to confirmed blocks
                            node.confirmed_blocks.insert(block_id);
                        }

                    }

                } else {
                    // Tip moved while mining this block.
                    // DUMP this block's transaction back to mmempool ONLY IF:
                    //  - they are not already in mempool
                    //  - they are not in any confirmed block on the current longest chain

                    // println!("Mine failed: {}", block.block_id);

                    let confirmed_ids = node.confirmed_blocks.clone();

                    // for tx in self.blocks[&block_id].transactions.into_iter() {
                    //     if !confirmed_ids.contains(&tx) && !node.mempool.transactions.contains(&tx) {
                    //         node.mempool.transactions.push(tx);
                    //     }
                    // }

                    if let Some(block) = self.blocks.get(&block_id) {
                        for tx in block.transactions.iter() {
                            if !confirmed_ids.contains(tx) && !node.mempool.transactions.contains(tx) {
                                node.mempool.transactions.push(*tx);
                            }
                        }
                    } else {
                        eprintln!("Block with id {} not found", block_id);
                    }

                }

                // In both cases, we schedule next mine block event
                // For that select a set of transactions to be included in the block
                let mut balances = self.blocks[&node.blockchain_tree.tip].balances.clone();
                // let mut balance: Vec<i64> = vec![0i64; self.cfg.num_nodes as usize];

                let mut selected_txns: Vec<u32> = Vec::new();

                for tx in node.mempool.transactions.iter() {
                    if selected_txns.len() >= 1023 {
                        break;
                    }

                    if let Some(from) = self.transactions[tx].from {
                        if balances[from as usize] >= self.transactions[tx].amount {
                            selected_txns.push(*tx);
                            balances[from as usize] -= self.transactions[tx].amount;
                            balances[self.transactions[tx].to as usize] += self.transactions[tx].amount;
                        }
                    }
                }

                let wait_time = sample_exponential(self.cfg.mine_interval_ms);
                next_block = Some(Block {
                    block_id: self.blocks.len() as u32,
                    parent_id: Some(node.blockchain_tree.tip),
                    transactions: selected_txns,
                    timestamp: OrderedFloat(time + wait_time),
                    block_height: self.blocks[&node.blockchain_tree.tip].block_height + 1,
                    miner: Some(node_id),
                    balances
                });
            }
        }

        // Case 1: broadcast only if block was valid
        if valid_block {
            for peer_id in peer_ids {
                let block_size = 1024 * self.blocks[&block_id].transactions.len() as u64;
                let latency = self.simulate_latency(block_size, node_id, peer_id);
                self.schedule_receive_block_event(node_id, peer_id, block_id, time + latency);
            }
        }

        // Case 1 + Case 2: always schedule next mining
        if let Some(next_block) = next_block {
            let wait_time = next_block.timestamp.into_inner() - time;
            let event = Event {
                node_id,
                event_type: EventType::MineBlock { block_id: next_block.block_id },
            };
            self.scheduler.schedule(event, OrderedFloat(time + wait_time));
            self.blocks.insert(next_block.block_id, next_block);
        }

    }



    // EVENT TYPE 4 -> Handle Receive Block Event Execution
    fn handle_receive_block(&mut self, node_id: u32, block_id: u32, time: f64, sender: u32) {
        // STEPS TO BE FOLLOWED CONDITIONALLY =>
        //
        // Check is already seen the block, discard if yes
        // Else, add to seen blocks, and verify the block
        // If verified, then check if parent exists, If not then add to orphaned area
        // Else add to main chain and make relevant changes in the confirmed chain and mempool

        let mut peer_ids = vec![];
        let mut valid_block = false;

        if let Some(node) = self.nodes.get_mut(node_id as usize) {
            // return if this node already received this block, if not then add block to seen blocks map
            if node.seen_blocks.contains_key(&block_id) {
                return;
            } else {
                node.seen_blocks.insert(block_id, time);
            }

            if let Some(parent_id) = self.blocks[&block_id].parent_id {
                // Verify the block
                for balance in self.blocks[&block_id].balances.clone() {
                    if balance < 0 {
                        return;
                    }
                }

                valid_block = true;

                // If blockchain tree has the parent block
                if node.blockchain_tree.blocks.contains_key(&parent_id) {
                    // Add the transactions to seen if this node hasn't already seen any transaction of this block
                    for tx in self.blocks[&block_id].transactions.clone() {
                        if !node.seen_transactions.contains_key(&tx) {
                            node.seen_transactions.insert(tx, time);
                        }
                    }

                    // Check if this block has some children in orphaned blocks
                    if let Some(orphan_tree_index) = node.orphan_blocks.iter().position(|(_blk, tree)| tree.blocks.contains(&block_id)) {
                        // Case 1: Parent is present and some children are in orphaned blocks
                        // Remove the orphaned tree that contains children of this block
                        let orphaned_tree_maybe = node.orphan_blocks.remove(&(orphan_tree_index as u32));
                        
                        // Add current block to blockchain tree first
                        node.blockchain_tree.blocks.insert(block_id, time);
                        node.blockchain_tree.children
                            .entry(parent_id)
                            .or_default()
                            .push(block_id);
                        

                        
                        if let Some(orphaned_tree) = orphaned_tree_maybe {
                            // Now add all blocks from orphaned tree to blockchain tree in correct order
                            // We need to do a topological sort starting from the root of orphaned tree
                            let mut blocks_to_add = Vec::new();
                            let mut queue = vec![orphaned_tree.root];

                            while let Some(current_block) = queue.pop() {
                                blocks_to_add.push(current_block);
                                if let Some(children) = orphaned_tree.children.get(&current_block) {
                                    for &child in children {
                                        queue.push(child);
                                    }
                                }
                            }
                            
                            // Add blocks in order to blockchain tree
                            for &orphan_block_id in &blocks_to_add {
                                if let Some(orphan_parent_id) = self.blocks[&orphan_block_id].parent_id {
                                    node.blockchain_tree.blocks.insert(orphan_block_id, time);
                                    node.blockchain_tree.children
                                        .entry(orphan_parent_id)
                                        .or_default()
                                        .push(orphan_block_id);
                                }
                            }
                            
                            // Find the new tip among all added blocks (highest block height)
                            let mut new_tip = block_id;
                            let mut max_height = self.blocks[&block_id].block_height;
                            
                            for &orphan_block_id in &blocks_to_add {
                                if self.blocks[&orphan_block_id].block_height > max_height {
                                    max_height = self.blocks[&orphan_block_id].block_height;
                                    new_tip = orphan_block_id;
                                }
                            }
                            // Check if the longest tip has changed
                            if max_height > self.blocks.get(&node.blockchain_tree.tip).map(|b| b.block_height).unwrap_or(0) {
                                let old_tip = node.blockchain_tree.tip;
                                node.blockchain_tree.tip = new_tip;

                                // Find the common ancestor
                                let mut old_tipp = old_tip;
                                let mut new_tipp = new_tip;

                                let block_height_old = self.blocks[&old_tipp].block_height;
                                let mut block_height_new = self.blocks[&new_tipp].block_height;

                                while block_height_new > block_height_old {
                                    if let Some(parent) = self.blocks[&new_tipp].parent_id {
                                        new_tipp = parent;
                                        block_height_new -= 1;
                                    }
                                }

                                while old_tipp != new_tipp {
                                    if let Some(parent_a) = self.blocks[&new_tipp].parent_id {
                                        new_tipp = parent_a;
                                    }

                                    if let Some(parent_b) = self.blocks[&old_tipp].parent_id {
                                        old_tipp = parent_b;
                                    }
                                }

                                let ancestor_id = old_tipp;
                                
                                // Fork resolution: revert old chain and apply new chain
                                let mut old_chain = Vec::new();
                                let mut new_chain = Vec::new();
                                
                                // Build old chain back to common ancestor
                                let mut current = old_tip;
                                while current != ancestor_id && self.blocks.contains_key(&current) {
                                    new_chain.push(current);
                                    if let Some(parent) = self.blocks[&current].parent_id {
                                        current = parent;
                                    } else {
                                        break;
                                    }
                                }
                                
                                // Build new chain from new tip back to parent
                                current = new_tip;
                                while current != ancestor_id && self.blocks.contains_key(&current) {
                                    new_chain.push(current);
                                    if let Some(parent) = self.blocks[&current].parent_id {
                                        current = parent;
                                    } else {
                                        break;
                                    }
                                }

                                let ancestor_block_height = self.blocks[&ancestor_id].block_height;

                                for &confirmed_block in &node.confirmed_blocks {
                                    if !new_chain.contains(&confirmed_block) && self.blocks[&confirmed_block].block_height > ancestor_block_height {
                                        old_chain.push(confirmed_block);
                                    }
                                }

                                for &orphan_block_id in &blocks_to_add {
                                    if !new_chain.contains(&orphan_block_id) {
                                        old_chain.push(orphan_block_id);
                                    }
                                }

                                
                                // Revert old chain: remove from confirmed_blocks and add transactions back to mempool
                                for &old_block in &old_chain {
                                    node.confirmed_blocks.remove(&old_block);
                                    for tx in self.blocks[&old_block].transactions.clone() {
                                        if !node.mempool.transactions.contains(&tx) {
                                            node.mempool.transactions.push(tx);
                                        }
                                    }
                                }
                                
                                // Apply new chain: add to confirmed_blocks and remove transactions from mempool
                                new_chain.reverse(); // Apply from parent to tip
                                for &new_block in &new_chain {
                                    node.confirmed_blocks.insert(new_block);
                                    for tx in self.blocks[&new_block].transactions.clone() {
                                        node.mempool.transactions.retain(|tx_id| *tx_id != tx);
                                    }
                                }
                                
                                // CRITICAL: Handle side chains from orphaned tree
                                // Only blocks on the path to new_tip are confirmed, others go to mempool
                                // let confirmed_path: HashSet<u32> = new_chain.iter().cloned().collect();
                                // confirmed_path.insert(block_id); // Include the current block
                                
                                // for &orphan_block_id in &blocks_to_add {
                                //     if !confirmed_path.contains(&orphan_block_id) {
                                //         // This block is on a side chain, add its transactions to mempool
                                //         for tx in self.blocks[&orphan_block_id].transactions.clone() {
                                //             if !node.mempool.transactions.contains(&tx) {
                                //                 node.mempool.transactions.push(tx);
                                //             }
                                //         }
                                //     }
                                // }
                            } else {
                                // Tip hasn't changed, add transactions to mempool for all orphaned blocks
                                for &orphan_block_id in &blocks_to_add {
                                    for tx in self.blocks[&orphan_block_id].transactions.clone() {
                                        if !node.mempool.transactions.contains(&tx) {
                                            node.mempool.transactions.push(tx);
                                        }
                                    }
                                }
                                
                                // Also add current block's transactions to mempool since it's not on main chain
                                for tx in self.blocks[&block_id].transactions.clone() {
                                    if !node.mempool.transactions.contains(&tx) {
                                        node.mempool.transactions.push(tx);
                                    }
                                }
                            }
                        }
                        
                    } else {
                        // Add to blockchain tree (common operations)
                        node.blockchain_tree.blocks.insert(block_id, time);
                        node.blockchain_tree.children
                            .entry(self.blocks[&block_id].parent_id.unwrap())
                            .or_default()
                            .push(block_id);

                        // Now check if received block's parent is current tip or not
                        if parent_id == node.blockchain_tree.tip {
                            node.blockchain_tree.tip = block_id;

                            // Remove the transactions of this block from mempool
                            for tx in self.blocks[&block_id].transactions.clone() {
                                if node.mempool.transactions.contains(&tx) {
                                    node.mempool.transactions.retain(|tx_id| *tx_id != tx);
                                }
                            }

                            node.confirmed_blocks.insert(block_id);
                        } else {
                            // Check is this block changes the longest tip
                            if self.blocks[&block_id].block_height > self.blocks[&node.blockchain_tree.tip].block_height {
                                let old_tip = node.blockchain_tree.tip;
                                node.blockchain_tree.tip = block_id;

                                // Fork Resolution
                                // Remove the transactions of this block from mempool
                                for tx in self.blocks[&block_id].transactions.clone() {
                                    if node.mempool.transactions.contains(&tx) {
                                        node.mempool.transactions.retain(|tx_id| *tx_id != tx);
                                    }
                                }

                                let mut a = old_tip;
                                let mut b = block_id;

                                node.confirmed_blocks.insert(block_id);

                                // Find common ancestor
                                while a != b {
                                    if self.blocks[&a].block_height > self.blocks[&b].block_height {
                                        // Add transactions of A into mempool
                                        for tx in self.blocks[&a].transactions.clone() {
                                            if !node.mempool.transactions.contains(&tx) {
                                                node.mempool.transactions.push(tx);
                                            }
                                        }
                                        node.confirmed_blocks.remove(&a);
                                        
                                        if let Some(parent_a) = self.blocks[&a].parent_id {
                                            a = parent_a;
                                        }
                                    } else {
                                        // Remove transactions of B from mempool
                                        for tx in self.blocks[&b].transactions.clone() {
                                            node.mempool.transactions.retain(|tx_id| *tx_id != tx);
                                        }
                                        node.confirmed_blocks.insert(b);
                                        
                                        if let Some(parent_b) = self.blocks[&b].parent_id {
                                            b = parent_b;
                                        }
                                    }
                                }
                            } else {
                                // Add it's transactions to mempool (since it's not on main chain)
                                for tx in self.blocks[&block_id].transactions.clone() {
                                    if !node.mempool.transactions.contains(&tx) {
                                        node.mempool.transactions.push(tx);
                                    }
                                }
                            }
                        }
                    }
                } else {
                    // Case 2: Parent is not present - add to orphaned blocks
                    
                    // Check if this block can be added to an existing orphaned tree
                    let mut added_to_existing = false;
                    
                    for (_x, orphan_tree) in &mut node.orphan_blocks {
                        // Check if parent exists in this orphaned tree
                        if orphan_tree.blocks.contains(&parent_id) {
                            // Add block to this orphaned tree
                            orphan_tree.blocks.insert(block_id);
                            orphan_tree.children.entry(parent_id).or_default().push(block_id);
                            
                            // Update tip if this block has higher height
                            if self.blocks[&block_id].block_height > self.blocks[&orphan_tree.tip].block_height {
                                orphan_tree.tip = block_id;
                            }
                            
                            added_to_existing = true;
                            break;
                        }
                        
                        // Check if this block is parent to the root of this orphaned tree  
                        if orphan_tree.root == block_id || 
                        (self.blocks[&orphan_tree.root].parent_id == Some(block_id)) {
                            // This block becomes the new root of this tree
                            orphan_tree.blocks.insert(block_id);
                            orphan_tree.children.insert(block_id, vec![orphan_tree.root]);
                            orphan_tree.root = block_id;
                            
                            // Tip remains the same unless this block has higher height
                            if self.blocks[&block_id].block_height > self.blocks[&orphan_tree.tip].block_height {
                                orphan_tree.tip = block_id;
                            }
                            
                            added_to_existing = true;
                            break;
                        }
                    }
                    
                    if !added_to_existing {
                        // Create a new orphaned tree with this block as root and tip
                        let new_orphaned_tree = OrphanedTree {
                            blocks: {
                                let mut set = HashSet::new();
                                set.insert(block_id);
                                set
                            },
                            children: HashMap::new(),
                            tip: block_id,
                            root: block_id,
                        };
                        
                        node.orphan_blocks.insert(block_id, new_orphaned_tree);
                    }
                    
                    // Add transactions to seen_transactions
                    for tx in self.blocks[&block_id].transactions.clone() {
                        if !node.seen_transactions.contains_key(&tx) {
                            node.seen_transactions.insert(tx, time);
                        }
                    }
                }
            }

            peer_ids = node.peers.iter().copied().collect();
        } else {
            eprintln!("No such node exists : {}", node_id);
        }

        // broadcast to peer if block is valid
        if valid_block {
            for peer_id in peer_ids {
                if peer_id == sender {
                    continue;
                }
                let block_size = 1024 * self.blocks[&block_id].transactions.len() as u64;
                let latency = self.simulate_latency(block_size, node_id, peer_id);
                self.schedule_receive_block_event(node_id, peer_id, block_id, time + latency);
            }
        }
        
    }


    // Schedule a Generate Tx Event on a Node with a radomized transaction
    fn schedule_generate_tx_event(&mut self, i: u32, time: f64) {
        let n = self.nodes.len() as u32;

        let js = select_random_numbers(n, 1, Some(i));
        let mut j = i;
        if !js.is_empty() {
            j = js[0];
        }

        let balance = self.blocks[&self.nodes[i as usize].blockchain_tree.tip].balances[i as usize];
        let mut amount = balance;
        if balance > 1 {
            let mut rng = rand::thread_rng();
            amount = rng.gen_range(1..=balance);
        }

        let wait_time = sample_exponential(self.cfg.tx_interval_mean_ms);

        let transaction = Transaction {
            id: self.transactions.len() as u32,
            from: Some(i),
            to: j,
            amount,
            created_at: OrderedFloat(time + wait_time),
        };

        let event = Event {
            node_id: i,
            event_type: EventType::GenerateTransaction { transaction_id: transaction.id },
        };

        self.scheduler.schedule(event, OrderedFloat(time + wait_time));

        self.transactions.insert(transaction.id, transaction);

    }


    // Schedule a Receive Tx event on a peer
    fn schedule_receive_tx_event(&mut self, source: u32, destination: u32, message_length: u64, time: f64, transaction_id: u32) {
        let latency = self.simulate_latency(message_length, source, destination);

        let event = Event {
            node_id: destination,
            event_type: EventType::ReceiveTransaction { transaction_id, sender: source }
        };

        self.scheduler.schedule(event, OrderedFloat(time + latency));
    }

    fn schedule_receive_block_event(&mut self, source: u32, destination: u32, block_id: u32, time: f64) {
        let event = Event {
            node_id: destination,
            event_type: EventType::ReceiveBlock { block_id, sender: source }
        };

        self.scheduler.schedule(event, OrderedFloat(time));
    }



    // Simulate Latencies
    fn simulate_latency(&mut self, message_length: u64, source_node: u32, destination_node: u32) -> f64 {
        let mut latency = self.cfg.positive_min_latency_ms;  // rho_i_j

        let mut link_speed = self.cfg.slow_link_speed;
        if self.nodes[source_node as usize].is_fast && self.nodes[destination_node as usize].is_fast {
            link_speed = self.cfg.fast_link_speed;
        }

        latency += (message_length as f64) / (link_speed as f64 * 1000.0);  // Link delay added

        let mean_queuing_delay_ms = 96.0 / (link_speed as f64);
        latency += sample_exponential(mean_queuing_delay_ms);  // Queuing Delay added


        latency
    }


}
