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
use crate::node::Node;

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
    // let ttx: f64 = read_input("Enter mean time for transaction generation in milliseconds (float):");
    // let i: f64 = read_input("Enter average interarrival time between 2 blocks in milliseconds (float):");
    let t_total: f64 = read_input("Enter the total time for simulation to run:");
    // let initiate_mining_delay: f64 = read_input("Enter the amount of time to wait to initiate mining after start of simulation, in milliseconds:");

    let n: u32 = 7;
    let z0: u32 = 90;
    let z1: u32 = 90;
    let ttx: f64 = 1000.0;
    let i: f64 = 3000.0;

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
        let mut modified_block = self.blocks[&block_id].clone();
        let mut peer_ids = vec![];
        let mut next_block: Option<Block> = None;
        let mut valid_block = false;

        {
            if let Some(node) = self.nodes.get_mut(node_id as usize) {

                // Case 1: Block builds on current tip
                if modified_block.parent_id == Some(node.blockchain_tree.tip) {

                    // Modify the balances of all peers on this node
                    // iterate through all transactions in the block, modify balances vector in node.balances
                    // After that check is someone's balance is negative, if not, then valid_block = true, else jump out of overall if
                    // Do add 500 coins to node_id, as coinbase txn gives 500 reward
                    let parent_id = self.blocks[&block_id].parent_id;
                    if let Some(parent_id) = parent_id {
                        let mut balances = self.blocks[&parent_id].balances.clone();
                        valid_block = true;

                        for tx_id in &modified_block.transactions {
                            if let Some(from) = self.transactions[tx_id].from {
                                if let Some(balance) = balances.get_mut(from as usize) {
                                    *balance -= self.transactions[tx_id].amount;
                                } else {
                                    valid_block = false;
                                    break;
                                }
                            }

                            balances[self.transactions[tx_id].to as usize] += self.transactions[tx_id].amount;

                            if balances.iter().any(|balance| *balance < (0 as i64)) {
                                valid_block = false;
                                break;
                            }
                        }

                        if valid_block {
                            // Add coinbase transaction
                            let coinbase_tx = Transaction {
                                id: self.transactions.len() as u32,
                                from: None,
                                to: node_id,
                                amount: 500,
                                created_at: OrderedFloat(time)
                            };
                            modified_block.transactions.insert(0, coinbase_tx.id);
                            self.transactions.insert(coinbase_tx.id, coinbase_tx);

                            balances[node_id as usize] += 500;
                            modified_block.balances = balances;

                            // Add block to blockchain tree
                            let block_id = modified_block.block_id;
                            node.blockchain_tree.blocks.insert(block_id, time);
                            node.blockchain_tree.children
                                .entry(modified_block.parent_id.unwrap())
                                .or_default()
                                .push(block_id);
                            node.blockchain_tree.tip = block_id;

                            // Store peer ids for broadcasting
                            peer_ids = node.peers.iter().copied().collect();

                            // Add this block_id to confirmed blocks
                            node.confirmed_blocks.insert(block_id);
                        }

                        self.blocks[&block_id] = modified_block.clone();
                    }

                } else {
                    // Tip moved while mining this block.
                    // DUMP this block's transaction back to mmempool ONLY IF:
                    //  - they are not already in mempool
                    //  - they are not in any confirmed block on the current longest chain

                    // println!("Mine failed: {}", block.block_id);

                    let confirmed_ids = node.confirmed_blocks.clone();

                    for tx in modified_block.transactions.into_iter() {
                        if !confirmed_ids.contains(&tx) && !node.mempool.transactions.contains(&tx) {
                            node.mempool.transactions.push(tx);
                        }
                    }

                }

                // In both cases, we schedule next mine block event
                // For that select a set of transactions to be included in the block
                let balances = self.blocks[&node.blockchain_tree.tip].balances.clone();

                let selected_txns: Vec<u32> = Vec::new();

                for tx in node.mempool.transactions.iter() {
                    if selected_txns.len() >= 1023 {
                        break;
                    }

                    if let Some(from) = self.transactions[tx].from {
                        if balances[from as usize] >= self.transactions[tx].amount {
                            selected_txns.push(*tx);
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
                let block_size = 1024 * modified_block.transactions.len() as u64;
                let latency = self.simulate_latency(block_size, node_id, peer_id);
                self.schedule_receive_block_event(node_id, peer_id, modified_block.clone(), time + latency);
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
    fn handle_receive_block(&mut self, _node_id: u32, _block_id: u32, _time: f64, _sender: u32) {
        // STEPS TO BE FOLLOWED CONDITIONALLY =>
        //
        // First check if the transactions in the block are valid or not
        // If block is valid, then we modify the blockchain tree
        // First add this block to blockchain_tree.blocks
        // Then check for it's parent from the prev_id field
        // And add it's block id to the children vector of it's parent
        // -----------------------------------------------------------
        // Now check if the it is added to the current longest tip
        // If it is added to the current longest tip, then change the tip to this block's id
        // Modify the balances by procesing these transactions and add this block to confirmed blocks vector
        // -----------------------------------------------------------
        // But if not, then see if this block creates a new longest chain
        // If it doesn't create the new longest chain, then just add the transactions in this block in the node's mempool
        // If it creates a new longest chain, this block itself is the new tip
        // So, the tip is changed, we have to do some work
        // -> First figure out block ids of all the blocks which are now in the orphaned chain, but were in longest chain before adding this new block
        // -> Add the transactions in those orphaned in the mempool.
        // -> Now check teh blocks which were orphaned before adding this new block, but are confirmed now
        // -> Remove their transactions from the mempool
        // -> Now modify the node.confirmed_blocks vector present in this node
        // -> And then iterate through the transactions of these confirmed blocks to find the modified balances of all nodes
        // -> And then update the balances vector in node.

        // let node = &mut self.nodes[node_id as usize];

        // if node.seen_blocks.contains(&block.block_id) {
        //     return;
        // }

        // let mut modified_block = block.clone();

        // node.seen_blocks.insert(block.block_id);
        // modified_block.received_at = OrderedFloat(time);

        // if let Some(parent_id) = block.prev_id {
        //     if !node.blockchain_tree.blocks.contains_key(&parent_id) && parent_id != node.blockchain_tree.genesis {
        //         node.orphan_blocks.insert(block.block_height, modified_block);
        //         return;
        //     }
        // } else {

        // }

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
        if balance != 0 {
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

    fn schedule_receive_block_event(&mut self, _source: u32, _destination: u32, _block: Block, _time: f64) {

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
