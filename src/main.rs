mod config;
mod event;
mod scheduler;
mod chain;
mod mempool;
mod node;
use std::time::{SystemTime, UNIX_EPOCH};
use std::{collections::HashSet, io};

use ordered_float::OrderedFloat;
use rand::{seq::SliceRandom, thread_rng, Rng};

use crate::chain::{Block, Transaction};
use crate::node::Node;

use crate::config::Config;
use crate::event::{Event, EventType};
use crate::scheduler::Scheduler;
   
#[derive(Debug)]
pub struct Simulation {
    pub cfg: Config, // simulation parameters
    pub nodes: Vec<node::Node>, // nodes
    pub scheduler: Scheduler, // contains the min-heap (event queue)
    pub next_transaction_id: u32, // so that no 2 txns have same TxID
    pub next_block_id: u32, // so that no 2 blocks have same BlkID
}



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

    let n: u32 = read_input("Enter number of nodes (positive integer):");
    let z0: u32 = read_input("Enter percentage of slow nodes (0–100):");
    let z1: u32 = read_input("Enter percentage of low CPU nodes (0–100):");
    let ttx: f64 = read_input("Enter mean time for transaction generation in milliseconds (float):");
    let i: f64 = read_input("Enter average interarrival time between 2 blocks in milliseconds (float):");
    let t_total: f64 = read_input("Enter the total time for simulation to run:");
    // let initiate_mining_delay: f64 = read_input("Enter the amount of time to wait to initiate mining after start of simulation, in milliseconds:");

    println!("\n--- Input Summary ---");
    println!("Number of nodes: {}", n);
    println!("Percentage of slow nodes: {}%", z0);
    println!("Percentage of low CPU nodes: {}%", z1);
    println!("Mean transaction generation time (Ttx) milliseconds: {}", ttx);
    println!("Average interarrival time (I): {} milliseconds", i);
    println!("Total Simulation time (T_total): {} milliseconds", t_total);
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
    println!("{:?}", slow_node_indices);
    println!("{:?}", low_cpu_node_indices);

    let mut nodes_propoerties: Vec<(bool, bool)> = vec![(true, true); n as usize];

    for i in slow_node_indices {
        nodes_propoerties[i as usize].0 = false;
    } // O(n)

    for i in low_cpu_node_indices {
        nodes_propoerties[i as usize].1 = false;
    } // O(n)

    println!("Nodes Properties: {:?}", nodes_propoerties);


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
            n
        );

        nodes.push(node);
    }

    println!("Nodes Generated according to the Topology!");

    // Calculate initial parameters
    let mut rng = rand::thread_rng();
    let positive_min_latency: f64 = rng.gen_range(10..=500) as f64;


    // Create the Simulation Object
    let config = Config::new(n, t_total, ttx, positive_min_latency, i);

    let mut simulation = Simulation {
        cfg: config,
        nodes,
        scheduler: Scheduler::new(),
        next_transaction_id: 0,
        next_block_id: 0
    };


    // Schedule GenerateTx Event for each Node in beginning
    for i in 0..n {
        // Find a random peer to send coins to
        let js = select_random_numbers(n, 1, Some(i));
        let mut j = i;
        if js.len() != 0 {
            j = js[0];
        }

        // Now check balance of Peer i according to Node i and assuming coin has only 1 decimal
        let balance = simulation.nodes[i as usize].balances[i as usize];
        let mut amount = balance;
        if balance != 0 {
            let mut rng = rand::thread_rng();
            let random_amount = rng.gen_range(1..=balance);
            amount = random_amount;
        }

        println!("Amount = {}", amount);


        // Simulate Latency
        // let mut link_speed = simulation.cfg.slow_link_speed;
        // if simulation.nodes[i as usize].is_fast && simulation.nodes[j as usize].is_fast {
        //     link_speed = simulation.cfg.fast_link_speed;
        // }

        // let mut latency = simulation.cfg.positive_min_latency_ms;
        // latency += 1024.0 / (link_speed as f64 * 1000.0);
        // let mean_queuing_delay_ms = 96.0 / (link_speed as f64);
        // latency += sample_exponential(mean_queuing_delay_ms);

        // println!("Latency = {}", latency);

        // Schedule GenerateTx Event
        let transaction = Transaction {
            id: simulation.next_transaction_id,
            from: Some(i),
            to: j,
            amount,
            created_at: OrderedFloat(0.0),
            received_at: OrderedFloat(0.0),
            received_from: None
        };

        let event = Event {
            node_id: i,
            event_type: EventType::GenerateTransaction { transaction }
        };

        simulation.scheduler.schedule(event, OrderedFloat(0.0));
        

        simulation.next_transaction_id += 1;
    }

    // Schedule MineBlock Event for each node in beginning
    for i in 0..n {
        // Simulate PoW. Calculate Tk. Assuming no transactions at that time, all nodes are just mining empty blocks on top of genesis block
        let mean_ms = simulation.cfg.mine_interval_ms / simulation.nodes[i as usize].hashing_power_fraction;
        let t_k = sample_exponential(mean_ms);

        // Create Empty Block to Mine on top of Genesis Block
        let block = Block {
            block_id: simulation.next_block_id,
            prev_id: Some(0),
            transactions: vec![],
            timestamp: OrderedFloat(t_k),
            block_height: 1,
            miner: Some(i),
            received_at: OrderedFloat(t_k)
        };

        // Create MineBlock Event at t = Tk
        let event = Event {
            node_id: i,
            event_type: EventType::MineBlock { block }
        };

        simulation.scheduler.schedule(event, OrderedFloat(t_k));

        simulation.next_block_id += 1;
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


}



impl Simulation {

    // Handle All Events Conditionally
    fn handle_event(&mut self, event: Event, time: f64) {

        match event.event_type {
            EventType::GenerateTransaction { transaction } => {
                self.handle_generate_transaction(event.node_id, transaction, time);
            }

            EventType::ReceiveTransaction { transaction } => {
                self.handle_receive_transaction(event.node_id, transaction, time);
            }

            EventType::MineBlock { block } => {
                self.handle_mine_block(event.node_id, block, time);
            }

            EventType::ReceiveBlock { block } => {
                self.handle_receive_block(event.node_id, block, time);
            }
        }

    }


    // EVENT TYPE 1 -> Handle Generate Transaction Event Execution
    fn handle_generate_transaction(&mut self, node_id: u32, transaction: Transaction, time: f64) {

        if let Some(node) = self.nodes.get_mut(node_id as usize) {
            // Verify Transaction
            let sender_balance = node.balances.get(node_id as usize).copied().unwrap_or(0);

            if transaction.amount <= sender_balance {
                // Add txn to mempool
                node.mempool.add_transaction(transaction.clone());

                // Add txnID to seen transactions hashset
                node.seen_transactions.insert(transaction.id);

                // Schedule Receive Txn Events for neighbor nodes
                let peer_ids: Vec<u32> = self.nodes[node_id as usize].peers.iter().copied().collect();
                for peer_id in peer_ids {
                    self.schedule_receive_tx_event(node_id, peer_id, 1024, time, transaction.clone());
                }
            }

            // Now Schedule one more Generate Transaction Event on this node
            self.schedule_generate_tx_event(node_id, time);


        } else {
            eprintln!("Invalid node_id {} in handle_generate_transaction", node_id);
        }

    }


    // EVENT TYPE 2 -> Handle Receive Transaction Event Execution
    fn handle_receive_transaction(&mut self, node_id: u32, transaction: Transaction, time: f64) {
        if let Some(node) = self.nodes.get_mut(node_id as usize) {
            // When a transaction is received, no need to verify it
            // Check if this node has already seen this transaction or not
            if !node.seen_transactions.contains(&transaction.id) {
                // Mark the transaction as seen
                node.seen_transactions.insert(transaction.id);

                // Add transaction to mempool
                node.mempool.add_transaction(transaction.clone());

                // Schedule Receive Txn Events for neighbor nodes, excluding the one it received tx from
                let peer_ids: Vec<u32> = self.nodes[node_id as usize]
                    .peers
                    .iter()
                    .copied()
                    .filter(|&peer_id| Some(peer_id) != transaction.received_from)
                    .collect();
                for peer_id in peer_ids {
                    self.schedule_receive_tx_event(node_id, peer_id, 1024, time, transaction.clone());
                }

            }
        } else {
            eprintln!("Invalid node_id {} in handle_receive_transaction", node_id);
        }
    }


    // EVENT TYPE 3 -> Handle Mine Block Event Execution
    fn handle_mine_block(&mut self, node_id: u32, block: Block, time: f64) {
        if let Some(node) = self.nodes.get_mut(node_id as usize) {
            let blockchain_tree = &mut node.blockchain_tree;

            let mut modified_block = block.clone();

            // Check if Block is still building on current tip
            if block.prev_id == Some(blockchain_tree.tip) {
                // Add coinbase transaction
                let coinbase_tx = Transaction {
                    id: self.next_transaction_id,
                    from: None,
                    to: node_id,
                    amount: 500,
                    created_at: OrderedFloat(time),
                    received_at: OrderedFloat(time),
                    received_from: None
                };

                modified_block.transactions.insert(0, coinbase_tx);

                // Add block to blockchain tree and modify the tip
                let block_id = block.block_id;
                blockchain_tree.blocks.insert(block_id, modified_block.clone());
                blockchain_tree.children
                    .entry(block.prev_id.unwrap())
                    .or_default()
                    .push(block_id);
                blockchain_tree.tip = block_id;

                // Wrong from here

                // Schedule recive block events for neighbors
                let peer_ids: Vec<u32> = node.peers.iter().copied().collect();
                for peer_id in peer_ids {
                    self.schedule_receive_block_event(node_id, peer_id, modified_block.clone(), time);
                }


                // Schedule next mine block event on this node
                let mut selected_txns: Vec<Transaction> = Vec::new();
                if node.mempool.transactions.len() <= 999 {
                    selected_txns = node.mempool.transactions.clone();
                } else {
                    selected_txns = node.mempool.transactions.iter().take(999).cloned().collect();
                }

                let wait_time = sample_exponential(self.cfg.mine_interval_ms);

                let next_block = Block {
                    block_id: self.next_block_id,
                    prev_id: Some(blockchain_tree.tip),
                    transactions: selected_txns,
                    timestamp: OrderedFloat(time + wait_time),
                    block_height: blockchain_tree.blocks[&blockchain_tree.tip].block_height + 1,
                    miner: Some(node_id),
                    received_at: OrderedFloat(time + wait_time)
                };

                self.schedule_mine_block(node_id, next_block)

            }
        }
    }


    // EVENT TYPE 4 -> Handle Receive Block Event Execution
    fn handle_receive_block(&mut self, node_id: u32, block: Block, time: f64) {

    }


    // Schedule a Generate Tx Event on a Node with a radomized transaction
    fn schedule_generate_tx_event(&mut self, i: u32, time: f64) {
        let n = self.nodes.len() as u32;

        let js = select_random_numbers(n, 1, Some(i));
        let mut j = i;
        if !js.is_empty() {
            j = js[0];
        }

        let balance = self.nodes[i as usize].balances[i as usize];
        let mut amount = balance;
        if balance != 0 {
            let mut rng = rand::thread_rng();
            amount = rng.gen_range(1..=balance);
        }

        let wait_time = sample_exponential(self.cfg.tx_interval_mean_ms);

        let transaction = Transaction {
            id: self.next_transaction_id,
            from: Some(i),
            to: j,
            amount,
            created_at: OrderedFloat(time + wait_time),
            received_at: OrderedFloat(time + wait_time),
            received_from: None
        };

        let event = Event {
            node_id: i,
            event_type: EventType::GenerateTransaction { transaction },
        };

        self.scheduler.schedule(event, OrderedFloat(time + wait_time));

        self.next_transaction_id += 1;

    }


    // Schedule a Receive Tx event on a peer
    fn schedule_receive_tx_event(&mut self, source: u32, destination: u32, message_length: u64, time: f64, transaction: Transaction) {
        let latency = self.simulate_latency(message_length, source, destination);

        let mut modified_transaction = transaction.clone();
        modified_transaction.received_at = OrderedFloat(time + latency);
        modified_transaction.received_from = Some(source);

        let event = Event {
            node_id: destination,
            event_type: EventType::ReceiveTransaction { transaction: modified_transaction }
        };

        self.scheduler.schedule(event, OrderedFloat(time + latency));
    }

    fn schedule_receive_block_event(&mut self, source: u32, destination: u32, block: Block, time: f64) {

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
