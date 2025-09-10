# Cryptocurrency Network Simulation

A discrete-event simulation of a peer-to-peer cryptocurrency network.

## How to Run

```bash
# Install the repository
git clone https://github.com/Shivam-Gujjar-Boy/Simulator-for-p2p-crypto-network.git
cd Simulator-for-p2p-crypto-network
```

## Configuration Parameters

- Set initial parameters in the main function by modifying the *simulation_params* vector.

## Example:

```rust
// Example configuration
(10, 20, 30, 100.0, 200.0, 30, 500.0),
(15, 25, 35, 150.0, 250.0, 40, 600.0),
(20, 30, 40, 200.0, 300.0, 50, 700.0),
// Add more configurations here
```

## Run the simulation

```bash
cargo run
```

## Output

After running, results will be exported into a time_priority directory.
Each simulation run will create a separate folder.

```
tree reports/
├── simulation_1/
│   ├── node_0_tree.json
│   ├── node_1_tree.json
│   └── ...
├── simulation_2/
│   ├── node_0_tree.json
│   ├── node_1_tree.json
│   └── ...
```