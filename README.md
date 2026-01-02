#  Lamarr-Turing Secure Network Simulator

A Python-based simulation environment for testing secure routing protocols in Mobile Ad-hoc Networks (MANETs). This project specifically demonstrates the **Lamarr-Turing (LT)** protocol's ability to survive in hostile "Cyber-Warfare" environments where standard protocols like AODV fail.

*(Replace this with one of your GUI screenshots)*

##  Overview

Standard routing protocols (like AODV) optimize for the **shortest path**. In a hostile network, the shortest path is often a trap—congested, monitored, or jammed by adversaries.

This simulator creates a **"Honey-Pot" Topology**:

* **The Center (Red Zone):** The shortest path, but populated with low-security, easily jammed nodes.
* **The Edges (Green Zone):** A longer path, but populated with high-security, trusted nodes.

This project visualizes how the **Lamarr-Turing** protocol intelligently routes around the danger zone, while AODV blindly walks into the trap.

##  Features

* **Real-time Visualization:** Tkinter-based GUI with dynamic node coloring (Green=Safe, Red=Danger).
* **Multi-Protocol Comparison:**
* **Lamarr-Turing (LT):** Custom Trust-based Geographic Routing.
* **AODV:** Standard Ad-hoc On-Demand Distance Vector.
* **SAODV:** Secure AODV (Simulated trust filtering).
* **Q-Routing:** Reinforcement Learning based routing.


* **Attack Simulation:** Probabilistic modeling of **Jamming** (Physical Layer) and **MITM** (Network Layer) attacks based on hardware security levels.
* **Integrated Analytics:** Built-in `Matplotlib` plotting for:
* Safety vs. PDR Correlation.
* Performance Bar Charts.


* **Zero-Asset Dependency:** PC icons are embedded directly in the code (Base64), so no external image files are needed.

##  Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/lamarr-turing-sim.git
cd lamarr-turing-sim

```


2. **Install dependencies:**
The core simulation uses standard Python libraries. `matplotlib` is required only for the "Plot Analysis" feature.
```bash
pip install matplotlib

```


3. **Run the simulator:**
```bash
python main.py

```



##  Simulation Logic

### The "Honey-Pot" Topology

The graph is generated with a specific bias:

* **Center Nodes:** Assigned `Security < 0.3`. High probability of packet loss (80% jam risk).
* **Edge Nodes:** Assigned `Security > 0.8`. Low probability of packet loss (1% jam risk).

### Metrics

* **PDR (Packet Delivery Ratio):** Percentage of packets that successfully reach the destination.
* **Avg Safety:** The average hardware security score of all nodes in the chosen path.
* **Avg Trust:** The historical reputation score of the links used.

##  Protocol Comparison

| Protocol | Strategy | Behavior in "Honey-Pot" | Typical PDR |
| --- | --- | --- | --- |
| **AODV** | Shortest Path (BFS) | Takes the center path (Trap). Gets jammed. | **~2% - 10%** |
| **Lamarr-Turing** | Greedy Best-First (Security * Distance) | Detects low security. Routes around the edge. | **~70% - 90%** |
| **Q-Routing** | Reinforcement Learning | Starts random, eventually learns the safe path (slow convergence). | **~40% - 60%** |

## 📸 Screenshots

### 1. The Visualization

*Green nodes represent safe infrastructure. Red nodes are the "Danger Zone".*

### 2. Performance Analysis

*Scatter plot showing LT maintaining high safety scores compared to AODV.*
