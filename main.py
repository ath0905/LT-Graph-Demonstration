import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import random, math, hashlib, collections, time

# Check for Matplotlib
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB = True
except ImportError:
    MATPLOTLIB = False
    print("Warning: Matplotlib not found. Plotting features will be disabled.")

# ================= CONFIGURATION =================
NUM_NODES = 40       
ROUNDS = 100
MAX_HOPS = 40        
DELAY = 50           

# ================= CRYPTO & UTILS =================
def HMAC(k, m): return hashlib.sha256((k + str(m)).encode()).hexdigest()
def ENC(x): return f"enc({x})"
def DEC(x): return x.startswith("enc(")

# ================= DATA STRUCTURES =================
class Node:
    def __init__(self, i, x, y):
        self.id = i
        self.x = x
        self.y = y
        self.state = "IDLE"

class Edge:
    def __init__(self, u_node, v_node):
        # --- HONEY-POT TOPOLOGY ---
        center_x, center_y = 450, 380
        mid_x = (u_node.x + v_node.x) / 2
        mid_y = (u_node.y + v_node.y) / 2
        dist_to_center = math.dist((mid_x, mid_y), (center_x, center_y))
        
        # Danger Zone Radius = 180px
        if dist_to_center < 180:
            self.security = random.uniform(0.1, 0.3) # Low Security (Danger)
        else:
            self.security = random.uniform(0.8, 1.0) # High Security (Safe)

        self.trust = 1.0 
        self.jam = 0.0

class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}

    def add_node(self, n): self.nodes[n.id] = n
    def add_edge(self, u, v):
        self.edges[(u, v)] = Edge(self.nodes[u], self.nodes[v])
        self.edges[(v, u)] = self.edges[(u, v)]
    def neighbors(self, u): return [v for (x, v) in self.edges if x == u]

# ================= PHYSICS LAYER (TUNED) =================
def transmit_packet(g, path):
    if not path or len(path) < 2: return False

    for i in range(len(path)-1):
        u, v = path[i], path[i+1]
        if (u, v) not in g.edges: return False
        e = g.edges[(u, v)]
        
        # --- BALANCED ATTACK LOGIC ---
        # If node is in Danger Zone (Security < 0.4), huge risk of jamming.
        # But NOT 100% instant death.
        if e.security < 0.4:
            jam_risk = 0.80  # 80% failure per hop (High, but probabilistic)
        else:
            jam_risk = 0.01  # 1% random failure (Normal noise)

        if random.random() < jam_risk: 
            e.jam += 1
            return False
        
        e.trust = max(0.1, e.trust * 0.99)
        
    return True

# ================= PROTOCOLS =================

# --- 1. Lamarr-Turing (Greedy + Secure) ---
class LT:
    def __init__(self, g):
        self.g = g
        self.name = "Lamarr-Turing"
        self.color = "#00ffcc" # Cyan
        self.key = "LT-SECRET"

    def walk(self, src, dst):
        path = [src]
        curr = src
        visited = {src}
        dst_n = self.g.nodes[dst]

        for _ in range(MAX_HOPS):
            nbrs = self.g.neighbors(curr)
            valid_nbrs = [n for n in nbrs if n not in visited]
            if not valid_nbrs: break
            
            best_nxt = None
            best_score = -1
            
            for n in valid_nbrs:
                e = self.g.edges[(curr, n)]
                n_node = self.g.nodes[n]
                
                # Metric: Security * Distance
                safety_factor = (e.trust * e.security) ** 3
                dist = math.dist((n_node.x, n_node.y), (dst_n.x, dst_n.y))
                dist_score = 1000 / (1 + dist)
                
                total_score = dist_score * safety_factor
                
                if total_score > best_score:
                    best_score = total_score
                    best_nxt = n
            
            if best_nxt is not None:
                path.append(best_nxt)
                visited.add(best_nxt)
                curr = best_nxt
                if curr == dst: return path
            else: break
        return None

    def feedback(self, path, success):
        if not path: return
        impact = 1.05 if success else 0.40
        for i in range(len(path)-1):
            e = self.g.edges[(path[i], path[i+1])]
            e.trust = min(1.0, e.trust * impact)

# --- 2. AODV ---
class AODV:
    def __init__(self, g):
        self.g = g
        self.name = "AODV (Standard)"
        self.color = "#ffaa00" # Orange

    def walk(self, src, dst):
        queue = collections.deque([[src]])
        visited = {src}
        while queue:
            path = queue.popleft()
            node = path[-1]
            if node == dst: return path
            for neighbor in self.g.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = list(path) + [neighbor]
                    queue.append(new_path)
        return None

    def feedback(self, path, success): pass

# --- 3. SAODV ---
class SAODV:
    def __init__(self, g):
        self.g = g
        self.name = "SAODV (Secure)"
        self.color = "#bd93f9" # Purple

    def walk(self, src, dst):
        queue = collections.deque([[src]])
        visited = {src}
        candidate = None
        while queue:
            path = queue.popleft()
            node = path[-1]
            if node == dst: candidate = path; break
            for neighbor in self.g.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = list(path) + [neighbor]
                    queue.append(new_path)
        
        if candidate:
            for i in range(len(candidate)-1):
                e = self.g.edges[(candidate[i], candidate[i+1])]
                if e.trust < 0.6: return None 
        return candidate

    def feedback(self, path, success): pass

# --- 4. Q-Routing ---
class QRouting:
    def __init__(self, g):
        self.g = g
        self.name = "Q-Routing"
        self.color = "#ff79c6" # Pink
        self.Q = {}
        self.alpha = 0.5; self.epsilon = 0.2

    def get_q(self, curr, dst, nbr):
        if (curr, dst) not in self.Q: self.Q[(curr, dst)] = {}
        return self.Q[(curr, dst)].get(nbr, 5.0)

    def walk(self, src, dst):
        path = [src]; curr = src; visited = {src}
        for _ in range(MAX_HOPS):
            nbrs = self.g.neighbors(curr)
            valid = [n for n in nbrs if n not in visited]
            if not valid: break
            
            if random.random() < self.epsilon: nxt = random.choice(valid)
            else:
                q_vals = [self.get_q(curr, dst, n) for n in valid]
                min_q = min(q_vals)
                best = [n for n in valid if self.get_q(curr, dst, n) == min_q]
                nxt = random.choice(best)
            path.append(nxt); visited.add(nxt); curr = nxt
            if curr == dst: return path
        return None

    def feedback(self, path, success):
        reward = 1 if success else 10
        if not path: return
        dst = path[-1]
        for i in range(len(path)-2, -1, -1):
            curr = path[i]; nxt = path[i+1]
            old_q = self.get_q(curr, dst, nxt)
            if nxt == dst: future_q = 0
            else:
                nbrs = self.g.neighbors(nxt)
                future_q = min([self.get_q(nxt, dst, n) for n in nbrs]) if nbrs else 10
            new_q = (1 - self.alpha) * old_q + self.alpha * (reward + 0.8 * future_q)
            if (curr, dst) not in self.Q: self.Q[(curr, dst)] = {}
            self.Q[(curr, dst)][nxt] = new_q

# ================= GUI INTERFACE =================
class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Simulator: Honey-Pot Topology (Secure Routing Test)")
        self.root.configure(bg="#1e1e1e")
        self.root.geometry("1300x850")

        self.main_frame = tk.Frame(self.root, bg="#1e1e1e")
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.canvas = tk.Canvas(self.main_frame, bg="#181818", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        
        self.panel = tk.Frame(self.main_frame, bg="#252526", width=380)
        self.panel.pack(side="right", fill="y", padx=(10, 0))
        self.panel.pack_propagate(False)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Treeview", background="#333", foreground="white", fieldbackground="#333", borderwidth=0)
        style.configure("Treeview.Heading", background="#444", foreground="white", relief="flat")
        
        self.setup_ui()
        self.build_graph()
        
        self.running = False
        self.round = 0
        self.stats = {"sent": 0, "success": 0, "trust": 0.0, "security": 0.0, "hops": 0, "routed": 0}
        self.history = [] 
        
        self.root.mainloop()

    def setup_ui(self):
        tk.Label(self.panel, text="PROTOCOL LAB", fg="#007acc", bg="#252526", font=("Segoe UI", 16, "bold")).pack(pady=(15, 5))
        
        # Select
        self.proto_var = tk.StringVar(value="Lamarr-Turing")
        tk.Label(self.panel, text="Select Protocol:", fg="#aaa", bg="#252526").pack(anchor="w", padx=15)
        cb = ttk.Combobox(self.panel, textvariable=self.proto_var, state="readonly",
                          values=["Lamarr-Turing", "AODV (Standard)", "SAODV (Secure)", "Q-Routing"])
        cb.pack(fill="x", padx=15, pady=5)
        cb.bind("<<ComboboxSelected>>", self.reset)

        # Controls
        cf = tk.LabelFrame(self.panel, text="Controls", bg="#252526", fg="#aaa")
        cf.pack(fill="x", padx=10, pady=5)
        bf = tk.Frame(cf, bg="#252526"); bf.pack(fill="x", pady=5)
        ttk.Button(bf, text="Start", command=self.start).grid(row=0, column=0, padx=2, sticky="ew")
        ttk.Button(bf, text="Pause", command=self.pause).grid(row=0, column=1, padx=2, sticky="ew")
        ttk.Button(bf, text="Reset", command=self.reset).grid(row=1, column=0, padx=2, sticky="ew")
        ttk.Button(bf, text="Plot Analysis", command=self.plot_analysis).grid(row=1, column=1, padx=2, sticky="ew")
        bf.columnconfigure(0, weight=1); bf.columnconfigure(1, weight=1)

        # Metrics
        sf = tk.LabelFrame(self.panel, text="Live Metrics", bg="#252526", fg="#aaa")
        sf.pack(fill="x", padx=10, pady=5)
        self.lbl_round = tk.Label(sf, text="Round: 0 / 100", fg="white", bg="#252526", anchor="w")
        self.lbl_round.pack(fill="x", padx=5)
        self.lbl_pdr = tk.Label(sf, text="PDR: 0.0%", fg="#4fc1ff", bg="#252526", anchor="w", font=("bold"))
        self.lbl_pdr.pack(fill="x", padx=5)
        self.lbl_safety = tk.Label(sf, text="Avg Safety: 0.00", fg="#ffaa00", bg="#252526", anchor="w")
        self.lbl_safety.pack(fill="x", padx=5)
        self.lbl_stat = tk.Label(sf, text="IDLE", fg="#888", bg="#252526")
        self.lbl_stat.pack(fill="x", pady=5)

        # LEGEND
        lf = tk.LabelFrame(self.panel, text="Legend", bg="#252526", fg="#aaa")
        lf.pack(fill="x", padx=10, pady=5)
        self.add_legend_item(lf, "#33aa33", "High Security Node (Safe)")
        self.add_legend_item(lf, "#aa3333", "Low Security Node (Danger)")
        self.add_legend_item(lf, "#4fc1ff", "Active Packet / Source")
        self.add_legend_item(lf, "#f44747", "Hacked / Failed Node")
        self.add_legend_item(lf, "#00ffcc", "Lamarr-Turing Path")
        self.add_legend_item(lf, "#ffaa00", "AODV Path")

        # Leaderboard
        rf = tk.LabelFrame(self.panel, text="Leaderboard", bg="#252526", fg="#aaa")
        rf.pack(fill="both", expand=True, padx=10, pady=5)
        cols = ("proto", "pdr", "safety", "trust")
        self.tree = ttk.Treeview(rf, columns=cols, show="headings", height=8)
        self.tree.heading("proto", text="Protocol"); self.tree.column("proto", width=90)
        self.tree.heading("pdr", text="PDR"); self.tree.column("pdr", width=50)
        self.tree.heading("safety", text="Safety"); self.tree.column("safety", width=50)
        self.tree.heading("trust", text="Trust"); self.tree.column("trust", width=50)
        self.tree.pack(fill="both", expand=True, padx=5, pady=5)

    def add_legend_item(self, parent, color, text):
        f = tk.Frame(parent, bg="#252526")
        f.pack(fill="x", padx=5, pady=2)
        tk.Canvas(f, width=12, height=12, bg=color, highlightthickness=0).pack(side="left")
        tk.Label(f, text=text, fg="#ddd", bg="#252526", font=("Segoe UI", 9)).pack(side="left", padx=5)

    def build_graph(self):
        self.g = Graph()
        cx, cy = 450, 380; rad = 300
        
        for i in range(NUM_NODES):
            a = 2 * math.pi * i / NUM_NODES
            r = rad * math.sqrt(random.uniform(0.0, 1.0)) 
            nx = cx + r * math.cos(a)
            ny = cy + r * math.sin(a)
            self.g.add_node(Node(i, nx, ny))
            
        for i in range(NUM_NODES):
            node_i = self.g.nodes[i]
            dists = []
            for j in range(NUM_NODES):
                if i == j: continue
                node_j = self.g.nodes[j]
                d = math.dist((node_i.x, node_i.y), (node_j.x, node_j.y))
                dists.append((d, j))
            dists.sort()
            for d, j in dists[:6]:
                self.g.add_edge(i, j)
        
        nodes_by_x = sorted(self.g.nodes.values(), key=lambda n: n.x)
        self.src = nodes_by_x[0].id
        self.dst = nodes_by_x[-1].id
        
        sel = self.proto_var.get()
        if "Lamarr" in sel: self.proto = LT(self.g)
        elif "AODV" in sel: self.proto = AODV(self.g)
        elif "SAODV" in sel: self.proto = SAODV(self.g)
        elif "Q-Routing" in sel: self.proto = QRouting(self.g)
        self.draw()

    def start(self): 
        if not self.running: self.running = True; self.step()
    
    def pause(self): self.running = False
    
    def reset(self, event=None):
        self.running = False
        self.round = 0
        self.stats = {"sent": 0, "success": 0, "trust": 0.0, "security": 0.0, "hops": 0, "routed": 0}
        self.build_graph()
        self.update_gui(0, 0, False, [])

    def step(self):
        if not self.running: return
        if self.round >= ROUNDS: self.running = False; self.record_results(); return

        self.round += 1
        for n in self.g.nodes.values(): n.state = "IDLE"

        path = self.proto.walk(self.src, self.dst)
        
        if not path:
            self.proto.feedback(None, False)
            self.stats["sent"] += 1
            self.update_gui(0, 0, False, [])
            self.root.after(DELAY, self.step)
            return

        success = transmit_packet(self.g, path)
        self.proto.feedback(path, success)

        self.stats["sent"] += 1
        if success: self.stats["success"] += 1
        
        p_trust = 0; p_sec = 0
        if len(path) > 1:
            edges = [self.g.edges[(path[i], path[i+1])] for i in range(len(path)-1)]
            p_trust = sum(e.trust for e in edges) / len(edges)
            p_sec = sum(e.security for e in edges) / len(edges)
        
        self.stats["trust"] += p_trust
        self.stats["security"] += p_sec
        self.stats["hops"] += (len(path) - 1)
        self.stats["routed"] += 1 

        for pid in path:
            n = self.g.nodes[pid]
            if pid == self.src: n.state = "SEND"
            elif pid == self.dst: n.state = "VALIDATED" if success else "FAILED"
            else: n.state = "FORWARD"
            
        self.draw(path)
        self.update_gui(p_sec, p_trust, success, path)
        self.root.after(DELAY, self.step)

    def update_gui(self, sec, trust, success, path):
        self.lbl_round.config(text=f"Round: {self.round} / {ROUNDS}")
        sent = self.stats["sent"]
        routed = self.stats["routed"]
        pdr = (self.stats["success"] / sent * 100) if sent > 0 else 0
        self.lbl_pdr.config(text=f"PDR: {pdr:.1f}%")
        avg_s = (self.stats["security"] / routed) if routed > 0 else 0
        self.lbl_safety.config(text=f"Avg Safety: {avg_s:.2f}")

        if not path: self.lbl_stat.config(text="NO ROUTE", fg="#d4d4d4")
        else:
            txt = "SUCCESS" if success else "FAILURE"
            col = "#6a9955" if success else "#f44747"
            self.lbl_stat.config(text=txt, fg=col)

    def record_results(self):
        sent = self.stats["sent"]
        routed = self.stats["routed"]
        if sent == 0: return
        pdr = (self.stats["success"] / sent) * 100
        if routed > 0:
            t = self.stats["trust"] / routed
            s = self.stats["security"] / routed
        else: t = 0; s = 0
        
        self.history.append({"proto": self.proto.name, "pdr": pdr, "safety": s, "trust": t})
        self.tree.insert("", "end", values=(self.proto.name, f"{pdr:.1f}%", f"{s:.2f}", f"{t:.2f}"))

    def plot_analysis(self):
        if not MATPLOTLIB: messagebox.showerror("Error", "Matplotlib not found."); return
        if not self.history: messagebox.showinfo("Info", "No data to plot."); return

        protos = [x["proto"] for x in self.history]
        pdrs = [x["pdr"] for x in self.history]
        safety = [x["safety"] for x in self.history]
        
        color_map = {"Lamarr-Turing": "#00ffcc", "AODV (Standard)": "#ffaa00",
                     "SAODV (Secure)": "#bd93f9", "Q-Routing": "#ff79c6"}
        colors = [color_map.get(p, "#888") for p in protos]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.set_title("Safety vs PDR Correlation")
        ax1.set_xlabel("Average Safety (Hardware/Encryption)")
        ax1.set_ylabel("Packet Delivery Ratio (%)")
        ax1.grid(True, linestyle="--", alpha=0.5)
        for i, p in enumerate(protos):
            ax1.scatter(safety[i], pdrs[i], color=colors[i], s=100, label=p, edgecolors='black')
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys(), loc='lower right')

        ax2.set_title("Performance Comparison")
        bars = ax2.bar(range(len(protos)), pdrs, color=colors, edgecolor='black')
        ax2.set_xticks(range(len(protos)))
        ax2.set_xticklabels(protos, rotation=45, ha='right')
        ax2.set_ylabel("PDR (%)")
        ax2.set_ylim(0, 100)
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}%', ha='center', va='bottom')
        plt.tight_layout()
        plt.show()

    def draw(self, path=None):
        self.canvas.delete("all")
        w, h = 900, 768
        
        # Grid
        for x in range(0, w, 40): self.canvas.create_line(x,0,x,h, fill="#222")
        for y in range(0, h, 40): self.canvas.create_line(0,y,w,y, fill="#222")

        # Edges
        for (u, v), e in self.g.edges.items():
            if u < v:
                a, b = self.g.nodes[u], self.g.nodes[v]
                col = "#444"
                if e.jam > 2: col = "#552222"
                self.canvas.create_line(a.x, a.y, b.x, b.y, fill=col, width=1)

        # Path
        if path:
            for i in range(len(path)-1):
                if (path[i], path[i+1]) in self.g.edges:
                    a, b = self.g.nodes[path[i]], self.g.nodes[path[i+1]]
                    self.canvas.create_line(a.x, a.y, b.x, b.y, fill=self.proto.color, width=3)

        # Base64 PC Icon
        if not hasattr(self, 'pc_img'):
            self.pc_data = """
            R0lGODlhIAAgAPCAAAAAAP///wAAACH5BAEAAAgALAAAAAAgACAAAAJ4jI+py+0Po5y0
            2ouz3rz7D4biSJbmiabqyrbuC8fyTNf2jef6zvf+DwwKh8Ti8YhMKpfMpvMJjUqn1Kr1
            is1qt9yu9wsOi8fksvmMTqvX7Lb7DY/L5/S6/Y7P6/f8vv8PGCg4SFhoeIiYqLjI2Oj4
            CBkpOUlZaXEAADs=
            """
            self.pc_img = tk.PhotoImage(data=self.pc_data)

        # Draw PC Nodes
        for n in self.g.nodes.values():
            # Color based on Security
            nbrs = self.g.neighbors(n.id)
            avg_sec = 0.5
            if nbrs:
                avg_sec = sum([self.g.edges[(n.id, x)].security for x in nbrs]) / len(nbrs)
            
            red = int(255 * (1 - avg_sec))
            green = int(255 * avg_sec)
            fill_col = f"#{red:02x}{green:02x}00"
            
            if n.state == "SEND": fill_col = "#4fc1ff"
            elif n.state == "VALIDATED": fill_col = "#6a9955"
            elif n.state == "FAILED": fill_col = "#f44747"

            self.canvas.create_rectangle(n.x-14, n.y-14, n.x+14, n.y+14, fill=fill_col, outline="")
            self.canvas.create_image(n.x, n.y, image=self.pc_img, anchor="center")
            self.canvas.create_text(n.x, n.y+20, text=f"PC{n.id}", fill="white", font=("Arial", 8))

if __name__ == "__main__":
    GUI()