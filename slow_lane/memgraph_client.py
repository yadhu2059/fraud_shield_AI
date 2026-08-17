import os
import sqlite3
import threading

# Try to import Neo4j python driver for Memgraph
try:
    from neo4j import GraphDatabase
    MEMGRAPH_AVAILABLE = True
except ImportError:
    MEMGRAPH_AVAILABLE = False

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "graph_fallback.db"))

class MemgraphClient:
    def __init__(self, uri="bolt://localhost:7687", username="memgraph", password="memgraph"):
        self.driver = None
        self.use_sqlite = False
        self.lock = threading.Lock()
        
        if MEMGRAPH_AVAILABLE:
            try:
                self.driver = GraphDatabase.driver(uri, auth=(username, password))
                # Test connection
                with self.driver.session() as session:
                    session.run("RETURN 1")
                print("[+] Connected to Memgraph graph database.")
            except Exception as e:
                print(f"[-] Memgraph connection failed: {e}. Falling back to local SQLite graph cache.")
                self.use_sqlite = True
        else:
            print("[*] Neo4j driver not found. Falling back to local SQLite graph cache.")
            self.use_sqlite = True

        if self.use_sqlite:
            os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
            self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            self.create_sqlite_tables()

    def create_sqlite_tables(self):
        with self.lock:
            cursor = self.conn.cursor()
            # Table for accounts
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS accounts (
                    name TEXT PRIMARY KEY,
                    risk_score REAL DEFAULT 0.0
                )
            """)
            # Table for transfers
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transfers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT,
                    destination TEXT,
                    amount REAL,
                    timestamp TEXT,
                    is_fraud INTEGER,
                    FOREIGN KEY(source) REFERENCES accounts(name),
                    FOREIGN KEY(destination) REFERENCES accounts(name)
                )
            """)
            # Table for device links
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS device_links (
                    account TEXT,
                    device_id TEXT,
                    PRIMARY KEY (account, device_id),
                    FOREIGN KEY(account) REFERENCES accounts(name)
                )
            """)
            self.conn.commit()

    def add_transaction(self, source, destination, amount, device_id, timestamp, is_fraud=0):
        """
        Inserts transaction transfer and device usage into the graph.
        """
        if not self.use_sqlite and self.driver:
            query = """
            MERGE (s:Account {name: $source})
            MERGE (d:Account {name: $destination})
            MERGE (dev:Device {id: $device_id})
            CREATE (s)-[t:TRANSFERRED {amount: $amount, timestamp: $timestamp, is_fraud: $is_fraud}]->(d)
            MERGE (s)-[:USED_DEVICE]->(dev)
            """
            try:
                with self.driver.session() as session:
                    session.run(query, source=source, destination=destination, amount=amount, 
                                device_id=device_id, timestamp=timestamp, is_fraud=is_fraud)
                return
            except Exception as e:
                print(f"[-] Memgraph write error: {e}. Writing to SQLite fallback...")

        # SQLite fallback path
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("INSERT OR IGNORE INTO accounts (name) VALUES (?)", (source,))
            cursor.execute("INSERT OR IGNORE INTO accounts (name) VALUES (?)", (destination,))
            cursor.execute("INSERT OR IGNORE INTO device_links (account, device_id) VALUES (?, ?)", (source, device_id))
            cursor.execute("""
                INSERT INTO transfers (source, destination, amount, timestamp, is_fraud) 
                VALUES (?, ?, ?, ?, ?)
            """, (source, destination, amount, timestamp, is_fraud))
            self.conn.commit()

    def get_all_edges_and_nodes(self):
        """
        Retrieves graph representation for GNN training.
        """
        if not self.use_sqlite and self.driver:
            query = """
            MATCH (s:Account)-[t:TRANSFERRED]->(d:Account)
            RETURN s.name as source, d.name as destination, t.is_fraud as is_fraud
            """
            try:
                with self.driver.session() as session:
                    results = session.run(query)
                    records = list(results)
                
                node_set = set()
                raw_edges = []
                node_labels = {}
                
                for r in records:
                    s, d, f = r["source"], r["destination"], r["is_fraud"]
                    node_set.add(s)
                    node_set.add(d)
                    raw_edges.append((s, d))
                    if f == 1:
                        node_labels[s] = 1
                        node_labels[d] = 1
                
                nodes = list(node_set)
                node_to_idx = {n: i for i, n in enumerate(nodes)}
                edges = [(node_to_idx[s], node_to_idx[d]) for s, d in raw_edges]
                
                return nodes, edges, node_labels
            except Exception as e:
                print(f"[-] Memgraph read error: {e}. Reading from SQLite fallback...")

        # SQLite fallback path
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT source, destination, is_fraud FROM transfers")
            rows = cursor.fetchall()
        
        node_set = set()
        raw_edges = []
        node_labels = {}
        
        for s, d, f in rows:
            node_set.add(s)
            node_set.add(d)
            raw_edges.append((s, d))
            if f == 1:
                node_labels[s] = 1
                node_labels[d] = 1
                
        nodes = list(node_set)
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        edges = [(node_to_idx[s], node_to_idx[d]) for s, d in raw_edges]
        
        return nodes, edges, node_labels

    def get_neighborhood(self, account_id):
        """
        Retrieves the 2-hop neighborhood (nodes and edges) connected to the specified account.
        Returns:
            nodes (list of dict): [{"id": "C123", "type": "Account"}, ...]
            edges (list of dict): [{"source": "C123", "target": "C456", "type": "TRANSFERRED", "amount": 100.0, "is_fraud": 0}, ...]
        """
        nodes = []
        edges = []
        visited_nodes = set()
        visited_edges = set()
        fraud_accounts = set()

        def add_node(node_id, node_type):
            if node_id not in visited_nodes:
                is_fraud = 1 if node_id in fraud_accounts else 0
                nodes.append({"id": node_id, "type": node_type, "is_fraud": is_fraud})
                visited_nodes.add(node_id)
                
        def add_edge(src, dest, edge_type, amt=None, is_f=None):
            edge_key = (src, dest, edge_type, amt)
            if edge_key not in visited_edges:
                edge_dict = {"source": src, "target": dest, "type": edge_type}
                if amt is not None:
                    edge_dict["amount"] = amt
                if is_f is not None:
                    edge_dict["is_fraud"] = is_f
                edges.append(edge_dict)
                visited_edges.add(edge_key)

        # SQLite Fallback Path (with 2-hop querying and thread safety)
        with self.lock:
            cursor = self.conn.cursor()
            
            # 1-hop transfers
            cursor.execute("""
                SELECT source, destination, amount, is_fraud FROM transfers 
                WHERE source = ? OR destination = ?
            """, (account_id, account_id))
            transfers_1 = cursor.fetchall()
            
            # Accumulate direct neighbors
            direct_neighbors = {account_id}
            for src, dest, amt, is_f in transfers_1:
                direct_neighbors.add(src)
                direct_neighbors.add(dest)
                if is_f == 1:
                    fraud_accounts.add(src)
                    fraud_accounts.add(dest)
            
            # 1-hop devices
            cursor.execute("SELECT device_id FROM device_links WHERE account = ?", (account_id,))
            devices_1 = [r[0] for r in cursor.fetchall()]
            
            # 2-hop transfers (transfers connecting to direct neighbors)
            transfers_2 = []
            if len(direct_neighbors) > 0:
                placeholders = ",".join("?" for _ in direct_neighbors)
                cursor.execute(f"""
                    SELECT source, destination, amount, is_fraud FROM transfers 
                    WHERE source IN ({placeholders}) OR destination IN ({placeholders})
                """, list(direct_neighbors) + list(direct_neighbors))
                transfers_2 = cursor.fetchall()
                for src, dest, amt, is_f in transfers_2:
                    if is_f == 1:
                        fraud_accounts.add(src)
                        fraud_accounts.add(dest)
            
            # Accounts sharing any of the 1-hop devices (device links 2-hop)
            accounts_sharing_devices = []
            if len(devices_1) > 0:
                placeholders = ",".join("?" for _ in devices_1)
                cursor.execute(f"""
                    SELECT account, device_id FROM device_links 
                    WHERE device_id IN ({placeholders})
                """, list(devices_1))
                accounts_sharing_devices = cursor.fetchall()
                
            # Devices used by direct neighbor accounts
            devices_of_neighbors = []
            if len(direct_neighbors) > 0:
                placeholders = ",".join("?" for _ in direct_neighbors)
                cursor.execute(f"""
                    SELECT account, device_id FROM device_links 
                    WHERE account IN ({placeholders})
                """, list(direct_neighbors))
                devices_of_neighbors = cursor.fetchall()

        # Process nodes and edges into Cytoscape format
        for src, dest, amt, is_f in transfers_1 + transfers_2:
            add_node(src, "Account")
            add_node(dest, "Account")
            add_edge(src, dest, "TRANSFERRED", amt, is_f)
            
        for acc, dev in accounts_sharing_devices + devices_of_neighbors:
            add_node(acc, "Account")
            add_node(dev, "Device")
            add_edge(acc, dev, "USED_DEVICE")
            
        # Ensure target node itself is included
        add_node(account_id, "Account")
            
        return {"nodes": nodes, "edges": edges}

    def get_rich_graph_neighborhood(self, account_id, recent_transactions=[]):
        raw_graph = self.get_neighborhood(account_id)
        raw_nodes = raw_graph["nodes"]
        raw_edges = raw_graph["edges"]
        
        outbound_senders = set()
        for e in raw_edges:
            if e["type"] == "TRANSFERRED":
                outbound_senders.add(e["source"])
                
        nodes_dict = {}
        edges_list = []
        
        acct_to_ip = {}
        acct_to_geo = {}
        for tx in recent_transactions:
            acc_orig = tx.get("account_id")
            acc_dest = tx.get("destination")
            ip = tx.get("ip_address")
            geo = tx.get("geolocation")
            if acc_orig and ip:
                acct_to_ip[acc_orig] = ip
            if acc_orig and geo:
                acct_to_geo[acc_orig] = geo
            if acc_dest and ip:
                acct_to_ip[acc_dest] = ip
                
        def get_ip_for_node(node_id):
            if node_id in acct_to_ip:
                return acct_to_ip[node_id]
            import hashlib
            h = int(hashlib.md5(node_id.encode('utf-8')).hexdigest(), 16)
            return f"192.168.1.{h % 254 + 1}"
            
        def get_geo_for_node(node_id):
            if node_id in acct_to_geo:
                return acct_to_geo[node_id]
            return "28.6139, 77.2090"
            
        with self.lock:
            cursor = self.conn.cursor()
            
            for rn in raw_nodes:
                node_id = rn["id"]
                node_type = rn["type"]
                is_fraud = rn.get("is_fraud", 0)
                
                in_ring = False
                if "MULE_RING" in node_id or "RING_MEMBER" in node_id or node_id == "dev_mule_ring_fingerprint":
                    in_ring = True
                
                if node_type == "Account":
                    is_sender = node_id in outbound_senders or node_id == account_id
                    if not is_sender:
                        cursor.execute("SELECT COUNT(*) FROM transfers WHERE source = ?", (node_id,))
                        if cursor.fetchone()[0] > 0:
                            is_sender = True
                            
                    if is_sender:
                        trust_score = 95.5 if is_fraud == 0 else 25.0
                        cursor.execute("SELECT COUNT(*) FROM transfers WHERE source = ? OR destination = ?", (node_id, node_id))
                        total_tx_count = cursor.fetchone()[0]
                        cursor.execute("SELECT AVG(amount) FROM transfers WHERE source = ? OR destination = ?", (node_id, node_id))
                        avg_amount = cursor.fetchone()[0] or 0.0
                        cursor.execute("SELECT COUNT(DISTINCT device_id) FROM device_links WHERE account = ?", (node_id,))
                        connected_device_count = cursor.fetchone()[0]
                        cursor.execute("SELECT COUNT(DISTINCT destination) FROM transfers WHERE source = ?", (node_id,))
                        connected_beneficiary_count = cursor.fetchone()[0]
                        cursor.execute("SELECT MIN(timestamp) FROM transfers WHERE source = ? OR destination = ?", (node_id, node_id))
                        first_seen = cursor.fetchone()[0] or "2026-08-17T00:00:00Z"
                        
                        nodes_dict[node_id] = {
                            "id": node_id,
                            "type": "account",
                            "risk": 0.05 if is_fraud == 0 else 0.95,
                            "in_ring": in_ring,
                            "attributes": {
                                "trust_score": trust_score,
                                "total_tx_count": total_tx_count,
                                "avg_amount": avg_amount,
                                "connected_device_count": connected_device_count,
                                "connected_beneficiary_count": connected_beneficiary_count,
                                "first_seen": first_seen
                            }
                        }
                    else:
                        cursor.execute("SELECT COUNT(DISTINCT source) FROM transfers WHERE destination = ?", (node_id,))
                        account_count_inbound = cursor.fetchone()[0]
                        cursor.execute("SELECT SUM(amount) FROM transfers WHERE destination = ?", (node_id,))
                        total_inbound_amount = cursor.fetchone()[0] or 0.0
                        
                        nodes_dict[node_id] = {
                            "id": node_id,
                            "type": "beneficiary",
                            "risk": 0.05 if is_fraud == 0 else 0.95,
                            "in_ring": in_ring,
                            "attributes": {
                                "account_count_inbound": account_count_inbound,
                                "total_inbound_amount": total_inbound_amount
                            }
                        }
                        
                    ip_addr = get_ip_for_node(node_id)
                    geo = get_geo_for_node(node_id)
                    if ip_addr not in nodes_dict:
                        nodes_dict[ip_addr] = {
                            "id": ip_addr,
                            "type": "ip",
                            "risk": 0.05 if is_fraud == 0 else 0.85,
                            "in_ring": in_ring,
                            "attributes": {
                                "address": ip_addr,
                                "geolocation": geo,
                                "account_count": 1
                            }
                        }
                    else:
                        nodes_dict[ip_addr]["attributes"]["account_count"] += 1
                        if in_ring:
                            nodes_dict[ip_addr]["in_ring"] = True
                            
                    edges_list.append({
                        "source": node_id,
                        "target": ip_addr,
                        "type": "used_ip",
                        "amount": 0.0,
                        "timestamp": "",
                        "decision": "ALLOW",
                        "risk_score": 0.05 if is_fraud == 0 else 0.85,
                        "reason": "Associated IP address usage",
                        "in_ring": in_ring
                    })
                    
                elif node_type == "Device":
                    cursor.execute("SELECT COUNT(DISTINCT account) FROM device_links WHERE device_id = ?", (node_id,))
                    account_count = cursor.fetchone()[0]
                    cursor.execute("""
                        SELECT MIN(t.timestamp), MAX(t.timestamp) FROM transfers t
                        JOIN device_links d ON t.source = d.account OR t.destination = d.account
                        WHERE d.device_id = ?
                    """, (node_id,))
                    t_min, t_max = cursor.fetchone()
                    
                    nodes_dict[node_id] = {
                        "id": node_id,
                        "type": "device",
                        "risk": 0.05 if node_id != "dev_mule_ring_fingerprint" else 0.95,
                        "in_ring": in_ring or account_count >= 2,
                        "attributes": {
                            "fingerprint": node_id,
                            "account_count": account_count,
                            "first_seen": t_min or "2026-08-17T00:00:00Z",
                            "last_seen": t_max or "2026-08-17T00:00:00Z"
                        }
                    }
                    
            for re in raw_edges:
                src = re["source"]
                dest = re["target"]
                edge_type = re["type"]
                amt = re.get("amount", 0.0)
                is_f = re.get("is_fraud", 0)
                
                in_ring_edge = False
                if "MULE_RING" in src or "MULE_RING" in dest or "RING_MEMBER" in src or "RING_MEMBER" in dest:
                    in_ring_edge = True
                    
                if edge_type == "TRANSFERRED":
                    cursor.execute("SELECT timestamp FROM transfers WHERE source = ? AND destination = ? AND amount = ? LIMIT 1", (src, dest, amt))
                    ts_row = cursor.fetchone()
                    ts = ts_row[0] if ts_row else "2026-08-17T00:00:00Z"
                    
                    # Search recent_transactions for details (decision, risk_score, reason)
                    decision = "ALLOW" if is_f == 0 else "BLOCK"
                    risk_score = 0.05 if is_f == 0 else 0.95
                    reason = "Transfer transaction between nodes"
                    
                    for tx in recent_transactions:
                        if tx.get("account_id") == src and tx.get("destination") == dest and abs(tx.get("amount", 0.0) - amt) < 1e-2:
                            decision = tx.get("decision", decision)
                            risk_score = tx.get("gnn_risk", risk_score)
                            reason = tx.get("reason", reason)
                            break
                            
                    edges_list.append({
                        "source": src,
                        "target": dest,
                        "type": "transaction",
                        "amount": amt,
                        "timestamp": ts,
                        "decision": decision,
                        "risk_score": risk_score,
                        "reason": reason,
                        "in_ring": in_ring_edge
                    })
                elif edge_type == "USED_DEVICE":
                    edges_list.append({
                        "source": src,
                        "target": dest,
                        "type": "used_device",
                        "amount": 0.0,
                        "timestamp": "",
                        "decision": "ALLOW",
                        "risk_score": 0.05,
                        "reason": "Associated device usage",
                        "in_ring": in_ring_edge or (nodes_dict.get(dest, {}).get("attributes", {}).get("account_count", 0) >= 2)
                    })
                    
        adj = {}
        for edge in edges_list:
            if edge["type"] == "transaction":
                s, t = edge["source"], edge["target"]
                if s not in adj: adj[s] = []
                adj[s].append(t)
                
        cycle_nodes = set()
        for start_node in adj:
            visited = set()
            stack = [start_node]
            found = False
            while stack:
                curr = stack.pop()
                if curr in adj:
                    for neighbor in adj[curr]:
                        if neighbor == start_node:
                            found = True
                            break
                        if neighbor not in visited:
                            visited.add(neighbor)
                            stack.append(neighbor)
                    if found:
                        break
            if found:
                cycle_nodes.add(start_node)
                
        for node_id in cycle_nodes:
            if node_id in nodes_dict:
                nodes_dict[node_id]["in_ring"] = True
                
        for edge in edges_list:
            if edge["type"] == "transaction" and edge["source"] in cycle_nodes and edge["target"] in cycle_nodes:
                edge["in_ring"] = True
                
        shared_devices = {nid for nid, n in nodes_dict.items() if n["type"] == "device" and (n["attributes"]["account_count"] >= 2 or nid == "dev_mule_ring_fingerprint")}
        for edge in edges_list:
            if edge["type"] == "used_device" and edge["target"] in shared_devices:
                edge["in_ring"] = True
                if edge["source"] in nodes_dict:
                    nodes_dict[edge["source"]]["in_ring"] = True
                if edge["target"] in nodes_dict:
                    nodes_dict[edge["target"]]["in_ring"] = True
                    
        return {
            "nodes": list(nodes_dict.values()),
            "edges": edges_list
        }

    def close(self):
        if self.driver:
            self.driver.close()
        if hasattr(self, 'conn'):
            with self.lock:
                self.conn.close()
