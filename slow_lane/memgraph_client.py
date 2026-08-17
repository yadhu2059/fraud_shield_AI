import os
import sqlite3

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
        Returns:
            nodes (list of str): unique account names
            edges (list of tuples): (source_index, destination_index)
            labels (dict): map from node name to is_fraud label
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
                
                # Process nodes
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
        Retrieves the neighborhood (nodes and edges) connected to the specified account.
        Returns:
            nodes (list of dict): [{"id": "C123", "type": "Account"}, ...]
            edges (list of dict): [{"source": "C123", "target": "C456", "type": "TRANSFERRED", "amount": 100.0, "is_fraud": 0}, ...]
        """
        nodes = []
        edges = []
        visited_nodes = set()

        if not self.use_sqlite and self.driver:
            # Query Memgraph
            query = """
            MATCH (s:Account {name: $account_id})-[r:TRANSFERRED]-(d:Account)
            RETURN s.name as source, d.name as destination, r.amount as amount, r.is_fraud as is_fraud
            """
            try:
                with self.driver.session() as session:
                    results = session.run(query, account_id=account_id)
                    for r in results:
                        src, dest, amt, is_f = r["source"], r["destination"], r["amount"], r["is_fraud"]
                        if src not in visited_nodes:
                            nodes.append({"id": src, "type": "Account"})
                            visited_nodes.add(src)
                        if dest not in visited_nodes:
                            nodes.append({"id": dest, "type": "Account"})
                            visited_nodes.add(dest)
                        edges.append({"source": src, "target": dest, "type": "TRANSFERRED", "amount": amt, "is_fraud": is_f})
                
                # Also get devices
                query_dev = """
                MATCH (s:Account {name: $account_id})-[r:USED_DEVICE]-(dev:Device)
                RETURN s.name as account, dev.id as device_id
                """
                with self.driver.session() as session:
                    results_dev = session.run(query_dev, account_id=account_id)
                    for r in results_dev:
                        acc, dev = r["account"], r["device_id"]
                        if dev not in visited_nodes:
                            nodes.append({"id": dev, "type": "Device"})
                            visited_nodes.add(dev)
                        edges.append({"source": acc, "target": dev, "type": "USED_DEVICE"})
                return {"nodes": nodes, "edges": edges}
            except Exception as e:
                print(f"[-] Memgraph neighborhood query failed: {e}. Falling back to SQLite...")

        # SQLite Fallback Path
        cursor = self.conn.cursor()
        
        # Get transfers
        cursor.execute("""
            SELECT source, destination, amount, is_fraud FROM transfers 
            WHERE source = ? OR destination = ?
        """, (account_id, account_id))
        rows = cursor.fetchall()
        
        for src, dest, amt, is_f in rows:
            if src not in visited_nodes:
                nodes.append({"id": src, "type": "Account"})
                visited_nodes.add(src)
            if dest not in visited_nodes:
                nodes.append({"id": dest, "type": "Account"})
                visited_nodes.add(dest)
            edges.append({"source": src, "target": dest, "type": "TRANSFERRED", "amount": amt, "is_fraud": is_f})
            
        # Get device links
        cursor.execute("SELECT account, device_id FROM device_links WHERE account = ?", (account_id,))
        dev_rows = cursor.fetchall()
        for acc, dev in dev_rows:
            if dev not in visited_nodes:
                nodes.append({"id": dev, "type": "Device"})
                visited_nodes.add(dev)
            edges.append({"source": acc, "target": dev, "type": "USED_DEVICE"})
            
        # If the account has no transactions yet, ensure it is in the node list
        if account_id not in visited_nodes:
            nodes.append({"id": account_id, "type": "Account"})
            
        return {"nodes": nodes, "edges": edges}

    def close(self):
        if self.driver:
            self.driver.close()
        if hasattr(self, 'conn'):
            self.conn.close()
