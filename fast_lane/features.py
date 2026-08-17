import time
import math
import json

# Try importing redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

class FeatureStore:
    def __init__(self, host="localhost", port=6379, db=0):
        self.redis_client = None
        self.local_store = {} # Fallback dictionary
        
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host=host, 
                    port=port, 
                    db=db, 
                    decode_responses=True,
                    socket_connect_timeout=1
                )
                self.redis_client.ping()
                print("[+] Connected to Redis feature store.")
            except Exception as e:
                print(f"[-] Redis connection failed: {e}. Using in-memory fallback feature store.")
                self.redis_client = None
        else:
            print("[*] Redis library not found. Using in-memory fallback feature store.")

    def _get_key(self, account_id):
        return f"user:{account_id}:features"

    def get_features(self, account_id):
        """
        Retrieves feature profile for a given account.
        Returns:
            dict containing:
                - avg_amount: float
                - total_tx_count: int
                - last_tx_time: float
                - last_latitude: float
                - last_longitude: float
                - devices: list of device_ids
        """
        if self.redis_client:
            try:
                data = self.redis_client.hgetall(self._get_key(account_id))
                if data:
                    return {
                        "avg_amount": float(data.get("avg_amount", 0.0)),
                        "total_tx_count": int(data.get("total_tx_count", 0)),
                        "last_tx_time": float(data.get("last_tx_time", 0.0)),
                        "last_latitude": float(data.get("last_latitude", 0.0)),
                        "last_longitude": float(data.get("last_longitude", 0.0)),
                        "devices": json.loads(data.get("devices", "[]"))
                    }
            except Exception as e:
                print(f"[-] Redis get failed: {e}")
                
        # Local dict fallback
        data = self.local_store.get(account_id, {})
        return {
            "avg_amount": data.get("avg_amount", 0.0),
            "total_tx_count": data.get("total_tx_count", 0),
            "last_tx_time": data.get("last_tx_time", 0.0),
            "last_latitude": data.get("last_latitude", 0.0),
            "last_longitude": data.get("last_longitude", 0.0),
            "devices": data.get("devices", [])
        }

    def update_features(self, account_id, amount, latitude, longitude, device_id, tx_time=None):
        """
        Updates the features of an account with the latest transaction.
        """
        if tx_time is None:
            tx_time = time.time()
            
        profile = self.get_features(account_id)
        
        # Calculate new average amount
        count = profile["total_tx_count"]
        new_count = count + 1
        new_avg = ((profile["avg_amount"] * count) + amount) / new_count
        
        # Update device list
        devices = profile["devices"]
        if device_id not in devices:
            devices.append(device_id)
            # Cap at last 5 devices
            if len(devices) > 5:
                devices.pop(0)

        updated_data = {
            "avg_amount": str(new_avg),
            "total_tx_count": str(new_count),
            "last_tx_time": str(tx_time),
            "last_latitude": str(latitude),
            "last_longitude": str(longitude),
            "devices": json.dumps(devices) if self.redis_client else devices
        }

        if self.redis_client:
            try:
                self.redis_client.hset(self._get_key(account_id), mapping=updated_data)
                # Keep keys in Redis for 30 days
                self.redis_client.expire(self._get_key(account_id), 30 * 86400)
                return
            except Exception as e:
                print(f"[-] Redis set failed: {e}")

        # Local dict update
        updated_data["devices"] = devices  # Maintain list in dict
        updated_data["avg_amount"] = new_avg
        updated_data["total_tx_count"] = new_count
        updated_data["last_tx_time"] = tx_time
        updated_data["last_latitude"] = latitude
        updated_data["last_longitude"] = longitude
        self.local_store[account_id] = updated_data

def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance in km between two points on the earth.
    """
    if not lat1 or not lon1 or not lat2 or not lon2:
        return 0.0
    
    # Earth radius in kilometers
    R = 6371.0
    
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c
