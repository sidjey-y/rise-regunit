# Setup Guide for Existing PostgreSQL Database

Since you already have `fingerprint_db` in pgAdmin4, let's configure the system to work with your existing database.

## 🎯 **Quick Setup Steps**

### **Step 1: Create Database User (if not exists)**

In pgAdmin4, connect to your PostgreSQL server and run these SQL commands:

```sql
-- Create user for the application (if not exists)
CREATE USER fingerprint_user WITH PASSWORD 'your_secure_password';

-- Grant privileges to the existing database
GRANT ALL PRIVILEGES ON DATABASE fingerprint_db TO fingerprint_user;

-- Connect to fingerprint_db
\c fingerprint_db

-- Grant schema privileges
GRANT ALL ON SCHEMA public TO fingerprint_user;
GRANT CREATE ON SCHEMA public TO fingerprint_user;
GRANT USAGE ON SCHEMA public TO fingerprint_user;
```

### **Step 2: Update Configuration**

Edit your `config.yaml` file with your actual database credentials:

```yaml
# Database settings
database:
  type: "postgresql"
  host: "localhost"  # or your PostgreSQL server IP
  port: 5432         # or your custom port
  database: "fingerprint_db"
  username: "fingerprint_user"
  password: "your_secure_password"  # Use the password you set above
  table_name: "fingerprints"
  similarity_threshold: 0.85
  max_similarity_score: 0.95
  connection_pool_size: 10
  max_connections: 20
```

### **Step 3: Install Python Dependencies**

```bash
cd Fingerprint
pip install -r requirements.txt
```

### **Step 4: Test the Connection**

```bash
python main.py --mode status
```

## 🔧 **pgAdmin4 Configuration**

### **If you need to find your connection details:**

1. **Open pgAdmin4**
2. **Right-click on your PostgreSQL server**
3. **Select "Properties"**
4. **Check the "Connection" tab for:**
   - Host name/address
   - Port
   - Username

### **Common pgAdmin4 Connection Settings:**

| Setting | Typical Value | Your Value |
|---------|---------------|------------|
| **Host** | localhost | `localhost` or your server IP |
| **Port** | 5432 | `5432` (or your custom port) |
| **Database** | fingerprint_db | `fingerprint_db` |
| **Username** | fingerprint_user | `fingerprint_user` |
| **Password** | your_secure_password | `your_secure_password` |

## 🧪 **Testing Commands**

### **Test 1: System Status**
```bash
cd Fingerprint
python main.py --mode status
```

**Expected Output:**
```
System Status:
  config_loaded: True
  preprocessor_initialized: True
  siamese_network_initialized: True
  database_initialized: True
  database_stats: {'total_fingerprints': 0, 'unique_subjects': 0, 'database_type': 'PostgreSQL', ...}
```

### **Test 2: Process First Fingerprint**
```bash
python main.py --mode process \
    --image "fingerprint_data/IRIS and FINGERPRINT DATASET/fingerprint_data/1/Fingerprint/1__M_Right_thumb_finger.BMP" \
    --subject-id "1" \
    --finger-type "thumb" \
    --hand-side "right"
```

### **Test 3: Check Database in pgAdmin4**
1. **Refresh your database in pgAdmin4**
2. **Look for the new `fingerprints` table**
3. **Check if data was inserted**

## 🚨 **Troubleshooting**

### **Connection Error: "password authentication failed"**
```sql
-- In pgAdmin4, run this SQL:
ALTER USER fingerprint_user WITH PASSWORD 'your_new_password';
```

### **Connection Error: "database does not exist"**
- Make sure you're connecting to the correct database name
- Check if the database name is case-sensitive

### **Permission Error: "permission denied"**
```sql
-- Grant additional permissions:
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO fingerprint_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO fingerprint_user;
```

### **Port Connection Error**
- Check if PostgreSQL is running on the correct port
- In pgAdmin4, check the server properties for the port number

## 📊 **Verify Setup in pgAdmin4**

After running the tests, you should see:

1. **New `fingerprints` table** in your database
2. **Table structure** with columns:
   - `id` (Primary Key)
   - `subject_id` (VARCHAR)
   - `finger_type` (VARCHAR)
   - `hand_side` (VARCHAR)
   - `embedding` (BYTEA)
   - `metadata` (TEXT)
   - `quality_score` (FLOAT)
   - `created_at` (TIMESTAMP)
   - `updated_at` (TIMESTAMP)

3. **Indexes** created automatically:
   - `idx_subject_id`
   - `idx_finger_type`
   - `idx_hand_side`
   - `idx_quality_score`
   - `idx_created_at`
   - `idx_subject_finger_hand`

## 🎉 **Success Indicators**

Your setup is working if:

1. ✅ `python main.py --mode status` shows `database_initialized: True`
2. ✅ Database statistics show `database_type: 'PostgreSQL'`
3. ✅ The `fingerprints` table appears in pgAdmin4
4. ✅ Processing fingerprints stores data successfully
5. ✅ No connection errors in the logs

## 🔄 **Next Steps**

Once your database is connected:

1. **Process your fingerprint dataset:**
```bash
# Process multiple fingerprints
python main.py --mode process --image "path/to/fingerprint1.bmp" --subject-id "1" --finger-type "thumb" --hand-side "right"
python main.py --mode process --image "path/to/fingerprint2.bmp" --subject-id "1" --finger-type "index" --hand-side "right"
```

2. **Test duplicate checking:**
```bash
python main.py --mode compare --image "path/to/fingerprint1.bmp" --image2 "path/to/fingerprint2.bmp"
```

3. **Monitor database growth in pgAdmin4**

Your PostgreSQL database is now ready for fingerprint uniqueness checking! 🎯 