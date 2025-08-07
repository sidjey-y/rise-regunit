# PostgreSQL Setup Guide for Fingerprint Uniqueness Checker

This guide will help you set up PostgreSQL for the fingerprint uniqueness checker system.

## 🗄️ **Why PostgreSQL?**

PostgreSQL is an excellent choice for fingerprint databases because:

1. **Enterprise-Grade**: Robust, reliable, and production-ready
2. **Advanced Features**: JSON support, full-text search, indexing
3. **Scalability**: Handles large datasets efficiently
4. **Security**: Built-in security features and user management
5. **Performance**: Optimized for complex queries and large data

## 📋 **Prerequisites**

- Windows, macOS, or Linux
- Python 3.8+ installed
- Administrative access to install PostgreSQL

## 🚀 **Installation Steps**

### **Step 1: Install PostgreSQL**

#### **Windows:**
1. Download PostgreSQL from: https://www.postgresql.org/download/windows/
2. Run the installer
3. Choose default port (5432)
4. Set a master password for the `postgres` user
5. Complete installation

#### **macOS:**
```bash
# Using Homebrew
brew install postgresql
brew services start postgresql

# Or download from: https://www.postgresql.org/download/macosx/
```

#### **Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

### **Step 2: Create Database and User**

#### **Connect to PostgreSQL:**
```bash
# Windows (if added to PATH)
psql -U postgres

# macOS/Linux
sudo -u postgres psql
```

#### **Create Database and User:**
```sql
-- Create database
CREATE DATABASE fingerprint_db;

-- Create user
CREATE USER fingerprint_user WITH PASSWORD 'your_secure_password';

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE fingerprint_db TO fingerprint_user;

-- Connect to the database
\c fingerprint_db

-- Grant schema privileges
GRANT ALL ON SCHEMA public TO fingerprint_user;

-- Exit
\q
```

### **Step 3: Update Configuration**

Edit `config.yaml` in the Fingerprint directory:

```yaml
# Database settings
database:
  type: "postgresql"
  host: "localhost"
  port: 5432
  database: "fingerprint_db"
  username: "fingerprint_user"
  password: "your_secure_password"  # Use the password you set
  table_name: "fingerprints"
  similarity_threshold: 0.85
  max_similarity_score: 0.95
  connection_pool_size: 10
  max_connections: 20
```

### **Step 4: Install Python Dependencies**

```bash
cd Fingerprint
pip install -r requirements.txt
```

## 🔧 **Configuration Options**

### **Connection Settings:**
- **host**: Database server address (localhost for local installation)
- **port**: PostgreSQL port (default: 5432)
- **database**: Database name
- **username**: Database user
- **password**: Database password

### **Performance Settings:**
- **connection_pool_size**: Number of connections in the pool (10-20 recommended)
- **max_connections**: Maximum total connections (20-50 recommended)

### **Security Settings:**
- **similarity_threshold**: Threshold for duplicate detection (0.85 recommended)
- **max_similarity_score**: Maximum similarity score (0.95 recommended)

## 🧪 **Testing the Connection**

### **Test 1: Check System Status**
```bash
python main.py --mode status
```

Expected output:
```
System Status:
  config_loaded: True
  preprocessor_initialized: True
  siamese_network_initialized: True
  database_initialized: True
  database_stats: {'total_fingerprints': 0, 'unique_subjects': 0, 'database_type': 'PostgreSQL', ...}
```

### **Test 2: Process a Fingerprint**
```bash
python main.py --mode process \
    --image "fingerprint_data/IRIS and FINGERPRINT DATASET/fingerprint_data/1/Fingerprint/1__M_Right_thumb_finger.BMP" \
    --subject-id "1" \
    --finger-type "thumb" \
    --hand-side "right"
```

## 🔒 **Security Best Practices**

### **1. Strong Passwords**
- Use complex passwords (12+ characters)
- Include uppercase, lowercase, numbers, and symbols
- Change passwords regularly

### **2. Network Security**
- Use SSL/TLS connections
- Restrict database access to specific IP addresses
- Use firewall rules

### **3. User Permissions**
- Create dedicated users with minimal required privileges
- Don't use the postgres superuser for applications
- Regularly audit user permissions

### **4. Database Security**
```sql
-- Enable SSL (in postgresql.conf)
ssl = on

-- Restrict connections (in pg_hba.conf)
host    fingerprint_db    fingerprint_user    127.0.0.1/32    md5
```

## 📊 **Performance Optimization**

### **1. Indexing**
The system automatically creates indexes for:
- `subject_id`
- `finger_type`
- `hand_side`
- `quality_score`
- `created_at`
- Composite index on `(subject_id, finger_type, hand_side)`

### **2. Connection Pooling**
- Configured with SQLAlchemy QueuePool
- Automatic connection recycling
- Pre-ping for connection health

### **3. Query Optimization**
- Uses SQLAlchemy ORM for efficient queries
- Prepared statements for security
- Batch operations for better performance

## 🚨 **Troubleshooting**

### **Connection Errors:**
```bash
# Check if PostgreSQL is running
# Windows
services.msc  # Look for "postgresql-x64-15" service

# macOS
brew services list | grep postgresql

# Linux
sudo systemctl status postgresql
```

### **Permission Errors:**
```sql
-- Grant additional permissions if needed
GRANT CREATE ON SCHEMA public TO fingerprint_user;
GRANT USAGE ON SCHEMA public TO fingerprint_user;
```

### **Port Conflicts:**
```bash
# Check if port 5432 is in use
netstat -an | grep 5432

# Change port in postgresql.conf if needed
port = 5433
```

## 📈 **Monitoring and Maintenance**

### **Database Statistics:**
```bash
python main.py --mode status
```

### **PostgreSQL Monitoring:**
```sql
-- Check active connections
SELECT * FROM pg_stat_activity;

-- Check database size
SELECT pg_size_pretty(pg_database_size('fingerprint_db'));

-- Check table statistics
SELECT schemaname, tablename, attname, n_distinct, correlation 
FROM pg_stats 
WHERE tablename = 'fingerprints';
```

### **Backup and Recovery:**
```bash
# Create backup
pg_dump -U fingerprint_user -d fingerprint_db > backup.sql

# Restore backup
psql -U fingerprint_user -d fingerprint_db < backup.sql
```

## 🎉 **Success Indicators**

Your PostgreSQL setup is working correctly if:

1. ✅ `python main.py --mode status` shows database_initialized: True
2. ✅ Database statistics show database_type: 'PostgreSQL'
3. ✅ Processing fingerprints stores data successfully
4. ✅ Duplicate checking works correctly
5. ✅ No connection errors in logs

## 🔄 **Migration from SQLite**

If you were previously using SQLite:

1. **Export SQLite data:**
```bash
# Use SQLite browser or command line to export data
```

2. **Import to PostgreSQL:**
```bash
# Use pg_dump/pg_restore or custom migration script
```

3. **Update configuration:**
```yaml
database:
  type: "postgresql"  # Changed from "sqlite"
  # ... other PostgreSQL settings
```

PostgreSQL provides enterprise-grade reliability and performance for your fingerprint uniqueness checking system! 