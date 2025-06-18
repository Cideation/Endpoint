# Neo4j Aura Integration Setup Guide

## 🎯 Overview
This guide helps you set up the hybrid data pipeline connecting Neon PostgreSQL to Neo4j Aura for graph processing.

## 📁 Folder Structure
```
aura/
├── aura_integration.py      # Main ETL/Sync engine
├── aura_config.py          # Aura credentials & settings
├── requirements_aura.txt   # Python dependencies
├── AURA_SETUP_GUIDE.md    # This guide
└── README.md              # Quick reference
```

## 🚀 Quick Setup

### 1. Install Dependencies
```bash
cd aura
pip install -r requirements_aura.txt
```

### 2. Configure Aura Credentials
Edit `aura_config.py` with your Neo4j Aura details:
```python
AURA_CONFIG = {
    'uri': 'neo4j+s://your-instance.databases.neo4j.io:7687',
    'username': 'neo4j',
    'password': 'your-actual-password'
}
```

### 3. Run Integration
```bash
python aura_integration.py
```

## 🔧 Detailed Configuration

### Aura Database Setup
1. **Create Aura Instance** at [neo4j.com/aura](https://neo4j.com/aura)
2. **Get Connection Details** from your Aura dashboard
3. **Update `aura_config.py`** with your credentials

### Neon PostgreSQL Setup
Ensure your Neon database has:
- Enhanced schema tables (from `postgre/schema_enrichment.py`)
- Component data loaded
- Relationship tables populated

## 📊 Data Flow Pipeline

```
┌─────────────────────┐
│  📂 CSV / Raw Input │
└─────────┬───────────┘
          ↓
┌─────────────────────────┐
│  🗃 PostgreSQL (NEON)    │
│  - Tables: nodes, A/B/G │
│  - Truth + validation   │
└─────────┬───────────────┘
          ↓
┌────────────────────────────────┐
│  🐍 Python ETL / Sync Engine   │
│  - Reads from Postgres         │
│  - Builds Cypher               │
│  - Pushes to Neo4j Aura        │
└────────────┬──────────────────┘
             ↓
┌────────────────────────┐
│  🧠 Neo4j Aura          │
│  - Live graph model    │
│  - Executes Cypher     │
│  - UI + Pulse + DGL in │
└─────────┬──────────────┘
          ↓
┌────────────────────────────┐
│  📘 DGL / PyTorch / GNN     │
│  - Learns from graph       │
│  - Predicts edge weights   │
│  - Calculates node roles   │
└─────────┬──────────────────┘
          ↓
┌────────────────────────────┐
│  📥 Writeback to Postgres  │
│  - Table: `graph_edge_results` │
│  - Stores weights, confidence │
└────────────────────────────┘
```

## 🔄 Sync Process

### 1. **Data Extraction** (Neon → Python)
- Reads from enhanced PostgreSQL tables
- Handles data type conversions
- Validates data integrity

### 2. **Cypher Generation** (Python)
- Builds node creation queries
- Creates relationship mappings
- Handles batch processing

### 3. **Graph Population** (Python → Aura)
- Executes Cypher queries in batches
- Creates indexes for performance
- Validates graph structure

### 4. **Analytics Processing** (Aura)
- Runs graph algorithms
- Calculates node centrality
- Identifies communities

### 5. **Results Writeback** (Aura → Neon)
- Stores analytics results
- Updates edge weights
- Maintains audit trail

## 📈 Available Analytics

### Graph Metrics
- **Component Count**: Total nodes in graph
- **Relationship Count**: Total edges
- **Node Degrees**: Connection density
- **Connected Components**: Graph structure

### Advanced Algorithms
- **PageRank**: Node importance
- **Betweenness Centrality**: Bridge nodes
- **Community Detection**: Clusters
- **Shortest Paths**: Routing optimization

## 🛠 Customization

### Adding New Node Types
```python
# In aura_integration.py
def sync_material_data(self):
    materials_df = self.read_from_neon('material_enhanced')
    node_queries = self.build_cypher_nodes(
        materials_df, 'Material', 'material_id'
    )
    return self.execute_cypher_batch(node_queries)
```

### Adding New Relationships
```python
# In aura_config.py
'relationship_mappings': {
    'supplier_components': {
        'source_label': 'Supplier',
        'target_label': 'Component',
        'relationship_type': 'SUPPLIES',
        'source_id_field': 'supplier_id',
        'target_id_field': 'component_id'
    }
}
```

## 🔍 Monitoring & Debugging

### Log Levels
- **INFO**: General progress
- **WARNING**: Non-critical issues
- **ERROR**: Failed operations
- **DEBUG**: Detailed execution

### Performance Metrics
- Sync duration
- Records processed
- Query execution time
- Memory usage

## 🚨 Troubleshooting

### Common Issues

**Connection Failed**
```bash
# Check credentials in aura_config.py
# Verify network connectivity
# Test with Neo4j Browser
```

**Data Sync Issues**
```bash
# Check Neon table structure
# Verify data types match
# Review error logs
```

**Performance Problems**
```bash
# Reduce batch size
# Add database indexes
# Monitor memory usage
```

## 🔮 Future Enhancements

### Planned Features
- **Real-time Sync**: Event-driven updates
- **Incremental Loading**: Delta processing
- **Graph Visualization**: Interactive UI
- **ML Integration**: DGL/PyTorch models

### Advanced Analytics
- **Link Prediction**: Future relationships
- **Anomaly Detection**: Unusual patterns
- **Recommendation Engine**: Similar components
- **Optimization Algorithms**: Resource allocation

## 📞 Support

For issues or questions:
1. Check the logs in `aura_integration.py`
2. Verify configuration in `aura_config.py`
3. Test connections individually
4. Review Neo4j Aura documentation

---

**Your hybrid Neon + Aura architecture is ready for advanced graph processing! 🎉** 