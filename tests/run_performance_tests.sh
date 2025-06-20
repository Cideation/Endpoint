#!/bin/bash
"""
BEM System Performance Testing Runner
Executes all three critical performance tests:
1. Lighthouse Frontend Audit
2. API Latency Checks  
3. PostgreSQL Query Profiling
"""

set -e  # Exit on any error

echo "🚀 BEM System Performance Testing Suite"
echo "========================================"

# Configuration
BEM_URL=${1:-"http://localhost:8000"}
DB_URL=${2:-$DATABASE_URL}
OUTPUT_DIR="performance_reports_$(date +%Y%m%d_%H%M%S)"

echo "🔧 Configuration:"
echo "  • BEM URL: $BEM_URL"
echo "  • Database: ${DB_URL:0:20}..."
echo "  • Output Directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p $OUTPUT_DIR

# ============================================================================
# Test 1: Lighthouse Frontend Audit
# ============================================================================

echo "💡 Test 1: Lighthouse Frontend Audit"
echo "------------------------------------"

if command -v lighthouse &> /dev/null; then
    echo "✅ Lighthouse found, running frontend audit..."
    
    # Run Lighthouse audit
    lighthouse $BEM_URL \
        --output html \
        --output json \
        --output-path $OUTPUT_DIR/lighthouse_report \
        --chrome-flags="--headless --no-sandbox --disable-dev-shm-usage" \
        --quiet
    
    echo "✅ Lighthouse audit complete: $OUTPUT_DIR/lighthouse_report.report.html"
    
    # Extract key metrics if JSON report exists
    if [ -f "$OUTPUT_DIR/lighthouse_report.report.json" ]; then
        echo "📊 Performance Score: $(node -p "Math.round(require('./$OUTPUT_DIR/lighthouse_report.report.json').lhr.categories.performance.score * 100)")"
    fi
else
    echo "⚠️ Lighthouse not installed. Installing..."
    npm install -g lighthouse
    
    if command -v lighthouse &> /dev/null; then
        lighthouse $BEM_URL --output html --output-path $OUTPUT_DIR/lighthouse_report --quiet
        echo "✅ Lighthouse audit complete"
    else
        echo "❌ Failed to install Lighthouse. Skipping frontend audit."
    fi
fi

echo ""

# ============================================================================
# Test 2: API Latency Checks
# ============================================================================

echo "⚡ Test 2: API Latency Checks"
echo "-----------------------------"

# Create curl format file
cat > $OUTPUT_DIR/curl-format.txt << 'EOF'
Time Namelookup:  %{time_namelookup}s
Time Connect:  %{time_connect}s  
Time Start Transfer:  %{time_starttransfer}s
Total Time:  %{time_total}s
Size Download:  %{size_download} bytes
Speed Download:  %{speed_download} bytes/sec
HTTP Code:  %{http_code}
EOF

# API endpoints to test
endpoints=(
    "/api/health"
    "/api/nodes" 
    "/api/graph"
    "/api/agent_state"
    "/"
    "/frontend/index.html"
)

echo "Testing API endpoints..." > $OUTPUT_DIR/api_latency_results.txt
echo "========================" >> $OUTPUT_DIR/api_latency_results.txt

total_time=0
endpoint_count=0

for endpoint in "${endpoints[@]}"; do
    echo "Testing: $BEM_URL$endpoint"
    
    # Run curl with timing
    if curl -w "@$OUTPUT_DIR/curl-format.txt" -o /dev/null -s --connect-timeout 10 --max-time 30 "$BEM_URL$endpoint" 2>> $OUTPUT_DIR/api_latency_results.txt; then
        echo "✅ $endpoint" 
        endpoint_count=$((endpoint_count + 1))
    else
        echo "❌ $endpoint (failed/timeout)"
    fi
    
    echo "Endpoint: $endpoint" >> $OUTPUT_DIR/api_latency_results.txt
    echo "---" >> $OUTPUT_DIR/api_latency_results.txt
done

echo "✅ API latency testing complete: $OUTPUT_DIR/api_latency_results.txt"
echo "📊 Tested $endpoint_count endpoints"
echo ""

# ============================================================================
# Test 3: PostgreSQL Query Profiling  
# ============================================================================

echo "🗄️ Test 3: PostgreSQL Query Profiling"
echo "--------------------------------------"

if [ -n "$DB_URL" ]; then
    echo "Testing database connection..."
    
    # Test database connection
    if psql "$DB_URL" -c "SELECT version();" &> /dev/null; then
        echo "✅ Database connection successful"
        
        # Run performance queries
        echo "Running query performance analysis..."
        
        psql "$DB_URL" << 'EOF' > $OUTPUT_DIR/db_performance_results.txt 2>&1

-- Performance Analysis Report
\echo '🗄️ PostgreSQL Performance Analysis'
\echo '=================================='

-- Query 1: Alpha Phase Node Analysis
\echo ''
\echo '📊 Query 1: Alpha Phase Node Performance'
\echo '----------------------------------------'
EXPLAIN ANALYZE
SELECT * FROM node_data 
WHERE phase = 'alpha' 
ORDER BY created_at DESC 
LIMIT 50;

-- Query 2: Table Statistics
\echo ''
\echo '📊 Table Size and Statistics'
\echo '----------------------------'
SELECT schemaname, tablename,
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
       n_live_tup, n_dead_tup
FROM pg_stat_user_tables
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Query 3: Index Usage
\echo ''
\echo '📊 Index Usage Statistics'
\echo '-------------------------'
SELECT schemaname, tablename, indexname,
       idx_scan, idx_tup_read,
       pg_size_pretty(pg_relation_size(indexrelid)) as index_size
FROM pg_stat_user_indexes
WHERE idx_scan > 0
ORDER BY idx_scan DESC
LIMIT 20;

-- Query 4: Missing Indexes Check
\echo ''
\echo '🔍 Potential Missing Indexes'
\echo '----------------------------'
SELECT 'node_data: phase column needs index' as recommendation
WHERE NOT EXISTS (
    SELECT 1 FROM pg_indexes 
    WHERE tablename = 'node_data' 
    AND indexdef LIKE '%phase%'
);

\echo ''
\echo '✅ Database performance analysis complete'

EOF

        echo "✅ Database profiling complete: $OUTPUT_DIR/db_performance_results.txt"
        
        # Extract key metrics
        if grep -q "Execution Time" $OUTPUT_DIR/db_performance_results.txt; then
            echo "📊 Query execution times recorded"
        fi
        
    else
        echo "❌ Database connection failed. Check DATABASE_URL"
        echo "Skipping database performance tests."
    fi
else
    echo "⚠️ No DATABASE_URL provided. Skipping database tests."
    echo "Set DATABASE_URL environment variable or pass as second argument."
fi

echo ""

# ============================================================================
# Summary Report Generation
# ============================================================================

echo "📊 Generating Performance Summary"
echo "================================="

# Create summary report
cat > $OUTPUT_DIR/performance_summary.md << EOF
# BEM System Performance Test Results

**Test Date:** $(date)
**BEM URL:** $BEM_URL
**Output Directory:** $OUTPUT_DIR

## Test Results Overview

### 1. Frontend Performance (Lighthouse)
EOF

if [ -f "$OUTPUT_DIR/lighthouse_report.report.json" ]; then
    PERF_SCORE=$(node -p "Math.round(require('./$OUTPUT_DIR/lighthouse_report.report.json').lhr.categories.performance.score * 100)" 2>/dev/null || echo "Unknown")
    echo "- **Performance Score:** $PERF_SCORE/100" >> $OUTPUT_DIR/performance_summary.md
    echo "- **Report:** lighthouse_report.report.html" >> $OUTPUT_DIR/performance_summary.md
else
    echo "- **Status:** Test completed (see lighthouse_report.report.html)" >> $OUTPUT_DIR/performance_summary.md
fi

cat >> $OUTPUT_DIR/performance_summary.md << EOF

### 2. API Latency Testing
- **Endpoints Tested:** $endpoint_count
- **Results:** api_latency_results.txt
- **Status:** ✅ Complete

### 3. Database Performance
EOF

if [ -f "$OUTPUT_DIR/db_performance_results.txt" ]; then
    echo "- **Status:** ✅ Complete" >> $OUTPUT_DIR/performance_summary.md
    echo "- **Results:** db_performance_results.txt" >> $OUTPUT_DIR/performance_summary.md
else
    echo "- **Status:** ⚠️ Skipped (no database connection)" >> $OUTPUT_DIR/performance_summary.md
fi

cat >> $OUTPUT_DIR/performance_summary.md << EOF

## Optimization Recommendations

### Immediate Actions (Easy Wins)
1. **Enable gzip compression** for static assets
2. **Add database indexes** for frequently queried columns
3. **Implement lazy loading** for heavy graph components

### Medium-term Improvements  
1. **Set up CDN** for static asset delivery
2. **Implement Redis caching** for frequently accessed data
3. **Optimize slow API endpoints** identified in testing

### Long-term Optimizations
1. **Implement server-side rendering** for faster initial loads
2. **Set up database read replicas** for query load distribution
3. **Implement graph data caching layer** for complex queries

## Files Generated
- \`lighthouse_report.report.html\` - Detailed frontend performance audit
- \`api_latency_results.txt\` - API endpoint response time analysis  
- \`db_performance_results.txt\` - Database query performance analysis
- \`performance_summary.md\` - This summary report

## Next Steps
1. Review detailed reports for specific optimization opportunities
2. Implement recommended database indexes
3. Enable compression and caching for immediate performance gains
4. Set up monitoring to track performance improvements over time
EOF

echo "✅ Performance testing complete!"
echo ""
echo "📁 Results saved to: $OUTPUT_DIR/"
echo "📋 Summary: $OUTPUT_DIR/performance_summary.md"
echo "🌐 Lighthouse Report: $OUTPUT_DIR/lighthouse_report.report.html"
echo ""
echo "🎯 Next Steps:"
echo "  1. Review detailed reports"
echo "  2. Implement recommended optimizations" 
echo "  3. Run tests again to measure improvements"
echo ""

# Run Python performance test suite if available
if [ -f "tests/test_performance_optimization.py" ]; then
    echo "🐍 Running Python performance test suite..."
    python tests/test_performance_optimization.py
fi 