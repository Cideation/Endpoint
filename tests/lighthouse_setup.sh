#!/bin/bash
"""
Lighthouse Setup Script for BEM System Performance Testing
Installs Lighthouse CLI and configures for automated frontend audits
"""

echo "🚀 Setting up Lighthouse for BEM System Performance Testing"

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required but not installed"
    echo "Please install Node.js from https://nodejs.org/"
    exit 1
fi

echo "✅ Node.js version: $(node --version)"

# Install Lighthouse globally
echo "📦 Installing Lighthouse CLI globally..."
npm install -g lighthouse

# Verify installation
if ! command -v lighthouse &> /dev/null; then
    echo "❌ Lighthouse installation failed"
    exit 1
fi

echo "✅ Lighthouse version: $(lighthouse --version)"

# Create curl format file for API latency testing
echo "📝 Creating curl format file for API latency testing..."
cat > curl-format.txt << 'EOF'
Time Namelookup:  %{time_namelookup}s
Time Connect:  %{time_connect}s
Time Start Transfer:  %{time_starttransfer}s
Total Time:  %{time_total}s
Size Download:  %{size_download} bytes
Speed Download:  %{speed_download} bytes/sec
HTTP Code:  %{http_code}
EOF

# Create Lighthouse configuration file
echo "⚙️ Creating Lighthouse configuration..."
cat > lighthouse-config.js << 'EOF'
module.exports = {
  extends: 'lighthouse:default',
  settings: {
    onlyAudits: [
      'first-contentful-paint',
      'largest-contentful-paint',
      'cumulative-layout-shift',
      'total-blocking-time',
      'speed-index',
      'interactive',
      'unused-javascript',
      'render-blocking-resources',
      'unminified-css',
      'unminified-javascript',
      'uses-text-compression',
      'uses-optimized-images'
    ],
  },
  audits: [
    'lighthouse/audits/metrics/first-contentful-paint',
    'lighthouse/audits/metrics/largest-contentful-paint',
    'lighthouse/audits/metrics/cumulative-layout-shift',
    'lighthouse/audits/byte-efficiency/unused-javascript',
    'lighthouse/audits/byte-efficiency/render-blocking-resources'
  ],
  categories: {
    performance: {
      title: 'BEM System Performance',
      description: 'Performance metrics for BEM system frontend',
      auditRefs: [
        {id: 'first-contentful-paint', weight: 10, group: 'metrics'},
        {id: 'largest-contentful-paint', weight: 25, group: 'metrics'},
        {id: 'cumulative-layout-shift', weight: 15, group: 'metrics'},
        {id: 'total-blocking-time', weight: 30, group: 'metrics'},
        {id: 'speed-index', weight: 10, group: 'metrics'},
        {id: 'interactive', weight: 10, group: 'metrics'},
        {id: 'unused-javascript', weight: 0, group: 'load-opportunities'},
        {id: 'render-blocking-resources', weight: 0, group: 'load-opportunities'},
        {id: 'unminified-css', weight: 0, group: 'load-opportunities'},
        {id: 'unminified-javascript', weight: 0, group: 'load-opportunities'},
        {id: 'uses-text-compression', weight: 0, group: 'load-opportunities'},
        {id: 'uses-optimized-images', weight: 0, group: 'load-opportunities'}
      ]
    }
  }
};
EOF

# Create performance testing script
echo "🔧 Creating performance testing script..."
cat > run_lighthouse_audit.sh << 'EOF'
#!/bin/bash

# BEM System Lighthouse Audit Script
URL=${1:-"http://localhost:8000"}
OUTPUT_DIR=${2:-"./lighthouse-reports"}

echo "🚀 Running Lighthouse audit on $URL"

# Create output directory
mkdir -p $OUTPUT_DIR

# Run Lighthouse audit
lighthouse $URL \
  --config-path=./lighthouse-config.js \
  --output html \
  --output json \
  --output-path $OUTPUT_DIR/bem-performance-report \
  --chrome-flags="--headless --no-sandbox --disable-dev-shm-usage" \
  --quiet

echo "✅ Lighthouse audit complete!"
echo "📊 Reports saved to $OUTPUT_DIR/"
echo "🌐 Open $OUTPUT_DIR/bem-performance-report.report.html to view results"

# Extract key metrics from JSON report
if [ -f "$OUTPUT_DIR/bem-performance-report.report.json" ]; then
    echo ""
    echo "📈 Key Performance Metrics:"
    node -e "
    const report = require('./$OUTPUT_DIR/bem-performance-report.report.json');
    const audits = report.lhr.audits;
    
    console.log('Performance Score:', Math.round(report.lhr.categories.performance.score * 100));
    console.log('First Contentful Paint:', audits['first-contentful-paint'].displayValue);
    console.log('Largest Contentful Paint:', audits['largest-contentful-paint'].displayValue);
    console.log('Cumulative Layout Shift:', audits['cumulative-layout-shift'].displayValue);
    console.log('Time to Interactive:', audits['interactive'].displayValue);
    
    console.log('\\n🔧 Optimization Opportunities:');
    if (audits['unused-javascript'].details.items.length > 0) {
        console.log('- Remove unused JavaScript:', audits['unused-javascript'].displayValue, 'potential savings');
    }
    if (audits['render-blocking-resources'].details.items.length > 0) {
        console.log('- Eliminate render-blocking resources:', audits['render-blocking-resources'].displayValue, 'potential savings');
    }
    if (audits['uses-text-compression'].score < 1) {
        console.log('- Enable text compression:', audits['uses-text-compression'].displayValue, 'potential savings');
    }
    "
fi
EOF

chmod +x run_lighthouse_audit.sh

echo ""
echo "✅ Lighthouse setup complete!"
echo ""
echo "🚀 Usage Examples:"
echo "  # Run audit on localhost"
echo "  ./run_lighthouse_audit.sh"
echo ""
echo "  # Run audit on live URL"
echo "  ./run_lighthouse_audit.sh https://yourdomain.com"
echo ""
echo "  # Run API latency check"
echo "  curl -w \"@curl-format.txt\" -o /dev/null -s http://localhost:8000/api/health"
echo ""
echo "📊 Run the full performance test suite:"
echo "  python tests/test_performance_optimization.py"
echo "" 