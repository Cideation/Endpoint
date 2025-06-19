// CAD Parser Frontend JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Initialize the application
    initializeApp();
});

function initializeApp() {
    // Set up event listeners
    setupFileUpload();
    setupParseButton();
    setupPushButton();
    setupDataView();
    
    // Load initial data
    loadDatabaseData();
}

function setupFileUpload() {
    const fileInput = document.getElementById('fileInput');
    const fileList = document.getElementById('fileList');
    
    fileInput.addEventListener('change', function(e) {
        fileList.innerHTML = '';
        const files = Array.from(e.target.files);
        
        files.forEach(file => {
            const fileItem = document.createElement('div');
            fileItem.className = 'file-item';
            fileItem.innerHTML = `
                <span class="file-name">${file.name}</span>
                <span class="file-size">${(file.size / 1024).toFixed(1)} KB</span>
            `;
            fileList.appendChild(fileItem);
        });
    });
}

function setupParseButton() {
    const parseBtn = document.getElementById('parseBtn');
    parseBtn.addEventListener('click', parseFiles);
}

function setupPushButton() {
    const pushBtn = document.getElementById('pushBtn');
    pushBtn.addEventListener('click', pushToPostgreSQL);
}

function setupDataView() {
    const refreshBtn = document.getElementById('refreshBtn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', loadDatabaseData);
    }
}

async function parseFiles() {
  const fileInput = document.getElementById('fileInput');
    const files = fileInput.files;

    if (files.length === 0) {
        alert('Please select files to parse');
    return;
  }

    const parseBtn = document.getElementById('parseBtn');
    parseBtn.disabled = true;
    parseBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Parsing...';
    
    try {
        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            const formData = new FormData();
            formData.append('file', file);
            
            const response = await fetch('/parse', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Parsing failed');
            }
            
            const result = await response.json();
            displayParseResult(result, file.name);
        }
        
        alert('✅ All files parsed successfully!');
        
    } catch (error) {
        console.error('Parse error:', error);
        alert(`❌ Parse failed: ${error.message}`);
    } finally {
        parseBtn.disabled = false;
        parseBtn.innerHTML = '<i class="fas fa-cogs"></i> Parse Files';
    }
}

function displayParseResult(result, filename) {
    const resultsContainer = document.getElementById('parseResults');
    if (!resultsContainer) return;
    
    const resultDiv = document.createElement('div');
    resultDiv.className = 'parse-result';
    resultDiv.innerHTML = `
        <h4>${filename}</h4>
        <pre>${JSON.stringify(result, null, 2)}</pre>
    `;
    resultsContainer.appendChild(resultDiv);
}

async function pushToPostgreSQL() {
    const pushBtn = document.getElementById('pushBtn');
    pushBtn.disabled = true;
    pushBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Pushing...';
    
    try {
        // Get parsed data (you might want to store this from parse results)
        const sampleData = {
            component_id: `COMP_${Date.now()}`,
            component_type: 'test_component',
            properties: {
                name: 'Test Component',
                material: 'Steel',
                dimensions: { width: 100, height: 200, depth: 50 }
            },
            geometry: {
                type: 'box',
                vertices: [[0,0,0], [100,0,0], [100,200,0], [0,200,0]],
                faces: [[0,1,2,3]]
            },
            metadata: {
                source: 'test_upload',
                timestamp: new Date().toISOString()
            }
        };
        
        const response = await fetch('/push', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(sampleData)
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Push failed');
        }
        
        const result = await response.json();
        alert(`✅ Pushed to PostgreSQL: ${result.component_id}`);
        
        // Refresh data view
        loadDatabaseData();
        
    } catch (error) {
        console.error('Push error:', error);
        alert(`❌ Push failed: ${error.message}`);
    } finally {
        pushBtn.disabled = false;
        pushBtn.innerHTML = '<i class="fas fa-database"></i> Push to PostgreSQL';
}
}

async function loadDatabaseData() {
    const dataContainer = document.getElementById('databaseData');
    if (!dataContainer) return;
    
    try {
        const response = await fetch('/db_data');
        if (!response.ok) {
            throw new Error('Failed to load data');
        }
        
        const result = await response.json();
        
        if (result.success && result.data) {
            displayDatabaseData(result.data);
        } else {
            dataContainer.innerHTML = '<p>No data available</p>';
  }
        
    } catch (error) {
        console.error('Load data error:', error);
        dataContainer.innerHTML = `<p>Error loading data: ${error.message}</p>`;
    }
}

function displayDatabaseData(data) {
    const dataContainer = document.getElementById('databaseData');
    if (!dataContainer) return;
    
    if (data.length === 0) {
        dataContainer.innerHTML = '<p>No components in database</p>';
    return;
  }

    const table = document.createElement('table');
    table.className = 'data-table';
    
    // Create header
    const header = document.createElement('thead');
    header.innerHTML = `
        <tr>
            <th>ID</th>
            <th>Component ID</th>
            <th>Type</th>
            <th>Created</th>
            <th>Actions</th>
        </tr>
    `;
    table.appendChild(header);
    
    // Create body
    const body = document.createElement('tbody');
    data.forEach(item => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${item.id}</td>
            <td>${item.component_id || 'N/A'}</td>
            <td>${item.component_type || 'N/A'}</td>
            <td>${new Date(item.created_at).toLocaleString()}</td>
            <td>
                <button onclick="viewDetails(${item.id})" class="btn-small">
                    <i class="fas fa-eye"></i> View
                </button>
            </td>
        `;
        body.appendChild(row);
    });
    table.appendChild(body);
    
    dataContainer.innerHTML = '';
    dataContainer.appendChild(table);
}

function viewDetails(id) {
    // Implement detailed view functionality
    alert(`View details for component ID: ${id}`);
}

// Health check function
async function checkHealth() {
    try {
        const response = await fetch('/health');
        const health = await response.json();
        
        const statusElement = document.getElementById('healthStatus');
        if (statusElement) {
            statusElement.innerHTML = `
                <span class="status-indicator ${health.postgresql_status === 'Connected' ? 'online' : 'offline'}"></span>
                PostgreSQL: ${health.postgresql_status}
            `;
        }
        
        return health;
    } catch (error) {
        console.error('Health check failed:', error);
        return null;
    }
}

// Auto-refresh health status
setInterval(checkHealth, 30000); // Check every 30 seconds