const API_URL = 'http://localhost:5000';
let statusCheckInterval = null;
let selectedModel = 'ensemble';

// Initialize
window.onload = function() {
    loadStores();
    setupModelSelector();
    loadMetrics();
};

// Model Selector
function setupModelSelector() {
    const modelBtns = document.querySelectorAll('.model-btn');
    modelBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            modelBtns.forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            selectedModel = this.dataset.model;
        });
    });
}

// Tab Switching
function switchTab(tabName) {
    // Update tab buttons
    const tabs = document.querySelectorAll('.tab');
    tabs.forEach(tab => tab.classList.remove('active'));
    event.target.classList.add('active');
    
    // Update tab content
    const contents = document.querySelectorAll('.tab-content');
    contents.forEach(content => content.classList.remove('active'));
    document.getElementById(`${tabName}-tab`).classList.add('active');
    
    // Load data if needed
    if (tabName === 'comparison') {
        const storeId = document.getElementById('storeId').value;
        if (storeId) {
            loadComparison(storeId);
        }
    } else if (tabName === 'metrics') {
        loadMetrics();
    }
}

// Load Stores
async function loadStores() {
    try {
        const response = await fetch(`${API_URL}/stores`);
        const data = await response.json();
        const select = document.getElementById('storeId');
        
        data.stores.forEach(store => {
            const option = document.createElement('option');
            option.value = store;
            option.textContent = store;
            select.appendChild(option);
        });
        
        if (data.stores.length > 0) {
            select.value = data.stores[0];
        }
    } catch (error) {
        showError('Failed to load stores: ' + error.message);
    }
}

// Run Prediction
async function runPrediction() {
    const runBtn = document.getElementById('runBtn');
    const loadBtn = document.getElementById('loadBtn');
    const statusBox = document.getElementById('statusBox');
    
    runBtn.disabled = true;
    loadBtn.disabled = true;
    
    statusBox.className = 'status-box status-running';
    statusBox.innerHTML = `⏳ Running prediction with <strong>${getModelName(selectedModel)}</strong>... This may take a few minutes.`;
    statusBox.style.display = 'flex';
    
    try {
        const response = await fetch(`${API_URL}/run-prediction`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: selectedModel })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            checkPredictionStatus();
        } else {
            throw new Error(data.message || 'Failed to start prediction');
        }
    } catch (error) {
        statusBox.className = 'status-box status-error';
        statusBox.innerHTML = '❌ Error: ' + error.message;
        runBtn.disabled = false;
        loadBtn.disabled = false;
    }
}

// Check Prediction Status
async function checkPredictionStatus() {
    statusCheckInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_URL}/prediction-status`);
            const status = await response.json();
            
            if (!status.running) {
                clearInterval(statusCheckInterval);
                const runBtn = document.getElementById('runBtn');
                const loadBtn = document.getElementById('loadBtn');
                const statusBox = document.getElementById('statusBox');
                
                if (status.error) {
                    statusBox.className = 'status-box status-error';
                    statusBox.innerHTML = '❌ Prediction failed: ' + status.error;
                } else {
                    const lastRun = status.last_run ? formatDateTime(status.last_run) : 'Unknown';
                    const modelName = getModelName(status.selected_model || selectedModel);
                    statusBox.className = 'status-box status-success';
                    statusBox.innerHTML = `✅ Prediction completed with <strong>${modelName}</strong>! Last run: ${lastRun}`;
                    
                    setTimeout(() => {
                        fetchPredictions();
                        loadMetrics();
                    }, 1000);
                }
                
                runBtn.disabled = false;
                loadBtn.disabled = false;
            }
        } catch (error) {
            console.error('Status check error:', error);
        }
    }, 15000);
}

// Fetch Predictions
async function fetchPredictions() {
    const storeId = document.getElementById('storeId').value.trim();
    const productCode = document.getElementById('productCode').value.trim();
    
    document.getElementById('error').style.display = 'none';
    document.getElementById('noData').style.display = 'none';
    document.getElementById('resultsTable').style.display = 'none';
    
    if (!storeId) {
        showError('Please select a Store ID');
        return;
    }
    
    document.getElementById('loading').style.display = 'block';

    try {
        let url = `${API_URL}/predictions?store_id=${storeId}`;
        if (productCode) url += `&prod_cd=${productCode}`;
        
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        
        const data = await response.json();
        document.getElementById('loading').style.display = 'none';
        
        if (data.length === 0) {
            document.getElementById('noData').style.display = 'block';
        } else {
            displayResults(data);
        }
    } catch (error) {
        document.getElementById('loading').style.display = 'none';
        showError(`Error fetching data: ${error.message}`);
    }
}

// Display Results
function displayResults(data) {
    const tableBody = document.getElementById('tableBody');
    tableBody.innerHTML = '';
    
    data.forEach(row => {
        const tr = document.createElement('tr');
        const createdAt = row.created_at ? formatDateTime(row.created_at) : 'N/A';
        tr.innerHTML = `
            <td>${row.date}</td>
            <td>${row.prod_cd}</td>
            <td><strong>${row.predicted_qty.toLocaleString()}</strong></td>
            <td>${createdAt}</td>
        `;
        tableBody.appendChild(tr);
    });
    
    document.getElementById('resultsTable').style.display = 'table';
}

// Show Comparison
function showComparison() {
    const storeId = document.getElementById('storeId').value.trim();
    if (!storeId) {
        showError('Please select a Store ID');
        return;
    }
    
    switchTab('comparison');
    loadComparison(storeId);
}

// Load Comparison
async function loadComparison(storeId) {
    document.getElementById('comparisonLoading').style.display = 'block';
    document.getElementById('comparisonTable').style.display = 'none';
    document.getElementById('comparisonNoData').style.display = 'none';
    
    try {
        const response = await fetch(`${API_URL}/model-predictions/${storeId}`);
        
        if (!response.ok) {
            if (response.status === 404) {
                throw new Error('No comparison data available. Please run prediction first.');
            }
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        document.getElementById('comparisonLoading').style.display = 'none';
        
        if (data.length === 0) {
            document.getElementById('comparisonNoData').style.display = 'block';
        } else {
            displayComparison(data);
        }
    } catch (error) {
        document.getElementById('comparisonLoading').style.display = 'none';
        document.getElementById('comparisonNoData').style.display = 'block';
        document.getElementById('comparisonNoData').textContent = error.message;
    }
}

// Display Comparison
function displayComparison(data) {
    const tbody = document.getElementById('comparisonBody');
    tbody.innerHTML = '';
    
    console.log('[DEBUG] Comparison data:', data);
    
    data.sort((a, b) => b.ensemble - a.ensemble);
    
    data.slice(0, 20).forEach((row, index) => {  // Show top 20 products
        console.log(`[DEBUG] Row ${index}:`, row);
        
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td><strong>${row.product_code || row.prod_cd || 'N/A'}</strong></td>
            <td class="comparison-cell model-ensemble">${(row.ensemble || 0).toLocaleString()}</td>
            <td class="comparison-cell model-autoencoder">${(row.autoencoder || 0).toLocaleString()}</td>
            <td class="comparison-cell model-exp">${(row.exp_smoothing || 0).toLocaleString()}</td>
            <td class="comparison-cell model-linear">${(row.linear_regression || 0).toLocaleString()}</td>
        `;
        tbody.appendChild(tr);
    });
    
    document.getElementById('comparisonTable').style.display = 'table';
}

// Load Metrics
async function loadMetrics() {
    document.getElementById('metricsLoading').style.display = 'block';
    document.getElementById('metricsGrid').style.display = 'none';
    document.getElementById('metricsNoData').style.display = 'none';
    
    try {
        const response = await fetch(`${API_URL}/model-comparison`);
        
        if (!response.ok) {
            if (response.status === 404) {
                throw new Error('No metrics available. Please run prediction first.');
            }
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        document.getElementById('metricsLoading').style.display = 'none';
        
        if (data.metrics && data.metrics.length > 0) {
            displayMetrics(data.metrics);
        } else {
            document.getElementById('metricsNoData').style.display = 'block';
        }
    } catch (error) {
        document.getElementById('metricsLoading').style.display = 'none';
        document.getElementById('metricsNoData').style.display = 'block';
        document.getElementById('metricsNoData').textContent = error.message;
    }
}

// Display Metrics
function displayMetrics(metrics) {
    const grid = document.getElementById('metricsGrid');
    grid.innerHTML = '';
    
    const modelClasses = {
        'Ensemble': 'ensemble',
        'Autoencoder': 'autoencoder',
        'Exponential Smoothing': 'exp',
        'Linear Regression': 'linear'
    };
    
    metrics.forEach(metric => {
        const card = document.createElement('div');
        const modelClass = modelClasses[metric.model] || 'ensemble';
        card.className = `metric-card ${modelClass}`;
        
        card.innerHTML = `
            <h4>${metric.model}</h4>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-top: 15px;">
                <div>
                    <div class="metric-value">${metric.mae.toFixed(2)}</div>
                    <div class="metric-label">MAE</div>
                </div>
                <div>
                    <div class="metric-value">${metric.rmse.toFixed(2)}</div>
                    <div class="metric-label">RMSE</div>
                </div>
                <div>
                    <div class="metric-value">${metric.mape.toFixed(1)}%</div>
                    <div class="metric-label">MAPE</div>
                </div>
            </div>
        `;
        
        grid.appendChild(card);
    });
    
    grid.style.display = 'grid';
}

// Helper Functions
function getModelName(modelKey) {
    const names = {
        'ensemble': 'Ensemble',
        'autoencoder': 'Autoencoder',
        'exp_smoothing': 'Exponential Smoothing',
        'linear_regression': 'Linear Regression'
    };
    return names[modelKey] || modelKey;
}

function formatDateTime(dateTimeString) {
    if (!dateTimeString) return 'N/A';
    
    try {
        const date = new Date(dateTimeString);
        return date.toLocaleString('th-TH', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            hour12: false
        });
    } catch (error) {
        return dateTimeString;
    }
}

function showError(message) {
    const errorDiv = document.getElementById('error');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
}