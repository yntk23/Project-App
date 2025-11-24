const API_URL = 'http://localhost:5000';
let statusCheckInterval = null;

// Load stores on page load
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
        if (data.stores.length > 0) select.value = data.stores[0];
    } catch (error) {
        showError('Failed to load stores: ' + error.message);
    }
}

async function runPrediction() {
    const runBtn = document.getElementById('runBtn');
    const loadBtn = document.getElementById('loadBtn');
    const statusBox = document.getElementById('statusBox');
    
    runBtn.disabled = true;
    loadBtn.disabled = true;
    
    statusBox.className = 'status-box status-running';
    statusBox.innerHTML = '⏳ Running prediction... This may take a few minutes.';
    statusBox.style.display = 'block';
    
    try {
        const response = await fetch(`${API_URL}/run-prediction`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });
        const data = await response.json();
        if (response.ok) {
            checkPredictionStatus();
        } else {
            throw new Error(data.message || 'Failed to start prediction');
        }
    } catch (error) {
        statusBox.className = 'status-box error';
        statusBox.innerHTML = '❌ Error: ' + error.message;
        runBtn.disabled = false;
        loadBtn.disabled = false;
    }
}

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
                    statusBox.className = 'status-box error';
                    statusBox.innerHTML = '❌ Prediction failed: ' + status.error;
                } else {
                    const lastRun = status.last_run 
                        ? formatDateTime(status.last_run)
                        : 'Unknown';
                    statusBox.className = 'status-box status-success';
                    statusBox.innerHTML = '✅ Prediction completed! Last run: ' + lastRun;
                    setTimeout(() => { fetchPredictions(); }, 1000);
                }
                
                runBtn.disabled = false;
                loadBtn.disabled = false;
            }
        } catch (error) {
            console.error('Status check error:', error);
        }
    }, 30000);
}

async function fetchPredictions() {
    const storeId = document.getElementById('storeId').value.trim();
    const productCode = document.getElementById('productCode').value.trim();
    
    document.getElementById('error').style.display = 'none';
    document.getElementById('noData').style.display = 'none';
    document.getElementById('resultsTable').style.display = 'none';
    
    if (!storeId) { showError('Please enter a Store ID'); return; }
    document.getElementById('loading').style.display = 'block';

    try {
        let url = `${API_URL}/predictions?store_id=${storeId}`;
        if (productCode) url += `&product_code=${productCode}`;
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();
        document.getElementById('loading').style.display = 'none';
        if (data.length === 0) document.getElementById('noData').style.display = 'block';
        else displayResults(data);
    } catch (error) {
        document.getElementById('loading').style.display = 'none';
        showError(`Error fetching data: ${error.message}`);
    }
}

function displayResults(data) {
    const tableBody = document.getElementById('tableBody');
    tableBody.innerHTML = '';
    data.forEach(row => {
        const tr = document.createElement('tr');
        const createdAt = row.created_at ? formatDateTime(row.created_at) : 'N/A';
        tr.innerHTML = `
            <td>${row.date}</td>
            <td>${row.product_code}</td>
            <td>${row.predicted_qty.toLocaleString()}</td>
            <td>${createdAt}</td>
        `;
        tableBody.appendChild(tr);
    });
    document.getElementById('resultsTable').style.display = 'table';
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

window.onload = function() {
    loadStores();
};