const API_BASE = 'http://localhost:8000/api/v1';

let activeTransactionId = null;
let allTransactions = [];

// Initialize Dashboard
document.addEventListener('DOMContentLoaded', () => {
    // Start polling data
    pollDashboardStats();
    setInterval(pollDashboardStats, 1500);

    // Setup Simulation Button
    const btnSimulate = document.getElementById('btn-run-simulation');
    btnSimulate.addEventListener('click', runAdversarialSimulation);
});

async function pollDashboardStats() {
    try {
        const response = await fetch(`${API_BASE}/dashboard`);
        if (!response.ok) return;

        const data = await response.json();
        allTransactions = data.recent_transactions;
        
        // Update Stats counters
        document.getElementById('val-total').innerText = data.stats.total_transactions;
        document.getElementById('val-allowed').innerText = data.stats.allowed_count;
        document.getElementById('val-stepup').innerText = data.stats.step_up_count;
        document.getElementById('val-blocked').innerText = data.stats.blocked_count;

        // Update Average Trust Score Circular Gauge
        const avgScore = Math.round(data.stats.average_trust_score);
        document.getElementById('average-score').innerText = avgScore;
        
        const gaugeArc = document.getElementById('gauge-fill-arc');
        // Circle circumference is 2 * PI * r = 2 * 3.14159 * 40 = 251.2
        const maxOffset = 251.2;
        const offset = maxOffset - (maxOffset * (avgScore / 100));
        gaugeArc.style.strokeDashoffset = offset;
        
        // Set gauge color based on score
        if (avgScore >= 85) {
            gaugeArc.style.stroke = '#10b981'; // Emerald
        } else if (avgScore >= 60) {
            gaugeArc.style.stroke = '#f59e0b'; // Amber
        } else {
            gaugeArc.style.stroke = '#ef4444'; // Red
        }

        // Render Transaction Feed Table
        renderTransactionTable(data.recent_transactions);

    } catch (error) {
        console.error("Dashboard poll failed:", error);
    }
}

function renderTransactionTable(transactions) {
    const feed = document.getElementById('transaction-feed');
    
    if (transactions.length === 0) {
        feed.innerHTML = `
            <tr class="placeholder-row">
                <td colspan="6">Waiting for transaction stream... Start the stream simulation.</td>
            </tr>`;
        return;
    }

    let rowsHtml = '';
    transactions.forEach((tx, idx) => {
        const isActive = (activeTransactionId === tx.account_id && idx === 0) || (activeTransactionId === `${tx.account_id}-${tx.timestamp}`);
        const activeClass = isActive ? 'active' : '';
        const uniqueId = `${tx.account_id}-${tx.timestamp}`;

        rowsHtml += `
            <tr class="transaction-row ${activeClass}" onclick="selectTransaction('${tx.account_id}', '${tx.timestamp}')" data-id="${uniqueId}">
                <td><span class="account-lbl">${tx.account_id}</span></td>
                <td><span class="amount-lbl">$${tx.amount.toLocaleString()}</span></td>
                <td><span class="geo-lbl">${tx.geolocation}</span></td>
                <td><span class="dev-lbl">${tx.device_fingerprint.substring(0, 8)}...</span></td>
                <td><strong class="trust-lbl" style="color: ${getTrustColor(tx.trust_score)}">${Math.round(tx.trust_score)}</strong></td>
                <td><span class="badge-decision ${tx.decision.toLowerCase()}">${tx.decision}</span></td>
            </tr>
        `;
    });

    feed.innerHTML = rowsHtml;
    
    // Auto-select the first transaction if none is active
    if (!activeTransactionId && transactions.length > 0) {
        selectTransaction(transactions[0].account_id, transactions[0].timestamp);
    } else {
        // If there's an active one, keep its details updated
        updateActiveCaseDetails();
    }
}

function selectTransaction(accountId, timestamp) {
    activeTransactionId = `${accountId}-${timestamp}`;
    
    // Toggle active class in DOM
    const rows = document.querySelectorAll('.transaction-row');
    rows.forEach(r => {
        if (r.getAttribute('data-id') === activeTransactionId) {
            r.classList.add('active');
        } else {
            r.classList.remove('active');
        }
    });

    updateActiveCaseDetails();
}

function updateActiveCaseDetails() {
    if (!activeTransactionId || allTransactions.length === 0) return;

    const tx = allTransactions.find(t => `${t.account_id}-${t.timestamp}` === activeTransactionId);
    if (!tx) return;

    // Update case display fields
    document.getElementById('active-case-id').innerText = `CASE #${tx.account_id.substring(0, 8)}`;
    
    const statusBadge = document.getElementById('active-case-status');
    statusBadge.innerText = tx.decision;
    statusBadge.className = `case-status-badge text-${getDecisionColorClass(tx.decision)}`;

    document.getElementById('case-xgb').innerText = `${(tx.xgb_prob * 100).toFixed(1)}%`;
    document.getElementById('case-gnn').innerText = `${(tx.gnn_risk * 100).toFixed(1)}%`;
    document.getElementById('case-bio').innerText = `${(tx.biometric_score * 100).toFixed(1)}%`;
    document.getElementById('case-narrative').innerText = tx.reason;

    // Render the account graph neighborhood
    renderNeighborhoodGraph(tx.account_id);
}

async function renderNeighborhoodGraph(accountId) {
    try {
        const response = await fetch(`${API_BASE}/graph/${accountId}`);
        if (!response.ok) return;
        const data = await response.json();
        
        const cyNodes = data.nodes.map(n => {
            let color = '#3b82f6'; // Blue for regular Accounts
            let shape = 'ellipse';
            let label = n.id;
            
            if (n.id === accountId) {
                color = '#ef4444'; // Red highlight for active account
            } else if (n.type === 'Device') {
                color = '#10b981'; // Green for Devices
                shape = 'rectangle';
                label = n.id.substring(0, 8);
            }
            
            return {
                data: { id: n.id, label: label, color: color, shape: shape }
            };
        });
        
        const cyEdges = data.edges.map((e, idx) => {
            let color = '#4b5563'; // Gray edge
            let label = '';
            if (e.type === 'TRANSFERRED') {
                color = e.is_fraud === 1 ? '#ef4444' : '#6b7280';
                label = `$${e.amount.toLocaleString()}`;
            } else if (e.type === 'USED_DEVICE') {
                color = '#10b981';
            }
            return {
                data: { id: `e_${idx}`, source: e.source, target: e.target, color: color, label: label }
            };
        });
        
        const elements = [...cyNodes, ...cyEdges];
        
        cytoscape({
            container: document.getElementById('cy'),
            elements: elements,
            style: [
                {
                    selector: 'node',
                    style: {
                        'background-color': 'data(color)',
                        'label': 'data(label)',
                        'shape': 'data(shape)',
                        'width': '20px',
                        'height': '20px',
                        'color': '#fff',
                        'font-size': '8px',
                        'text-valign': 'bottom',
                        'text-margin-y': '4px'
                    }
                },
                {
                    selector: 'edge',
                    style: {
                        'width': 1.5,
                        'line-color': 'data(color)',
                        'target-arrow-color': 'data(color)',
                        'target-arrow-shape': 'triangle',
                        'curve-style': 'bezier',
                        'label': 'data(label)',
                        'font-size': '6px',
                        'color': '#9ca3af',
                        'text-background-opacity': 0.7,
                        'text-background-color': '#111827',
                        'text-background-padding': '1px'
                    }
                }
            ],
            layout: {
                name: 'cose',
                animate: false,
                fit: true,
                padding: 15
            }
        });
        
    } catch (error) {
        console.error("Failed to render neighborhood graph:", error);
    }
}

async function runAdversarialSimulation() {
    const btn = document.getElementById('btn-run-simulation');
    const loading = document.getElementById('simulation-loading');

    // Show loading
    btn.disabled = true;
    loading.style.display = 'flex';

    try {
        const response = await fetch(`${API_BASE}/simulate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        if (response.ok) {
            const result = await response.json();
            document.getElementById('rb-before').innerText = result.before_detection;
            document.getElementById('rb-after').innerText = result.after_detection;
            console.log("Retraining log:", result.logs);
        } else {
            alert("Simulation request failed.");
        }
    } catch (err) {
        console.error("Adversarial simulation error:", err);
        alert("Could not connect to decision API.");
    } finally {
        btn.disabled = false;
        loading.style.display = 'none';
    }
}

function getTrustColor(score) {
    if (score >= 85) return '#10b981'; // Green
    if (score >= 60) return '#f59e0b'; // Yellow
    return '#ef4444'; // Red
}

function getDecisionColorClass(decision) {
    if (decision === 'ALLOW') return 'green';
    if (decision === 'STEP_UP') return 'yellow';
    if (decision === 'FLAG') return 'blue';
    return 'red';
}
