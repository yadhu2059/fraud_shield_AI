const API_BASE = 'http://localhost:8000/api/v1';

let activeTransactionId = null;
let allTransactions = [];
let lastRenderedAccountId = null;
let cyInstance = null;

// Initialize Dashboard
document.addEventListener('DOMContentLoaded', () => {
    // Start polling data
    pollDashboardStats();
    setInterval(pollDashboardStats, 1500);

    // Setup Simulation Button
    const btnSimulate = document.getElementById('btn-run-simulation');
    btnSimulate.addEventListener('click', runAdversarialSimulation);

    // Setup Navigation Tabs
    const navButtons = document.querySelectorAll('.nav-btn');
    navButtons.forEach(btn => {
        btn.addEventListener('click', (e) => {
            const targetView = btn.getAttribute('data-view');
            if (!targetView) return;
            
            // Switch active buttons
            navButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Hide all views
            document.querySelectorAll('.view-section').forEach(view => {
                view.style.display = 'none';
            });
            
            // Show selected view
            const viewEl = document.getElementById(`view-${targetView}`);
            if (viewEl) {
                viewEl.style.display = targetView === 'dashboard' ? 'block' : 'block';
                if (targetView === 'graph') {
                    loadInteractiveFraudNetwork();
                }
            }
        });
    });

    // Setup Search & Toggle in Fraud Network View
    const btnSearch = document.getElementById('btn-graph-search');
    const searchInput = document.getElementById('graph-search-input');
    const toggleSuspicious = document.getElementById('toggle-suspicious-only');

    if (btnSearch && searchInput) {
        const performSearch = () => {
            const q = searchInput.value.trim();
            if (q) {
                loadInteractiveFraudNetwork(q);
            }
        };
        btnSearch.addEventListener('click', performSearch);
        searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') performSearch();
        });
    }

    if (toggleSuspicious) {
        toggleSuspicious.addEventListener('change', () => {
            if (currentGraphData) {
                renderVisNetwork(currentGraphData);
            }
        });
    }
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

        // Update health status badges
        if (data.health) {
            const onnxDot = document.getElementById('status-onnx-dot');
            const onnxLabel = document.getElementById('status-onnx-label');
            const gnnDot = document.getElementById('status-gnn-dot');
            const gnnLabel = document.getElementById('status-gnn-label');

            if (onnxDot && onnxLabel) {
                onnxLabel.innerText = `FAST-LANE ONNX: ${data.health.onnx} (sub-10ms)`;
                if (data.health.onnx === 'ONLINE') {
                    onnxDot.className = 'status-indicator online';
                } else {
                    onnxDot.className = 'status-indicator offline';
                }
            }
            if (gnnDot && gnnLabel) {
                gnnLabel.innerText = `SLOW-LANE GNN: ${data.health.gnn}`;
                if (data.health.gnn === 'ONLINE') {
                    gnnDot.className = 'status-indicator online';
                } else {
                    gnnDot.className = 'status-indicator warning';
                }
            }
        }

        // Update Average Trust Score Circular Gauge
        const avgScore = Math.round(data.stats.average_trust_score);
        document.getElementById('average-score').innerText = avgScore;
        
        const gaugeArc = document.getElementById('gauge-fill-arc');
        // Circle circumference is 2 * PI * r = 2 * 3.14159 * 40 = 251.2
        const maxOffset = 251.2;
        const offset = maxOffset - (maxOffset * (avgScore / 100));
        gaugeArc.style.strokeDashoffset = offset;
        
        // Set gauge color based on score
        if (avgScore >= 90) {
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
    const container = document.querySelector('.transaction-list-container');
    const scrollTop = container ? container.scrollTop : 0;
    
    if (transactions.length === 0) {
        feed.innerHTML = `
            <tr class="placeholder-row">
                <td colspan="8">Waiting for transaction stream... Start the stream simulation.</td>
            </tr>`;
        return;
    }

    let rowsHtml = '';
    transactions.forEach((tx) => {
        const uniqueId = `${tx.account_id}-${tx.timestamp}`;
        const isActive = activeTransactionId === uniqueId;
        const activeClass = isActive ? 'active' : '';

        rowsHtml += `
            <tr class="transaction-row ${activeClass}" onclick="selectTransaction('${tx.account_id}', '${tx.timestamp}')" data-id="${uniqueId}">
                <td><span class="account-lbl">${tx.account_id}</span></td>
                <td><span class="account-lbl">${tx.destination || 'N/A'}</span></td>
                <td><span class="amount-lbl">$${tx.amount.toLocaleString()}</span></td>
                <td><span class="geo-lbl">${tx.geolocation}</span></td>
                <td><span class="dev-lbl">${tx.device_fingerprint.substring(0, 8)}...</span></td>
                <td><span class="dev-lbl">${tx.ip_address || '0.0.0.0'}</span></td>
                <td><strong class="trust-lbl" style="color: ${getTrustColor(tx.trust_score)}">${Math.round(tx.trust_score)}</strong></td>
                <td><span class="badge-decision ${tx.decision.toLowerCase()}">${tx.decision}</span></td>
            </tr>
        `;
    });

    feed.innerHTML = rowsHtml;
    
    if (container) {
        container.scrollTop = scrollTop;
    }
    
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

    document.getElementById('case-dest').innerText = tx.destination || 'N/A';
    document.getElementById('case-ip').innerText = tx.ip_address || '0.0.0.0';

    document.getElementById('case-xgb').innerText = `${(tx.xgb_prob * 100).toFixed(1)}%`;
    document.getElementById('case-gnn').innerText = `${(tx.gnn_risk * 100).toFixed(1)}%`;
    document.getElementById('case-bio').innerText = `${(tx.biometric_score * 100).toFixed(1)}%`;
    
    const ifAnomalyBadge = document.getElementById('case-if-anomaly');
    if (ifAnomalyBadge) {
        if (tx.if_anomaly === 1) {
            ifAnomalyBadge.innerText = 'ANOMALY DETECTED';
            ifAnomalyBadge.className = 'badge-if anomaly';
        } else {
            ifAnomalyBadge.innerText = 'SAFE';
            ifAnomalyBadge.className = 'badge-if safe';
        }
    }
    
    document.getElementById('case-narrative').innerText = tx.reason;

    // Render SHAP Bars
    const shapContainer = document.getElementById('shap-bars');
    if (shapContainer) {
        if (tx.shap_attributions && Object.keys(tx.shap_attributions).length > 0) {
            let shapHtml = '';
            const values = Object.values(tx.shap_attributions);
            const maxAbsVal = Math.max(...values.map(Math.abs), 0.01);
            
            Object.entries(tx.shap_attributions).forEach(([featName, val]) => {
                const absVal = Math.abs(val);
                const percentage = Math.min(100, Math.round((absVal / maxAbsVal) * 100));
                const directionClass = val >= 0 ? 'positive' : 'negative';
                const sign = val >= 0 ? '+' : '-';
                shapHtml += `
                    <div class="shap-bar-row">
                        <div class="shap-bar-label">
                            <span>${featName}</span>
                            <strong>${sign}${absVal.toFixed(2)}</strong>
                        </div>
                        <div class="shap-bar-bg">
                            <div class="shap-bar-fill ${directionClass}" style="width: ${percentage}%"></div>
                        </div>
                    </div>
                `;
            });
            shapContainer.innerHTML = shapHtml;
        } else {
            shapContainer.innerHTML = '<span style="font-size: 11px; color: var(--text-secondary);">No SHAP explanation details available for this record.</span>';
        }
    }

    // Render the account graph neighborhood only when selection changes
    if (tx.account_id !== lastRenderedAccountId) {
        renderNeighborhoodGraph(tx.account_id);
        lastRenderedAccountId = tx.account_id;
    }
}

async function renderNeighborhoodGraph(accountId) {
    try {
        const response = await fetch(`${API_BASE}/graph/${accountId}`);
        if (!response.ok) return;
        const data = await response.json();
        
        // Clean up previous Cytoscape instances to prevent leaks and canvas layers
        if (cyInstance) {
            cyInstance.destroy();
            cyInstance = null;
        }

        // Setup tooltip overlay
        let tooltipEl = document.getElementById('cy-tooltip');
        if (!tooltipEl) {
            tooltipEl = document.createElement('div');
            tooltipEl.id = 'cy-tooltip';
            tooltipEl.style.position = 'absolute';
            tooltipEl.style.background = '#1f2937';
            tooltipEl.style.border = '1px solid rgba(255, 255, 255, 0.1)';
            tooltipEl.style.padding = '6px 10px';
            tooltipEl.style.borderRadius = '4px';
            tooltipEl.style.fontSize = '11px';
            tooltipEl.style.color = '#fff';
            tooltipEl.style.pointerEvents = 'none';
            tooltipEl.style.display = 'none';
            tooltipEl.style.zIndex = '999';
            document.getElementById('cy').parentNode.style.position = 'relative';
            document.getElementById('cy').parentNode.appendChild(tooltipEl);
        }

        const cyNodes = data.nodes.map(n => {
            let cyType = n.type === 'device' ? 'device' : 'account';
            let label = n.id;
            let size = 44;
            let borderWidth = 0;
            let borderColor = '#ffffff';
            
            if (n.id === accountId) {
                borderWidth = 3;
                size = 50;
            }
            if (cyType === 'device') {
                label = n.id.length > 10 ? n.id.substring(0, 10) : n.id;
            }
            
            return {
                data: { 
                    id: n.id, 
                    label: label, 
                    size: size, 
                    border_width: borderWidth, 
                    border_color: borderColor,
                    type: cyType,
                    status: n.status
                }
            };
        });
        
        const cyEdges = data.edges.map((e, idx) => {
            let label = '';
            const typeLower = e.type.toLowerCase();
            if (typeLower === 'transaction' || typeLower === 'transferred') {
                label = 'USED_CARD';
            } else if (typeLower === 'used_device') {
                label = 'ON_DEVICE';
            }
            return {
                data: { 
                    id: `e_${idx}`, 
                    source: e.source, 
                    target: e.target, 
                    label: label, 
                    type: e.type,
                    status: e.status,
                    amount: e.amount,
                    reason: e.reason
                }
            };
        });
        
        const elements = [...cyNodes, ...cyEdges];
        
        cyInstance = cytoscape({
            container: document.getElementById('cy'),
            elements: elements,
            style: [
                {
                    selector: 'node',
                    style: {
                        'label': 'data(label)',
                        'width': 'data(size)',
                        'height': 'data(size)',
                        'border-width': 'data(border_width)',
                        'border-color': 'data(border_color)',
                        'color': '#ffffff',
                        'font-size': '6px',
                        'font-family': 'sans-serif',
                        'font-weight': 'bold',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'text-wrap': 'wrap',
                        'text-overlap': 'show',
                        'min-zoomed-font-size': 0
                    }
                },
                // Safe/Normal Account node -> green (#22C55E)
                {
                    selector: 'node[type = "account"][status = "safe"]',
                    style: {
                        'background-color': '#22C55E',
                        'shape': 'ellipse'
                    }
                },
                // Device node -> purple (#8B5CF6)
                {
                    selector: 'node[type = "device"]',
                    style: {
                        'background-color': '#8B5CF6',
                        'shape': 'round-rectangle'
                    }
                },
                // Fraud Account node -> red (#DC2626)
                {
                    selector: 'node[type = "account"][status = "fraud"]',
                    style: {
                        'background-color': '#DC2626',
                        'shape': 'ellipse',
                        'z-index': 100
                    }
                },
                // Default Edge
                {
                    selector: 'edge',
                    style: {
                        'width': 1.5,
                        'line-color': '#9CA3AF',
                        'target-arrow-color': '#9CA3AF',
                        'target-arrow-shape': 'triangle',
                        'curve-style': 'bezier',
                        'label': 'data(label)',
                        'font-size': '5px',
                        'color': '#9ca3af',
                        'text-background-opacity': 0.8,
                        'text-background-color': '#0b0f19',
                        'text-background-padding': '1px'
                    }
                },
                // Fraud Transfer edge -> red (#DC2626)
                {
                    selector: 'edge[status = "fraud_transfer"]',
                    style: {
                        'width': 3.0,
                        'line-color': '#DC2626',
                        'target-arrow-color': '#DC2626',
                        'z-index': 200
                    }
                },
                // Used device edge
                {
                    selector: 'edge[type = "used_device"]',
                    style: {
                        'line-color': '#8B5CF6',
                        'target-arrow-color': '#8B5CF6',
                        'line-style': 'dashed'
                    }
                }
            ],
            layout: {
                name: 'cose',
                animate: false,
                fit: true,
                padding: 10,
                componentSpacing: 80,
                nodeRepulsion: function(node) { return 1000000; },
                idealEdgeLength: function(edge) { return 60; },
                edgeElasticity: function(edge) { return 100; }
            }
        });

        // Add Hover Tooltips on Nodes & Edges
        cyInstance.on('mouseover', 'node', function(evt) {
            const node = evt.target;
            const type = node.data('type') || 'Account';
            const status = node.data('status') || 'safe';
            tooltipEl.innerHTML = `<strong>Type:</strong> ${type.toUpperCase()}<br><strong>ID:</strong> ${node.id()}<br><strong>Status:</strong> ${status.toUpperCase()}`;
            tooltipEl.style.display = 'block';
        });
        cyInstance.on('mousemove', 'node', function(evt) {
            const pos = evt.renderedPosition;
            tooltipEl.style.left = (pos.x + 10) + 'px';
            tooltipEl.style.top = (pos.y + 10) + 'px';
        });
        cyInstance.on('mouseout', 'node', function() {
            tooltipEl.style.display = 'none';
        });

        cyInstance.on('mouseover', 'edge', function(evt) {
            const edge = evt.target;
            const type = edge.data('type');
            const status = edge.data('status');
            const amount = edge.data('amount');
            const reason = edge.data('reason');
            
            if (status === 'fraud_transfer') {
                const amtStr = amount !== undefined ? '$' + amount.toLocaleString() : 'N/A';
                tooltipEl.innerHTML = `<strong>Fraud Transfer</strong><br><strong>Amount:</strong> ${amtStr}<br><strong>Reason:</strong> ${reason || 'N/A'}`;
                tooltipEl.style.display = 'block';
            } else if (type === 'TRANSFERRED' || type === 'transaction') {
                const amtStr = amount !== undefined ? '$' + amount.toLocaleString() : 'N/A';
                tooltipEl.innerHTML = `<strong>Transfer Amount:</strong> ${amtStr}`;
                tooltipEl.style.display = 'block';
            } else if (type === 'USED_DEVICE' || type === 'used_device') {
                tooltipEl.innerHTML = `<strong>Device Link</strong>`;
                tooltipEl.style.display = 'block';
            }
        });
        cyInstance.on('mousemove', 'edge', function(evt) {
            const pos = evt.renderedPosition;
            tooltipEl.style.left = (pos.x + 10) + 'px';
            tooltipEl.style.top = (pos.y + 10) + 'px';
        });
        cyInstance.on('mouseout', 'edge', function() {
            tooltipEl.style.display = 'none';
        });
        
    } catch (error) {
        console.error("Failed to render neighborhood graph:", error);
    }
}

async function runAdversarialSimulation() {
    const btn = document.getElementById('btn-run-simulation');
    const loading = document.getElementById('simulation-loading');
    const terminalContainer = document.getElementById('terminal-container');
    const terminalLogs = document.getElementById('terminal-logs');

    // Show loading
    btn.disabled = true;
    loading.style.display = 'flex';
    if (terminalContainer) terminalContainer.style.display = 'block';
    if (terminalLogs) terminalLogs.innerText = '[*] Starting adversarial simulation...\n[*] Generating evasive smurfing/structuring patterns...\n';

    try {
        const response = await fetch(`${API_BASE}/simulate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        if (response.ok) {
            const result = await response.json();
            document.getElementById('rb-before').innerText = result.before_detection;
            document.getElementById('rb-after').innerText = result.after_detection;
            if (terminalLogs) {
                terminalLogs.innerText = result.logs || '[+] Retraining completed successfully.';
            }
            // Trigger refresh after retraining
            pollDashboardStats();
        } else {
            if (terminalLogs) terminalLogs.innerText = '[-] Simulation failed with server error.';
            alert("Simulation request failed.");
        }
    } catch (err) {
        console.error("Adversarial simulation error:", err);
        if (terminalLogs) terminalLogs.innerText = `[-] Connection error: ${err.message}`;
        alert("Could not connect to decision API.");
    } finally {
        btn.disabled = false;
        loading.style.display = 'none';
    }
}

function getTrustColor(score) {
    if (score >= 90) return '#10b981'; // Green
    if (score >= 60) return '#f59e0b'; // Yellow
    return '#ef4444'; // Red
}

function getDecisionColorClass(decision) {
    if (decision === 'ALLOW') return 'green';
    if (decision === 'STEP_UP') return 'yellow';
    if (decision === 'FLAG') return 'blue';
    return 'red';
}

let currentGraphData = null;
let visNetworkInstance = null;

async function loadInteractiveFraudNetwork(accountId) {
    if (!accountId) {
        if (activeTransactionId) {
            accountId = activeTransactionId.split('-')[0];
        } else if (allTransactions.length > 0) {
            accountId = allTransactions[0].account_id;
        }
    }
    
    if (!accountId) {
        console.warn("No active account selected for graph search.");
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/graph/${accountId}`);
        if (!response.ok) {
            console.error("Failed to fetch graph data");
            return;
        }
        const data = await response.json();
        currentGraphData = data;
        renderVisNetwork(data);
    } catch (err) {
        console.error("Error loading interactive network graph:", err);
    }
}

function getRiskColor(risk) {
    if (risk < 0.5) {
        const pct = risk / 0.5;
        const r = Math.round(16 + (245 - 16) * pct);
        const g = Math.round(185 + (158 - 185) * pct);
        const b = Math.round(129 + (11 - 129) * pct);
        return `rgb(${r}, ${g}, ${b})`;
    } else {
        const pct = (risk - 0.5) / 0.5;
        const r = Math.round(245 + (239 - 245) * pct);
        const g = Math.round(158 + (68 - 158) * pct);
        const b = Math.round(11 + (68 - 11) * pct);
        return `rgb(${r}, ${g}, ${b})`;
    }
}

function renderVisNetwork(data) {
    const container = document.getElementById('fraud-network-graph');
    if (!container) return;

    const showSuspiciousOnly = document.getElementById('toggle-suspicious-only')?.checked;
    
    let nodesToRender = data.nodes;
    if (showSuspiciousOnly) {
        nodesToRender = data.nodes.filter(n => n.risk > 0.3 || n.in_ring === true);
    }
    
    const nodeIds = new Set(nodesToRender.map(n => n.id));
    const edgesToRender = data.edges.filter(e => nodeIds.has(e.source) && nodeIds.has(e.target));

    const visNodes = nodesToRender.map(n => {
        let shape = 'circle';
        if (n.type === 'device') shape = 'box';
        else if (n.type === 'ip') shape = 'diamond';
        else if (n.type === 'beneficiary') shape = 'triangle';
        
        let bgColor = '#1f77b4'; // Default Blue for Cards/Accounts
        if (n.type === 'device') {
            bgColor = '#f58220'; // Orange for Devices
        } else if (n.type === 'ip') {
            bgColor = '#a855f7'; // Purple for IP Addresses
        } else if (n.risk > 0.5 || n.in_ring === true) {
            bgColor = '#ef4444'; // Red for Fraud/In-Ring accounts
        } else if (n.type === 'beneficiary') {
            bgColor = '#eab308'; // Yellow for Beneficiaries
        }
        const borderWidth = n.in_ring ? 3 : 1;
        const borderColor = n.in_ring ? '#ef4444' : '#374151';
        
        let title = `<strong>Type:</strong> ${n.type.toUpperCase()}<br><strong>ID:</strong> ${n.id}<br><strong>Risk Score:</strong> ${(n.risk * 100).toFixed(1)}%<br><strong>In Ring:</strong> ${n.in_ring}`;
        if (n.attributes) {
            title += "<hr style='border:0;border-top:1px solid rgba(255,255,255,0.1);margin:5px 0;'>";
            for (const [key, val] of Object.entries(n.attributes)) {
                const cleanKey = key.replace(/_/g, ' ');
                const cleanVal = typeof val === 'number' ? (val % 1 === 0 ? val : val.toFixed(2)) : val;
                title += `<strong>${cleanKey}:</strong> ${cleanVal}<br>`;
            }
        }

        return {
            id: n.id,
            label: `${n.type.toUpperCase()}\n${n.id.substring(0, 10)}`,
            shape: shape,
            color: {
                background: bgColor,
                border: borderColor,
                highlight: {
                    background: bgColor,
                    border: '#00f2fe'
                }
            },
            borderWidth: borderWidth,
            font: {
                color: '#ffffff',
                size: 9,
                face: 'sans-serif'
            },
            size: 24,
            title: title
        };
    });

    const visEdges = edgesToRender.map(e => {
        let color = '#4b5563'; 
        let width = 1.5;
        let dashes = false;
        
        if (e.in_ring) {
            color = '#ef4444'; 
            width = 3.0;
        } else if (e.type === 'used_device') {
            color = '#f58220';
            dashes = true;
        } else if (e.type === 'used_ip') {
            color = '#a855f7';
            dashes = true;
        }
        
        let label = '';
        if (e.type === 'transaction' && e.amount > 0) {
            label = `$${e.amount.toLocaleString()}`;
        }
        
        return {
            from: e.source,
            to: e.target,
            color: {
                color: color,
                highlight: '#00f2fe',
                hover: '#00f2fe'
            },
            width: width,
            arrows: {
                to: {
                    enabled: true,
                    scaleFactor: 0.8
                }
            },
            dashes: dashes,
            label: label,
            font: {
                color: '#9ca3af',
                size: 8,
                background: '#0b0f19',
                strokeWidth: 0
            }
        };
    });

    const dataset = {
        nodes: new vis.DataSet(visNodes),
        edges: new vis.DataSet(visEdges)
    };

    const options = {
        interaction: {
            hover: true,
            tooltipDelay: 100,
            navigationButtons: true,
            keyboard: true
        },
        physics: {
            solver: 'forceAtlas2Based',
            forceAtlas2Based: {
                gravitationalConstant: -50,
                centralGravity: 0.01,
                springLength: 100,
                springConstant: 0.08
            },
            stabilization: {
                enabled: true,
                iterations: 150
            }
        }
    };

    if (visNetworkInstance) {
        visNetworkInstance.destroy();
    }
    
    visNetworkInstance = new vis.Network(container, dataset, options);
    
    let tooltipDiv = document.getElementById('vis-tooltip');
    if (!tooltipDiv) {
        tooltipDiv = document.createElement('div');
        tooltipDiv.id = 'vis-tooltip';
        tooltipDiv.style.position = 'absolute';
        tooltipDiv.style.background = '#1f2937';
        tooltipDiv.style.border = '1px solid rgba(255, 255, 255, 0.1)';
        tooltipDiv.style.padding = '8px 12px';
        tooltipDiv.style.borderRadius = '6px';
        tooltipDiv.style.fontSize = '11px';
        tooltipDiv.style.color = '#fff';
        tooltipDiv.style.pointerEvents = 'none';
        tooltipDiv.style.display = 'none';
        tooltipDiv.style.zIndex = '9999';
        container.appendChild(tooltipDiv);
    }

    visNetworkInstance.on('showPopup', function (params) {
        const nodeData = dataset.nodes.get(params);
        if (nodeData && nodeData.title) {
            tooltipDiv.innerHTML = nodeData.title;
            tooltipDiv.style.display = 'block';
        }
    });

    visNetworkInstance.on('hidePopup', function () {
        tooltipDiv.style.display = 'none';
    });

    container.addEventListener('mousemove', function (e) {
        if (tooltipDiv.style.display === 'block') {
            const rect = container.getBoundingClientRect();
            tooltipDiv.style.left = (e.clientX - rect.left + 15) + 'px';
            tooltipDiv.style.top = (e.clientY - rect.top + 15) + 'px';
        }
    });
}
