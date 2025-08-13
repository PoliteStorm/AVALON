/**
 * Environmental Monitoring Dashboard - Main JavaScript
 * Core functionality for real-time environmental monitoring
 */

class EnvironmentalDashboard {
    constructor() {
        this.socket = null;
        this.charts = {};
        this.updateInterval = null;
        this.lastUpdate = new Date();
        this.systemStatus = 'initializing';
        this.monitoringActive = false;
        this.dataHistory = [];
        this.maxDataPoints = 100;
        
        this.initialize();
    }

    initialize() {
        console.log('🚀 Initializing Environmental Dashboard...');
        
        // Initialize WebSocket connection
        this.initializeWebSocket();
        
        // Initialize charts
        this.initializeCharts();
        
        // Set up event listeners
        this.setupEventListeners();
        
        // Start periodic updates
        this.startPeriodicUpdates();
        
        // Load initial data
        this.loadInitialData();
        
        console.log('✅ Dashboard initialization complete');
    }

    initializeWebSocket() {
        try {
            this.socket = io();
            this.setupWebSocketHandlers();
        } catch (error) {
            console.error('❌ WebSocket initialization failed:', error);
            this.showAlert('WebSocket connection failed. Using fallback mode.', 'warning');
        }
    }

    setupWebSocketHandlers() {
        if (!this.socket) return;

        this.socket.on('connect', () => {
            console.log('✅ WebSocket connected');
            this.updateConnectionStatus('connected', 'Connected');
            this.socket.emit('request_data');
        });

        this.socket.on('disconnect', () => {
            console.log('❌ WebSocket disconnected');
            this.updateConnectionStatus('disconnected', 'Disconnected');
        });

        this.socket.on('environmental_update', (data) => {
            this.handleEnvironmentalUpdate(data);
        });

        this.socket.on('connection_status', (data) => {
            this.updateConnectionStatus(data.status, data.message);
        });

        this.socket.on('monitoring_status', (data) => {
            this.updateMonitoringStatus(data.status, data.message);
        });

        this.socket.on('error', (data) => {
            this.showAlert(data.message, 'danger');
        });
    }

    initializeCharts() {
        // Temperature chart
        this.charts.temperature = this.createLineChart('temperature-chart', 'Temperature (°C)', '#198754');
        
        // Humidity chart
        this.charts.humidity = this.createLineChart('humidity-chart', 'Humidity (%)', '#0dcaf0');
        
        console.log('✅ Charts initialized');
    }

    createLineChart(canvasId, label, color) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) {
            console.error(`Canvas element ${canvasId} not found`);
            return null;
        }

        return new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: label,
                    data: [],
                    borderColor: color,
                    backgroundColor: color + '20',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 3,
                    pointHoverRadius: 5
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#fff',
                        bodyColor: '#fff',
                        borderColor: color,
                        borderWidth: 1
                    }
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Time'
                        },
                        grid: {
                            color: 'rgba(0, 0, 0, 0.1)'
                        }
                    },
                    y: {
                        display: true,
                        title: {
                            display: true,
                            text: label
                        },
                        grid: {
                            color: 'rgba(0, 0, 0, 0.1)'
                        }
                    }
                },
                interaction: {
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false
                },
                animation: {
                    duration: 750,
                    easing: 'easeInOutQuart'
                }
            }
        });
    }

    setupEventListeners() {
        // Start monitoring button
        const startBtn = document.getElementById('start-monitoring');
        if (startBtn) {
            startBtn.addEventListener('click', () => this.startMonitoring());
        }

        // Stop monitoring button
        const stopBtn = document.getElementById('stop-monitoring');
        if (stopBtn) {
            stopBtn.addEventListener('click', () => this.stopMonitoring());
        }

        // Refresh data button
        const refreshBtn = document.getElementById('refresh-data');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.refreshData());
        }

        // Export data button
        const exportBtn = document.getElementById('export-data');
        if (exportBtn) {
            exportBtn.addEventListener('click', () => this.exportData());
        }

        // Fullscreen map button
        const fullscreenBtn = document.getElementById('fullscreen-map');
        if (fullscreenBtn) {
            fullscreenBtn.addEventListener('click', () => this.toggleFullscreenMap());
        }

        // Parameter selection
        const paramSelect = document.getElementById('parameter-select');
        if (paramSelect) {
            paramSelect.addEventListener('change', (e) => this.updateEnvironmentalMap(e.target.value));
        }

        // Time range selection
        const timeRangeSelect = document.getElementById('time-range');
        if (timeRangeSelect) {
            timeRangeSelect.addEventListener('change', (e) => this.updateTimeRange(e.target.value));
        }

        // Window resize handler
        window.addEventListener('resize', () => this.handleResize());
    }

    startPeriodicUpdates() {
        this.updateInterval = setInterval(() => {
            this.updateSystemStatus();
            this.updateTimestamp();
        }, 1000);
    }

    async loadInitialData() {
        try {
            // Load system status
            const statusResponse = await fetch('/api/status');
            if (statusResponse.ok) {
                const status = await statusResponse.json();
                this.updateSystemStatusDisplay(status);
            }

            // Load current environmental data
            const dataResponse = await fetch('/api/environmental_data');
            if (dataResponse.ok) {
                const data = await dataResponse.json();
                this.handleEnvironmentalUpdate(data);
            }

            // Load historical data
            const historyResponse = await fetch('/api/historical_data');
            if (historyResponse.ok) {
                const history = await historyResponse.json();
                this.dataHistory = history.data || [];
                this.updateChartsWithHistory();
            }

        } catch (error) {
            console.error('❌ Error loading initial data:', error);
            this.showAlert('Failed to load initial data. Check connection.', 'warning');
        }
    }

    handleEnvironmentalUpdate(data) {
        if (!data) return;

        console.log('📊 Environmental update received:', data);
        
        // Update status cards
        this.updateStatusCards(data);
        
        // Update charts
        this.updateCharts(data);
        
        // Update environmental map
        this.updateEnvironmentalMap();
        
        // Store in history
        this.addToHistory(data);
        
        // Update timestamp
        this.lastUpdate = new Date();
        this.updateTimestamp();
        
        // Update system status
        this.systemStatus = 'operational';
    }

    updateStatusCards(data) {
        // Temperature
        const tempElement = document.getElementById('temperature-value');
        if (tempElement && data.temperature !== undefined) {
            tempElement.textContent = `${data.temperature.toFixed(1)}°C`;
            this.updateTrend('temperature-trend', data.temperature);
        }

        // Humidity
        const humidityElement = document.getElementById('humidity-value');
        if (humidityElement && data.humidity !== undefined) {
            humidityElement.textContent = `${data.humidity.toFixed(1)}%`;
            this.updateTrend('humidity-trend', data.humidity);
        }

        // pH Level
        const phElement = document.getElementById('ph-value');
        if (phElement && data.ph !== undefined) {
            phElement.textContent = data.ph.toFixed(2);
            this.updateTrend('ph-trend', data.ph);
        }

        // Moisture
        const moistureElement = document.getElementById('moisture-value');
        if (moistureElement && data.moisture !== undefined) {
            moistureElement.textContent = `${data.moisture.toFixed(1)}%`;
            this.updateTrend('moisture-trend', data.moisture);
        }
    }

    updateTrend(elementId, value) {
        const element = document.getElementById(elementId);
        if (!element) return;

        // Simple trend calculation (you can enhance this)
        const trend = value > 50 ? '↗️' : value < 30 ? '↘️' : '→';
        element.textContent = trend;
    }

    updateCharts(data) {
        const timestamp = new Date().toLocaleTimeString();
        
        // Update temperature chart
        if (this.charts.temperature && data.temperature !== undefined) {
            this.updateChart(this.charts.temperature, timestamp, data.temperature);
        }
        
        // Update humidity chart
        if (this.charts.humidity && data.humidity !== undefined) {
            this.updateChart(this.charts.humidity, timestamp, data.humidity);
        }
    }

    updateChart(chart, label, value) {
        if (!chart) return;

        // Add new data point
        chart.data.labels.push(label);
        chart.data.datasets[0].data.push(value);

        // Keep only last N data points
        if (chart.data.labels.length > this.maxDataPoints) {
            chart.data.labels.shift();
            chart.data.datasets[0].data.shift();
        }

        // Update chart
        chart.update('none');
    }

    updateChartsWithHistory() {
        if (this.dataHistory.length === 0) return;

        // Process historical data for charts
        const timestamps = [];
        const temperatures = [];
        const humidities = [];

        this.dataHistory.forEach(item => {
            const time = new Date(item.timestamp).toLocaleTimeString();
            timestamps.push(time);
            
            if (item.data.temperature !== undefined) {
                temperatures.push(item.data.temperature);
            }
            
            if (item.data.humidity !== undefined) {
                humidities.push(item.data.humidity);
            }
        });

        // Update charts with historical data
        if (this.charts.temperature && temperatures.length > 0) {
            this.charts.temperature.data.labels = timestamps;
            this.charts.temperature.data.datasets[0].data = temperatures;
            this.charts.temperature.update();
        }

        if (this.charts.humidity && humidities.length > 0) {
            this.charts.humidity.data.labels = timestamps;
            this.charts.humidity.data.datasets[0].data = humidities;
            this.charts.humidity.update();
        }
    }

    addToHistory(data) {
        this.dataHistory.push({
            timestamp: new Date().toISOString(),
            data: data
        });

        // Keep only last N data points
        if (this.dataHistory.length > this.maxDataPoints) {
            this.dataHistory.shift();
        }
    }

    updateEnvironmentalMap(parameter = 'temperature') {
        const mapElement = document.getElementById('environmental-map');
        if (!mapElement) return;

        // Simple visualization for now - you can enhance this with D3.js or other libraries
        const value = this.getCurrentParameterValue(parameter);
        if (value !== null) {
            mapElement.innerHTML = `
                <div class="d-flex align-items-center justify-content-center h-100">
                    <div class="text-center">
                        <i class="fas fa-map-marked-alt fa-3x mb-3 text-primary"></i>
                        <h4>${parameter.charAt(0).toUpperCase() + parameter.slice(1)} Map</h4>
                        <p class="h2 text-primary">${value}</p>
                        <small class="text-muted">Current ${parameter} value</small>
                    </div>
                </div>
            `;
        }
    }

    getCurrentParameterValue(parameter) {
        if (this.dataHistory.length === 0) return null;
        
        const latestData = this.dataHistory[this.dataHistory.length - 1];
        return latestData.data[parameter];
    }

    async startMonitoring() {
        try {
            if (this.socket) {
                this.socket.emit('start_monitoring');
            } else {
                const response = await fetch('/api/start_monitoring', { method: 'POST' });
                if (response.ok) {
                    const result = await response.json();
                    this.showAlert(result.message, 'success');
                    this.monitoringActive = true;
                    this.updateMonitoringStatusDisplay();
                }
            }
        } catch (error) {
            console.error('❌ Error starting monitoring:', error);
            this.showAlert('Failed to start monitoring', 'danger');
        }
    }

    async stopMonitoring() {
        try {
            if (this.socket) {
                this.socket.emit('stop_monitoring');
            } else {
                const response = await fetch('/api/stop_monitoring', { method: 'POST' });
                if (response.ok) {
                    const result = await response.json();
                    this.showAlert(result.message, 'success');
                    this.monitoringActive = false;
                    this.updateMonitoringStatusDisplay();
                }
            }
        } catch (error) {
            console.error('❌ Error stopping monitoring:', error);
            this.showAlert('Failed to stop monitoring', 'danger');
        }
    }

    async refreshData() {
        try {
            await this.loadInitialData();
            this.showAlert('Data refreshed successfully', 'success');
        } catch (error) {
            console.error('❌ Error refreshing data:', error);
            this.showAlert('Failed to refresh data', 'danger');
        }
    }

    exportData() {
        try {
            const dataStr = JSON.stringify(this.dataHistory, null, 2);
            const dataBlob = new Blob([dataStr], { type: 'application/json' });
            const url = URL.createObjectURL(dataBlob);
            
            const link = document.createElement('a');
            link.href = url;
            link.download = `environmental_data_${new Date().toISOString().split('T')[0]}.json`;
            link.click();
            
            URL.revokeObjectURL(url);
            this.showAlert('Data exported successfully', 'success');
        } catch (error) {
            console.error('❌ Error exporting data:', error);
            this.showAlert('Failed to export data', 'danger');
        }
    }

    toggleFullscreenMap() {
        const mapElement = document.getElementById('environmental-map');
        if (!mapElement) return;

        if (!document.fullscreenElement) {
            mapElement.requestFullscreen().catch(err => {
                console.error('Error entering fullscreen:', err);
            });
        } else {
            document.exitFullscreen();
        }
    }

    updateTimeRange(range) {
        console.log('Time range changed to:', range);
        // Implement time range filtering logic here
        this.showAlert(`Time range updated to: ${range}`, 'info');
    }

    updateConnectionStatus(status, message) {
        const indicator = document.getElementById('connection-indicator');
        const statusText = document.getElementById('connection-status');
        
        if (indicator) {
            indicator.className = `fas fa-circle text-${status === 'connected' ? 'success' : 'danger'} me-1`;
        }
        
        if (statusText) {
            statusText.textContent = message;
        }
    }

    updateMonitoringStatus(status, message) {
        this.monitoringActive = status === 'started';
        this.updateMonitoringStatusDisplay();
        this.showAlert(message, status === 'started' ? 'success' : 'warning');
    }

    updateMonitoringStatusDisplay() {
        const statusElement = document.getElementById('monitoring-status');
        if (statusElement) {
            statusElement.textContent = this.monitoringActive ? 'Active' : 'Inactive';
            statusElement.className = `badge ${this.monitoringActive ? 'bg-success' : 'bg-secondary'}`;
        }
    }

    updateSystemStatus() {
        // Update system uptime
        const uptimeElement = document.getElementById('system-uptime');
        if (uptimeElement) {
            const uptime = this.calculateUptime();
            uptimeElement.textContent = uptime;
        }
    }

    updateSystemStatusDisplay(status) {
        const statusElement = document.getElementById('system-status');
        if (statusElement) {
            statusElement.textContent = status.status;
            statusElement.className = `badge bg-${status.status === 'operational' ? 'success' : 'warning'}`;
        }

        const clientsElement = document.getElementById('connected-clients');
        if (clientsElement) {
            clientsElement.textContent = status.connected_clients || 0;
        }

        const dataPointsElement = document.getElementById('data-points');
        if (dataPointsElement) {
            dataPointsElement.textContent = status.data_points_available || 0;
        }
    }

    updateTimestamp() {
        const timestampElement = document.getElementById('last-update');
        const footerTimestampElement = document.getElementById('footer-timestamp');
        
        const timeString = this.lastUpdate.toLocaleTimeString();
        
        if (timestampElement) {
            timestampElement.textContent = timeString;
        }
        
        if (footerTimestampElement) {
            footerTimestampElement.textContent = `Last updated: ${timeString}`;
        }
    }

    calculateUptime() {
        const now = new Date();
        const diff = now - this.lastUpdate;
        const hours = Math.floor(diff / (1000 * 60 * 60));
        const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
        const seconds = Math.floor((diff % (1000 * 60)) / 1000);
        
        return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    }

    showAlert(message, type = 'info') {
        const alertsContainer = document.getElementById('alerts-container');
        if (!alertsContainer) return;

        const alertElement = document.createElement('div');
        alertElement.className = `alert alert-${type} alert-dismissible fade show`;
        alertElement.innerHTML = `
            <i class="fas fa-${this.getAlertIcon(type)} me-2"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;

        alertsContainer.appendChild(alertElement);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alertElement.parentNode) {
                alertElement.remove();
            }
        }, 5000);
    }

    getAlertIcon(type) {
        const icons = {
            'success': 'check-circle',
            'warning': 'exclamation-triangle',
            'danger': 'times-circle',
            'info': 'info-circle'
        };
        return icons[type] || 'info-circle';
    }

    handleResize() {
        // Resize charts if needed
        Object.values(this.charts).forEach(chart => {
            if (chart && chart.resize) {
                chart.resize();
            }
        });
    }

    destroy() {
        // Cleanup
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
        
        if (this.socket) {
            this.socket.disconnect();
        }
        
        Object.values(this.charts).forEach(chart => {
            if (chart && chart.destroy) {
                chart.destroy();
            }
        });
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.environmentalDashboard = new EnvironmentalDashboard();
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (window.environmentalDashboard) {
        window.environmentalDashboard.destroy();
    }
}); 