/**
 * Environmental Monitoring Dashboard - Enhanced JavaScript v3.0
 * Core functionality for real-time environmental monitoring with Plotly.js integration
 */

class EnvironmentalDashboard {
    constructor() {
        this.socket = null;
        this.dataHistory = [];
        this.maxDataPoints = 1000;
        this.monitoringActive = false;
        this.updateFrequency = 1;
        this.lastData = null;
        this.plotlyCharts = {};
        this.connectionStatus = 'disconnected';
        this.dataSource = 'unknown';
        this.dataQuality = 0;
        this.dataPointsReceived = 0;
        this.lastUpdateTime = null;
        
        // Performance optimization: batch updates
        this.updateQueue = [];
        this.isUpdating = false;
        this.updateInterval = null;
        
        console.log('🚀 Dashboard initialized - Clean & Clear version');
        this.initialize();
    }

    async initialize() {
        try {
            console.log('🔄 Initializing dashboard...');
            this.showProgress('Initializing dashboard...', 10);
            
            // Initialize WebSocket connection
            await this.initializeWebSocket();
            this.showProgress('WebSocket connected...', 30);
            
            // Initialize charts
            await this.initializeCharts();
            this.showProgress('Charts initialized...', 60);
            
            // Start data monitoring
            this.startDataMonitoring();
            this.showProgress('Data monitoring started...', 80);
            
            // Final setup
            this.setupEventListeners();
            this.showProgress('Setup complete...', 100);
            
            setTimeout(() => {
                this.hideProgress();
                console.log('✅ Dashboard initialization complete');
            }, 1000);
            
        } catch (error) {
            console.error('❌ Dashboard initialization failed:', error);
            this.showAlert('Dashboard initialization failed', 'error');
        }
    }

    async initializeWebSocket() {
        return new Promise((resolve, reject) => {
            try {
                this.socket = io();
                
                this.socket.on('connect', () => {
                    console.log('✅ WebSocket connected');
                    this.connectionStatus = 'connected';
                    this.updateConnectionStatus();
                    resolve();
                });
                
                this.socket.on('disconnect', () => {
                    console.log('❌ WebSocket disconnected');
                    this.connectionStatus = 'disconnected';
                    this.updateConnectionStatus();
                });
                
                this.socket.on('environmental_update', (data) => {
                    this.handleEnvironmentalUpdate(data);
                });
                
                this.socket.on('connect_error', (error) => {
                    console.error('❌ WebSocket connection error:', error);
                    reject(error);
                });
                
            } catch (error) {
                reject(error);
            }
        });
    }

    async initializeCharts() {
        try {
            // Initialize main chart with efficient configuration
            const chartConfig = {
                responsive: true,
                displayModeBar: true,
                modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
                displaylogo: false
            };
            
            const layout = {
                title: 'Environmental Monitoring Data',
                xaxis: { title: 'Time' },
                yaxis: { title: 'Values' },
                hovermode: 'closest',
                showlegend: true,
                legend: { x: 0, y: 1 },
                margin: { l: 50, r: 50, t: 50, b: 50 }
            };
            
            const traces = [
                { name: 'Temperature', type: 'scatter', mode: 'lines', line: { color: '#007bff' } },
                { name: 'Humidity', type: 'scatter', mode: 'lines', line: { color: '#17a2b8' } },
                { name: 'pH', type: 'scatter', mode: 'lines', line: { color: '#ffc107' } },
                { name: 'Moisture', type: 'scatter', mode: 'lines', line: { color: '#28a745' } },
                { name: 'Pollution', type: 'scatter', mode: 'lines', line: { color: '#dc3545' } },
                { name: 'Electrical', type: 'scatter', mode: 'lines', line: { color: '#6c757d' } }
            ];
            
            this.plotlyCharts.main = {
                element: document.getElementById('main-chart'),
                traces: traces,
                layout: layout,
                config: chartConfig
            };
            
            // Create initial empty chart
            Plotly.newPlot('main-chart', traces, layout, chartConfig);
            
            console.log('✅ Charts initialized with efficient configuration');
            
        } catch (error) {
            console.error('❌ Chart initialization failed:', error);
            throw error;
        }
    }

    startDataMonitoring() {
        this.monitoringActive = true;
        this.updateInterval = setInterval(() => {
            this.refreshEnvironmentalData();
        }, this.updateFrequency * 1000);
        
        console.log('🔄 Data monitoring started');
    }

    setupEventListeners() {
        // Efficient event delegation for chart controls
        document.addEventListener('change', (e) => {
            if (e.target.id === 'chart-type-select') {
                this.changeChartType(e.target.value);
            } else if (e.target.id === 'time-range-select') {
                this.updateChartsWithTimeRange(e.target.value);
            } else if (e.target.id === 'resolution-select') {
                this.changeChartResolution(e.target.value);
            }
        });
        
        console.log('✅ Event listeners configured');
    }

    // OPTIMIZED DATA HANDLING - Clear and Efficient
    handleEnvironmentalUpdate(data) {
        try {
            console.log('🔄 Processing environmental update:', data);
            
            // Update data source status immediately
            this.updateDataSourceStatus(data);
            
            // Add to history efficiently
            this.addToHistory(data);
            
            // Update current values display
            this.updateCurrentValues(data);
            
            // Queue chart update for performance
            this.queueChartUpdate();
            
            // Update metrics
            this.updateMetrics(data);
            
        } catch (error) {
            console.error('❌ Error handling environmental update:', error);
            this.showAlert('Error processing data update', 'error');
        }
    }

    updateDataSourceStatus(data) {
        const dataSourceBadge = document.getElementById('data-source-badge');
        const dataStatusMessage = document.getElementById('data-status-message');
        
        if (data.data_source === 'real_csv' || data.data_source === 'real_integrated') {
            // Real data available
            this.dataSource = 'real';
            dataSourceBadge.innerHTML = '<span class="badge bg-success">Real CSV Data</span>';
            
            dataStatusMessage.innerHTML = `
                <div class="alert alert-success alert-dismissible fade show">
                    <i class="fas fa-check-circle me-2"></i>
                    <strong>Real-time CSV data streaming successfully!</strong>
                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                </div>
            `;
            dataStatusMessage.style.display = 'block';
            
        } else if (data.data_source === 'fallback' || data.data_source === 'simulated') {
            // Using fallback data
            this.dataSource = 'simulated';
            dataSourceBadge.innerHTML = '<span class="badge bg-warning">Simulated Data</span>';
            
            dataStatusMessage.innerHTML = `
                <div class="alert alert-warning alert-dismissible fade show">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    <strong>Warning: Using simulated data</strong> - Real CSV data not available
                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                </div>
            `;
            dataStatusMessage.style.display = 'block';
            
        } else {
            // Unknown data source
            this.dataSource = 'unknown';
            dataSourceBadge.innerHTML = '<span class="badge bg-danger">Unknown Source</span>';
            
            dataStatusMessage.innerHTML = `
                <div class="alert alert-danger alert-dismissible fade show">
                    <i class="fas fa-exclamation-circle me-2"></i>
                    <strong>Error: Unknown data source</strong> - Data may not be reliable
                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                </div>
            `;
            dataStatusMessage.style.display = 'block';
        }
    }

    addToHistory(data) {
        // Add timestamp if not present
        if (!data.timestamp) {
            data.timestamp = new Date().toISOString();
        }
        
        // Add to history
        this.dataHistory.push({
            timestamp: data.timestamp,
            data: data
        });
        
        // Limit history size for performance
        if (this.dataHistory.length > this.maxDataPoints) {
            this.dataHistory = this.dataHistory.slice(-this.maxDataPoints);
        }
        
        this.dataPointsReceived = this.dataHistory.length;
        this.lastUpdateTime = new Date();
        
        // Update display
        document.getElementById('data-points-count').textContent = this.dataPointsReceived;
        document.getElementById('last-update').textContent = this.lastUpdateTime.toLocaleTimeString();
    }

    updateCurrentValues(data) {
        // Update current values display efficiently
        const updates = [
            { id: 'current-temperature', value: data.temperature, unit: '°C' },
            { id: 'current-humidity', value: data.humidity, unit: '%' },
            { id: 'current-ph', value: data.ph, unit: '' },
            { id: 'current-moisture', value: data.moisture, unit: '%' },
            { id: 'current-pollution', value: data.pollution, unit: 'ppm' },
            { id: 'current-electrical', value: data.electrical, unit: 'mV' }
        ];
        
        updates.forEach(update => {
            const element = document.getElementById(update.id);
            if (element && update.value !== undefined) {
                element.textContent = `${update.value.toFixed(2)}${update.unit}`;
            }
        });
    }

    // PERFORMANCE OPTIMIZED CHART UPDATES
    queueChartUpdate() {
        if (!this.isUpdating) {
            this.isUpdating = true;
            // Use requestAnimationFrame for smooth updates
            requestAnimationFrame(() => {
                this.updateChart();
                this.isUpdating = false;
            });
        }
    }

    updateChart() {
        if (!this.plotlyCharts.main || this.dataHistory.length === 0) return;
        
        try {
            const recentData = this.dataHistory.slice(-this.maxDataPoints);
            
            const timestamps = recentData.map(item => new Date(item.timestamp));
            const temperatures = recentData.map(item => item.data.temperature || 0);
            const humidities = recentData.map(item => item.data.humidity || 0);
            const phs = recentData.map(item => item.data.ph || 0);
            const moistures = recentData.map(item => item.data.moisture || 0);
            const pollutions = recentData.map(item => item.data.pollution || 0);
            const electricals = recentData.map(item => item.data.electrical || 0);
            
            // Efficient chart update using Plotly.restyle
            Plotly.restyle('main-chart', {
                x: [timestamps, timestamps, timestamps, timestamps, timestamps, timestamps],
                y: [temperatures, humidities, phs, moistures, pollutions, electricals]
            }, [0, 1, 2, 3, 4, 5]);
            
            // Update chart status
            document.getElementById('chart-status').textContent = 
                `Chart updated: ${recentData.length} data points (${this.dataSource === 'real' ? 'Real CSV' : 'Simulated'} data)`;
            
        } catch (error) {
            console.error('❌ Chart update failed:', error);
            document.getElementById('chart-status').textContent = 'Chart update failed';
        }
    }

    // EFFICIENT CHART TYPE CHANGES
    changeChartType(type) {
        console.log(`🔄 Changing chart type to: ${type}`);
        
        if (!this.plotlyCharts.main) return;
        
        try {
            const newTraces = this.plotlyCharts.main.traces.map(trace => ({
                ...trace,
                type: type === '3d' ? 'surface' : type
            }));
            
            Plotly.react('main-chart', newTraces, this.plotlyCharts.main.layout, this.plotlyCharts.main.config);
            
            document.getElementById('chart-status').textContent = 
                `Chart type changed to: ${type.charAt(0).toUpperCase() + type.slice(1)}`;
            
            console.log(`✅ Chart type changed to: ${type}`);
            
        } catch (error) {
            console.error(`❌ Error changing chart type: ${error}`);
            this.showAlert('Error changing chart type', 'error');
        }
    }

    // EFFICIENT TIME RANGE FILTERING
    updateChartsWithTimeRange(timeRange) {
        console.log(`🔄 Updating time range: ${timeRange}`);
        
        if (!this.plotlyCharts.main || this.dataHistory.length === 0) return;
        
        try {
            let filteredData = [];
            const now = new Date();
            
            switch (timeRange) {
                case '1h':
                    const oneHourAgo = new Date(now.getTime() - 60 * 60 * 1000);
                    filteredData = this.dataHistory.filter(data => new Date(data.timestamp) >= oneHourAgo);
                    break;
                case '6h':
                    const sixHoursAgo = new Date(now.getTime() - 6 * 60 * 60 * 1000);
                    filteredData = this.dataHistory.filter(data => new Date(data.timestamp) >= sixHoursAgo);
                    break;
                case '24h':
                    const oneDayAgo = new Date(now.getTime() - 24 * 60 * 60 * 1000);
                    filteredData = this.dataHistory.filter(data => new Date(data.timestamp) >= oneDayAgo);
                    break;
                case '7d':
                    const oneWeekAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
                    filteredData = this.dataHistory.filter(data => new Date(data.timestamp) >= oneWeekAgo);
                    break;
                case 'all':
                default:
                    filteredData = this.dataHistory;
                    break;
            }
            
            if (filteredData.length === 0) {
                this.showAlert('No data available for selected time range', 'warning');
                return;
            }
            
            // Update chart with filtered data
            this.updateChartWithData(filteredData);
            
            document.getElementById('chart-status').textContent = 
                `Time range: ${timeRange} (${filteredData.length} data points)`;
            
            console.log(`✅ Time range updated: ${filteredData.length} data points`);
            
        } catch (error) {
            console.error(`❌ Error updating time range: ${error}`);
            this.showAlert('Error updating time range', 'error');
        }
    }

    updateChartWithData(data) {
        if (!this.plotlyCharts.main) return;
        
        const timestamps = data.map(item => new Date(item.timestamp));
        const temperatures = data.map(item => item.data.temperature || 0);
        const humidities = data.map(item => item.data.humidity || 0);
        const phs = data.map(item => item.data.ph || 0);
        const moistures = data.map(item => item.data.moisture || 0);
        const pollutions = data.map(item => item.data.pollution || 0);
        const electricals = data.map(item => item.data.electrical || 0);
        
        Plotly.restyle('main-chart', {
            x: [timestamps, timestamps, timestamps, timestamps, timestamps, timestamps],
            y: [temperatures, humidities, phs, moistures, pollutions, electricals]
        }, [0, 1, 2, 3, 4, 5]);
    }

    // EFFICIENT RESOLUTION CHANGES
    changeChartResolution(resolution) {
        console.log(`🔄 Changing chart resolution to: ${resolution}`);
        
        this.maxDataPoints = parseInt(resolution);
        
        // Update chart immediately with new resolution
        this.updateChart();
        
        document.getElementById('chart-status').textContent = 
            `Resolution: ${this.maxDataPoints} data points`;
        
        console.log(`✅ Chart resolution changed to: ${this.maxDataPoints} points`);
    }

    // UTILITY METHODS
    updateConnectionStatus() {
        const indicator = document.getElementById('connection-indicator');
        const status = document.getElementById('connection-status');
        
        if (this.connectionStatus === 'connected') {
            indicator.className = 'fas fa-circle text-success';
            status.textContent = 'Connected';
        } else {
            indicator.className = 'fas fa-circle text-danger';
            status.textContent = 'Disconnected';
        }
    }

    updateMetrics(data) {
        // Update data quality display
        if (data.data_quality !== undefined) {
            this.dataQuality = data.data_quality;
            document.getElementById('data-quality-display').textContent = `${this.dataQuality}%`;
        }
        
        // Update frequency display
        document.getElementById('update-frequency').textContent = `${this.updateFrequency}s`;
    }

    refreshEnvironmentalData() {
        if (this.socket && this.socket.connected) {
            this.socket.emit('request_environmental_data');
            console.log('🔄 Requested environmental data refresh');
        } else {
            console.warn('⚠️ Socket not connected, cannot refresh data');
        }
    }

    showProgress(message, progress) {
        const container = document.getElementById('progress-container');
        const messageEl = document.getElementById('progress-message');
        const bar = document.getElementById('progress-bar');
        
        if (container && messageEl && bar) {
            messageEl.textContent = message;
            bar.style.width = `${progress}%`;
            container.style.display = 'block';
        }
    }

    hideProgress() {
        const container = document.getElementById('progress-container');
        if (container) {
            container.style.display = 'none';
        }
    }

    showAlert(message, type = 'info') {
        const container = document.getElementById('alert-container');
        if (!container) return;
        
        const alertId = 'alert-' + Date.now();
        const alertHtml = `
            <div id="${alertId}" class="alert alert-${type} alert-dismissible fade show">
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;
        
        container.insertAdjacentHTML('beforeend', alertHtml);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            const alert = document.getElementById(alertId);
            if (alert) {
                alert.remove();
            }
        }, 5000);
    }
}

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new EnvironmentalDashboard();
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (window.dashboard) {
        window.dashboard.destroy();
    }
}); 