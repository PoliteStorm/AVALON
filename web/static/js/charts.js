/**
 * Charts Management for Environmental Monitoring Dashboard
 * Handles real-time chart updates and data visualization
 */

class ChartManager {
    constructor() {
        this.charts = {};
        this.chartConfigs = {};
        this.dataBuffers = {};
        this.maxDataPoints = 100;
        this.updateInterval = null;
        this.isInitialized = false;
        
        this.initialize();
    }

    initialize() {
        console.log('📊 Initializing Chart Manager...');
        
        // Set up chart configurations
        this.setupChartConfigs();
        
        // Initialize charts
        this.initializeCharts();
        
        // Start update loop
        this.startUpdateLoop();
        
        this.isInitialized = true;
        console.log('✅ Chart Manager initialized');
    }

    setupChartConfigs() {
        this.chartConfigs = {
            temperature: {
                label: 'Temperature (°C)',
                color: '#198754',
                backgroundColor: '#19875420',
                borderColor: '#198754',
                yAxisMin: -10,
                yAxisMax: 50,
                unit: '°C',
                precision: 1
            },
            humidity: {
                label: 'Humidity (%)',
                color: '#0dcaf0',
                backgroundColor: '#0dcaf020',
                borderColor: '#0dcaf0',
                yAxisMin: 0,
                yAxisMax: 100,
                unit: '%',
                precision: 1
            },
            ph: {
                label: 'pH Level',
                color: '#ffc107',
                backgroundColor: '#ffc10720',
                borderColor: '#ffc107',
                yAxisMin: 0,
                yAxisMax: 14,
                unit: '',
                precision: 2
            },
            moisture: {
                label: 'Moisture (%)',
                color: '#6c757d',
                backgroundColor: '#6c757d20',
                borderColor: '#6c757d',
                yAxisMin: 0,
                yAxisMax: 100,
                unit: '%',
                precision: 1
            },
            pollution: {
                label: 'Pollution Level (ppm)',
                color: '#dc3545',
                backgroundColor: '#dc354520',
                borderColor: '#dc3545',
                yAxisMin: 0,
                yAxisMax: 1000,
                unit: 'ppm',
                precision: 2
            }
        };
    }

    initializeCharts() {
        // Initialize temperature chart
        this.initializeChart('temperature', 'temperature-chart');
        
        // Initialize humidity chart
        this.initializeChart('humidity', 'humidity-chart');
        
        // Initialize pH chart
        this.initializeChart('ph', 'ph-chart');
        
        // Initialize moisture chart
        this.initializeChart('moisture', 'moisture-chart');
        
        // Initialize pollution chart
        this.initializeChart('pollution', 'pollution-chart');
        
        console.log('📊 All charts initialized');
    }

    initializeChart(type, canvasId) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) {
            console.warn(`⚠️ Canvas element ${canvasId} not found for ${type} chart`);
            return;
        }

        const config = this.chartConfigs[type];
        if (!config) {
            console.error(`❌ No configuration found for chart type: ${type}`);
            return;
        }

        // Initialize data buffer
        this.dataBuffers[type] = {
            labels: [],
            values: [],
            timestamps: []
        };

        // Create Chart.js instance
        this.charts[type] = new Chart(canvas, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: config.label,
                    data: [],
                    borderColor: config.borderColor,
                    backgroundColor: config.backgroundColor,
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 3,
                    pointHoverRadius: 5,
                    pointBackgroundColor: config.color,
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2
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
                        borderColor: config.borderColor,
                        borderWidth: 1,
                        callbacks: {
                            label: (context) => {
                                const value = context.parsed.y;
                                return `${config.label}: ${value.toFixed(config.precision)}${config.unit}`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Time',
                            color: '#666',
                            font: {
                                size: 12
                            }
                        },
                        grid: {
                            color: 'rgba(0, 0, 0, 0.1)',
                            drawBorder: false
                        },
                        ticks: {
                            color: '#666',
                            maxTicksLimit: 8,
                            callback: (value, index, values) => {
                                if (this.dataBuffers[type].labels[index]) {
                                    return this.dataBuffers[type].labels[index];
                                }
                                return '';
                            }
                        }
                    },
                    y: {
                        display: true,
                        title: {
                            display: true,
                            text: config.label,
                            color: '#666',
                            font: {
                                size: 12
                            }
                        },
                        grid: {
                            color: 'rgba(0, 0, 0, 0.1)',
                            drawBorder: false
                        },
                        ticks: {
                            color: '#666',
                            callback: (value) => {
                                return `${value.toFixed(config.precision)}${config.unit}`;
                            }
                        },
                        min: config.yAxisMin,
                        max: config.yAxisMax
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
                },
                elements: {
                    point: {
                        hoverRadius: 6
                    }
                }
            }
        });

        console.log(`✅ ${type} chart initialized`);
    }

    updateChart(type, value, timestamp = null) {
        if (!this.charts[type] || !this.dataBuffers[type]) {
            console.warn(`⚠️ Chart or buffer not found for type: ${type}`);
            return;
        }

        const config = this.chartConfigs[type];
        if (!config) {
            console.error(`❌ No configuration found for chart type: ${type}`);
            return;
        }

        // Validate value
        if (value === null || value === undefined || isNaN(value)) {
            console.warn(`⚠️ Invalid value for ${type}: ${value}`);
            return;
        }

        // Format timestamp
        const timeLabel = timestamp ? new Date(timestamp).toLocaleTimeString() : new Date().toLocaleTimeString();
        
        // Add to data buffer
        this.dataBuffers[type].labels.push(timeLabel);
        this.dataBuffers[type].values.push(value);
        this.dataBuffers[type].timestamps.push(timestamp || Date.now());

        // Keep only last N data points
        if (this.dataBuffers[type].labels.length > this.maxDataPoints) {
            this.dataBuffers[type].labels.shift();
            this.dataBuffers[type].values.shift();
            this.dataBuffers[type].timestamps.shift();
        }

        // Update chart data
        this.charts[type].data.labels = this.dataBuffers[type].labels;
        this.charts[type].data.datasets[0].data = this.dataBuffers[type].values;

        // Update chart
        this.charts[type].update('none');
    }

    updateMultipleCharts(data) {
        if (!data || typeof data !== 'object') {
            console.warn('⚠️ Invalid data for chart update:', data);
            return;
        }

        const timestamp = data.timestamp || Date.now();
        
        // Update each chart with available data
        Object.keys(this.chartConfigs).forEach(type => {
            if (data[type] !== undefined && data[type] !== null) {
                this.updateChart(type, data[type], timestamp);
            }
        });
    }

    addDataPoint(type, value, timestamp = null) {
        this.updateChart(type, value, timestamp);
    }

    clearChart(type) {
        if (!this.charts[type] || !this.dataBuffers[type]) {
            return;
        }

        // Clear data buffers
        this.dataBuffers[type].labels = [];
        this.dataBuffers[type].values = [];
        this.dataBuffers[type].timestamps = [];

        // Clear chart
        this.charts[type].data.labels = [];
        this.charts[type].data.datasets[0].data = [];
        this.charts[type].update();
    }

    clearAllCharts() {
        Object.keys(this.charts).forEach(type => {
            this.clearChart(type);
        });
    }

    setMaxDataPoints(maxPoints) {
        this.maxDataPoints = Math.max(10, Math.min(1000, maxPoints));
        
        // Trim existing data if necessary
        Object.keys(this.dataBuffers).forEach(type => {
            if (this.dataBuffers[type].labels.length > this.maxDataPoints) {
                const excess = this.dataBuffers[type].labels.length - this.maxDataPoints;
                this.dataBuffers[type].labels.splice(0, excess);
                this.dataBuffers[type].values.splice(0, excess);
                this.dataBuffers[type].timestamps.splice(0, excess);
                
                // Update chart
                this.charts[type].data.labels = this.dataBuffers[type].labels;
                this.charts[type].data.datasets[0].data = this.dataBuffers[type].values;
                this.charts[type].update();
            }
        });
    }

    getChartData(type) {
        if (!this.dataBuffers[type]) {
            return null;
        }
        
        return {
            labels: [...this.dataBuffers[type].labels],
            values: [...this.dataBuffers[type].values],
            timestamps: [...this.dataBuffers[type].timestamps]
        };
    }

    getAllChartData() {
        const allData = {};
        Object.keys(this.dataBuffers).forEach(type => {
            allData[type] = this.getChartData(type);
        });
        return allData;
    }

    exportChartData(type, format = 'json') {
        const data = this.getChartData(type);
        if (!data) {
            console.error(`❌ No data available for export: ${type}`);
            return null;
        }

        switch (format.toLowerCase()) {
            case 'json':
                return JSON.stringify(data, null, 2);
            
            case 'csv':
                const csvContent = [
                    'Timestamp,Value,Label',
                    ...data.timestamps.map((timestamp, index) => 
                        `${new Date(timestamp).toISOString()},${data.values[index]},${data.labels[index]}`
                    )
                ].join('\n');
                return csvContent;
            
            case 'array':
                return data;
            
            default:
                console.error(`❌ Unsupported export format: ${format}`);
                return null;
        }
    }

    exportAllChartData(format = 'json') {
        const allData = this.getAllChartData();
        
        switch (format.toLowerCase()) {
            case 'json':
                return JSON.stringify(allData, null, 2);
            
            case 'csv':
                let csvContent = 'Chart,Timestamp,Value,Label\n';
                Object.keys(allData).forEach(type => {
                    if (allData[type]) {
                        allData[type].timestamps.forEach((timestamp, index) => {
                            csvContent += `${type},${new Date(timestamp).toISOString()},${allData[type].values[index]},${allData[type].labels[index]}\n`;
                        });
                    }
                });
                return csvContent;
            
            case 'array':
                return allData;
            
            default:
                console.error(`❌ Unsupported export format: ${format}`);
                return null;
        }
    }

    startUpdateLoop() {
        // Update charts every second for smooth animations
        this.updateInterval = setInterval(() => {
            this.updateChartAnimations();
        }, 1000);
    }

    updateChartAnimations() {
        // Add subtle animations to charts
        Object.keys(this.charts).forEach(type => {
            if (this.charts[type] && this.dataBuffers[type].values.length > 0) {
                // Add subtle glow effect to active charts
                const lastValue = this.dataBuffers[type].values[this.dataBuffers[type].values.length - 1];
                const config = this.chartConfigs[type];
                
                if (config && lastValue !== undefined) {
                    // Update point colors based on value ranges
                    this.updatePointColors(type, lastValue);
                }
            }
        });
    }

    updatePointColors(type, value) {
        if (!this.charts[type]) return;
        
        const config = this.chartConfigs[type];
        const dataset = this.charts[type].data.datasets[0];
        
        // Update point colors based on value ranges
        if (type === 'temperature') {
            if (value < 0) {
                dataset.pointBackgroundColor = '#0d6efd'; // Blue for cold
            } else if (value > 30) {
                dataset.pointBackgroundColor = '#dc3545'; // Red for hot
            } else {
                dataset.pointBackgroundColor = config.color; // Normal color
            }
        } else if (type === 'humidity') {
            if (value < 30) {
                dataset.pointBackgroundColor = '#ffc107'; // Yellow for dry
            } else if (value > 70) {
                dataset.pointBackgroundColor = '#0dcaf0'; // Blue for wet
            } else {
                dataset.pointBackgroundColor = config.color; // Normal color
            }
        }
        
        // Update chart
        this.charts[type].update('none');
    }

    resizeCharts() {
        Object.values(this.charts).forEach(chart => {
            if (chart && chart.resize) {
                chart.resize();
            }
        });
    }

    destroy() {
        console.log('🧹 Cleaning up Chart Manager...');
        
        // Stop update loop
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
        
        // Destroy all charts
        Object.values(this.charts).forEach(chart => {
            if (chart && chart.destroy) {
                chart.destroy();
            }
        });
        
        // Clear references
        this.charts = {};
        this.dataBuffers = {};
        this.isInitialized = false;
    }

    // Utility methods
    getChartCount() {
        return Object.keys(this.charts).length;
    }

    getActiveCharts() {
        return Object.keys(this.charts).filter(type => 
            this.charts[type] && this.dataBuffers[type].values.length > 0
        );
    }

    getChartStatistics(type) {
        if (!this.dataBuffers[type] || this.dataBuffers[type].values.length === 0) {
            return null;
        }
        
        const values = this.dataBuffers[type].values;
        const sum = values.reduce((a, b) => a + b, 0);
        const avg = sum / values.length;
        const min = Math.min(...values);
        const max = Math.max(...values);
        
        return {
            count: values.length,
            sum: sum,
            average: avg,
            minimum: min,
            maximum: max,
            range: max - min
        };
    }

    getAllChartStatistics() {
        const stats = {};
        Object.keys(this.dataBuffers).forEach(type => {
            stats[type] = this.getChartStatistics(type);
        });
        return stats;
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ChartManager;
} 