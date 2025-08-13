/**
 * WebSocket Client for Environmental Monitoring Dashboard
 * Handles real-time communication with the server
 */

class WebSocketClient {
    constructor() {
        this.socket = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        this.isConnected = false;
        this.messageQueue = [];
        this.eventHandlers = new Map();
        
        this.initialize();
    }

    initialize() {
        console.log('🔌 Initializing WebSocket client...');
        
        // Set up event handlers
        this.setupEventHandlers();
        
        // Connect to server
        this.connect();
        
        // Start heartbeat
        this.startHeartbeat();
    }

    setupEventHandlers() {
        // Connection events
        this.on('connect', this.handleConnect.bind(this));
        this.on('disconnect', this.handleDisconnect.bind(this));
        this.on('connect_error', this.handleConnectError.bind(this));
        this.on('reconnect', this.handleReconnect.bind(this));
        this.on('reconnect_attempt', this.handleReconnectAttempt.bind(this));
        this.on('reconnect_error', this.handleReconnectError.bind(this));
        this.on('reconnect_failed', this.handleReconnectFailed.bind(this));
        
        // Environmental data events
        this.on('environmental_update', this.handleEnvironmentalUpdate.bind(this));
        this.on('connection_status', this.handleConnectionStatus.bind(this));
        this.on('monitoring_status', this.handleMonitoringStatus.bind(this));
        this.on('error', this.handleError.bind(this));
        
        // Custom events
        this.on('data_request', this.handleDataRequest.bind(this));
        this.on('system_status', this.handleSystemStatus.bind(this));
    }

    connect() {
        try {
            console.log('🔌 Attempting to connect to WebSocket server...');
            
            // Create Socket.IO connection
            this.socket = io({
                transports: ['websocket', 'polling'],
                upgrade: true,
                rememberUpgrade: true,
                timeout: 20000,
                forceNew: true
            });
            
            // Set up Socket.IO event handlers
            this.setupSocketIOHandlers();
            
        } catch (error) {
            console.error('❌ WebSocket connection failed:', error);
            this.handleConnectionError(error);
        }
    }

    setupSocketIOHandlers() {
        if (!this.socket) return;

        // Connection events
        this.socket.on('connect', () => {
            console.log('✅ WebSocket connected successfully');
            this.isConnected = true;
            this.reconnectAttempts = 0;
            this.emit('connect');
            
            // Process queued messages
            this.processMessageQueue();
        });

        this.socket.on('disconnect', (reason) => {
            console.log('❌ WebSocket disconnected:', reason);
            this.isConnected = false;
            this.emit('disconnect', reason);
            
            if (reason === 'io server disconnect') {
                // Server initiated disconnect, try to reconnect
                this.socket.connect();
            }
        });

        this.socket.on('connect_error', (error) => {
            console.error('❌ WebSocket connection error:', error);
            this.emit('connect_error', error);
        });

        // Custom events
        this.socket.on('environmental_update', (data) => {
            this.emit('environmental_update', data);
        });

        this.socket.on('connection_status', (data) => {
            this.emit('connection_status', data);
        });

        this.socket.on('monitoring_status', (data) => {
            this.emit('monitoring_status', data);
        });

        this.socket.on('error', (data) => {
            this.emit('error', data);
        });

        this.socket.on('system_status', (data) => {
            this.emit('system_status', data);
        });
    }

    emit(event, data) {
        // Emit to local event handlers
        if (this.eventHandlers.has(event)) {
            this.eventHandlers.get(event).forEach(handler => {
                try {
                    handler(data);
                } catch (error) {
                    console.error(`❌ Error in event handler for ${event}:`, error);
                }
            });
        }
        
        // Emit to server if connected
        if (this.socket && this.isConnected) {
            this.socket.emit(event, data);
        } else {
            // Queue message for later
            this.messageQueue.push({ event, data, timestamp: Date.now() });
        }
    }

    on(event, handler) {
        if (!this.eventHandlers.has(event)) {
            this.eventHandlers.set(event, []);
        }
        this.eventHandlers.get(event).push(handler);
    }

    off(event, handler) {
        if (this.eventHandlers.has(event)) {
            const handlers = this.eventHandlers.get(event);
            const index = handlers.indexOf(handler);
            if (index > -1) {
                handlers.splice(index, 1);
            }
        }
    }

    send(event, data) {
        this.emit(event, data);
    }

    disconnect() {
        if (this.socket) {
            this.socket.disconnect();
        }
        this.isConnected = false;
    }

    // Event handlers
    handleConnect() {
        console.log('✅ WebSocket connection established');
        this.updateConnectionUI('connected');
        this.showConnectionStatus('Connected to environmental monitoring system', 'success');
    }

    handleDisconnect(reason) {
        console.log('❌ WebSocket disconnected:', reason);
        this.updateConnectionUI('disconnected');
        this.showConnectionStatus(`Disconnected: ${reason}`, 'warning');
        
        // Attempt to reconnect if not manually disconnected
        if (reason !== 'io client disconnect') {
            this.scheduleReconnect();
        }
    }

    handleConnectError(error) {
        console.error('❌ WebSocket connection error:', error);
        this.updateConnectionUI('error');
        this.showConnectionStatus('Connection failed. Retrying...', 'danger');
        this.scheduleReconnect();
    }

    handleReconnect() {
        console.log('🔄 WebSocket reconnected');
        this.updateConnectionUI('connected');
        this.showConnectionStatus('Reconnected successfully', 'success');
    }

    handleReconnectAttempt(attemptNumber) {
        console.log(`🔄 Reconnection attempt ${attemptNumber}/${this.maxReconnectAttempts}`);
        this.updateConnectionUI('connecting');
        this.showConnectionStatus(`Reconnecting... (${attemptNumber}/${this.maxReconnectAttempts})`, 'warning');
    }

    handleReconnectError(error) {
        console.error('❌ WebSocket reconnection error:', error);
        this.updateConnectionUI('error');
        this.showConnectionStatus('Reconnection failed', 'danger');
    }

    handleReconnectFailed() {
        console.error('❌ WebSocket reconnection failed after maximum attempts');
        this.updateConnectionUI('failed');
        this.showConnectionStatus('Connection failed permanently. Please refresh the page.', 'danger');
    }

    handleEnvironmentalUpdate(data) {
        console.log('📊 Environmental update received via WebSocket:', data);
        // This will be handled by the main dashboard
    }

    handleConnectionStatus(data) {
        console.log('🔗 Connection status update:', data);
        this.updateConnectionUI(data.status);
    }

    handleMonitoringStatus(data) {
        console.log('📡 Monitoring status update:', data);
        // This will be handled by the main dashboard
    }

    handleError(data) {
        console.error('❌ WebSocket error:', data);
        this.showConnectionStatus(`Error: ${data.message}`, 'danger');
    }

    handleDataRequest(data) {
        console.log('📋 Data request received:', data);
        // Handle data request from server
    }

    handleSystemStatus(data) {
        console.log('🖥️ System status update:', data);
        // Handle system status update
    }

    // Connection management
    scheduleReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.error('❌ Maximum reconnection attempts reached');
            this.handleReconnectFailed();
            return;
        }

        this.reconnectAttempts++;
        const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
        
        console.log(`🔄 Scheduling reconnection in ${delay}ms (attempt ${this.reconnectAttempts})`);
        
        setTimeout(() => {
            if (!this.isConnected) {
                this.connect();
            }
        }, delay);
    }

    processMessageQueue() {
        if (this.messageQueue.length === 0) return;

        console.log(`📨 Processing ${this.messageQueue.length} queued messages`);
        
        const now = Date.now();
        const maxAge = 30000; // 30 seconds
        
        // Process messages and remove old ones
        this.messageQueue = this.messageQueue.filter(item => {
            if (now - item.timestamp > maxAge) {
                console.log(`🗑️ Removing old queued message: ${item.event}`);
                return false;
            }
            
            if (this.isConnected) {
                console.log(`📤 Sending queued message: ${item.event}`);
                this.socket.emit(item.event, item.data);
                return false;
            }
            
            return true;
        });
    }

    // UI updates
    updateConnectionUI(status) {
        const indicator = document.getElementById('connection-indicator');
        const statusText = document.getElementById('connection-status');
        
        if (!indicator || !statusText) return;
        
        const statusConfig = {
            'connected': { class: 'text-success', text: 'Connected', icon: 'fas fa-circle' },
            'connecting': { class: 'text-warning', text: 'Connecting...', icon: 'fas fa-circle' },
            'disconnected': { class: 'text-danger', text: 'Disconnected', icon: 'fas fa-circle' },
            'error': { class: 'text-danger', text: 'Error', icon: 'fas fa-exclamation-triangle' },
            'failed': { class: 'text-danger', text: 'Failed', icon: 'fas fa-times-circle' }
        };
        
        const config = statusConfig[status] || statusConfig['error'];
        
        indicator.className = `${config.icon} ${config.class} me-1`;
        statusText.textContent = config.text;
        
        // Add animation for connecting state
        if (status === 'connecting') {
            indicator.style.animation = 'pulse 1s infinite';
        } else {
            indicator.style.animation = 'none';
        }
    }

    showConnectionStatus(message, type = 'info') {
        // Show status message in the alerts container
        const alertsContainer = document.getElementById('alerts-container');
        if (!alertsContainer) return;
        
        const alertElement = document.createElement('div');
        alertElement.className = `alert alert-${type} alert-dismissible fade show`;
        alertElement.innerHTML = `
            <i class="fas fa-${this.getStatusIcon(type)} me-2"></i>
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

    getStatusIcon(type) {
        const icons = {
            'success': 'check-circle',
            'warning': 'exclamation-triangle',
            'danger': 'times-circle',
            'info': 'info-circle'
        };
        return icons[type] || 'info-circle';
    }

    // Heartbeat mechanism
    startHeartbeat() {
        setInterval(() => {
            if (this.isConnected && this.socket) {
                this.socket.emit('ping');
            }
        }, 30000); // Send ping every 30 seconds
    }

    // Utility methods
    isConnected() {
        return this.isConnected;
    }

    getConnectionStatus() {
        return {
            connected: this.isConnected,
            reconnectAttempts: this.reconnectAttempts,
            maxReconnectAttempts: this.maxReconnectAttempts
        };
    }

    // Cleanup
    destroy() {
        console.log('🧹 Cleaning up WebSocket client...');
        
        if (this.socket) {
            this.socket.disconnect();
        }
        
        this.isConnected = false;
        this.eventHandlers.clear();
        this.messageQueue = [];
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = WebSocketClient;
} 