class Logger {
    constructor() {
        this.logs = [];
        this.maxLogs = 1000; // Maximum number of logs to keep in memory
    }

    log(level, message, data = null) {
        const timestamp = new Date().toISOString();
        const logEntry = {
            timestamp,
            level,
            message,
            data
        };

        // Add to logs array
        this.logs.push(logEntry);

        // Trim logs if they exceed maxLogs
        if (this.logs.length > this.maxLogs) {
            this.logs = this.logs.slice(-this.maxLogs);
        }

        // Log to console
        console.log(`[${timestamp}] [${level}] ${message}`, data || '');
    }

    info(message, data = null) {
        this.log('INFO', message, data);
    }

    warn(message, data = null) {
        this.log('WARN', message, data);
    }

    error(message, data = null) {
        this.log('ERROR', message, data);
        // Show error in UI
        this.showError(message);
    }

    showError(message) {
        const errorModal = document.getElementById('errorModal');
        const errorMessage = document.getElementById('errorMessage');
        
        if (errorModal && errorMessage) {
            errorMessage.textContent = message;
            errorModal.classList.remove('hidden');
        }
    }

    getLogs() {
        return this.logs;
    }

    clearLogs() {
        this.logs = [];
    }
}

// Create singleton instance
const logger = new Logger();

// Export the logger instance
export default logger;
