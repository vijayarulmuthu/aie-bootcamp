import logger from '../utils/logger.js';
import ragService from './ragService.js';
import visualization from './visualization.js';

class RAGUI {
    constructor() {
        logger.info('Initializing RAGUI');
        // Wait for DOM to be fully loaded
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => {
                this.initializeElements();
                this.attachEventListeners();
                this.currentTaskId = null;
                this.setDefaultQueries();
            });
        } else {
            this.initializeElements();
            this.attachEventListeners();
            this.currentTaskId = null;
            this.setDefaultQueries();
        }
    }

    initializeElements() {
        logger.info('Initializing elements');
        // Upload elements
        this.uploadInput = document.getElementById('documentUpload');
        this.uploadButton = document.getElementById('uploadButton');
        this.uploadProgress = document.getElementById('uploadProgress');
        this.uploadStatus = document.getElementById('uploadStatus');

        // Query elements
        this.queryInput = document.getElementById('queryInput');
        this.processButton = document.getElementById('processButton');
        this.topKInput = document.getElementById('topKInput');
        this.progressBar = document.getElementById('progressBar');
        this.statusMessage = document.getElementById('statusMessage');

        // Modal elements
        this.errorModal = document.getElementById('errorModal');
        this.closeButton = document.querySelector('.close-button');

        // Log element initialization
        logger.info('Elements initialized', {
            uploadButton: !!this.uploadButton,
            uploadInput: !!this.uploadInput,
            processButton: !!this.processButton,
            queryInput: !!this.queryInput,
            topKInput: !!this.topKInput,
            progressBar: !!this.progressBar,
            statusMessage: !!this.statusMessage,
            errorModal: !!this.errorModal,
            closeButton: !!this.closeButton
        });
    }

    setDefaultQueries() {
        const defaultQueries = [
            "What is prompt engineering?",
            "What are the different types of prompting techniques?",
            "What is chain of thought prompting?",
            "How does temperature affect LLM outputs?",
            "What are the best practices for prompt engineering?"
        ];
        
        if (this.queryInput) {
            this.queryInput.value = defaultQueries.join('\n');
            logger.info('Default queries set');
        }
    }

    attachEventListeners() {
        logger.info('Attaching event listeners');
        
        // Upload button click
        if (this.uploadButton) {
            this.uploadButton.addEventListener('click', (event) => {
                logger.info('Upload button clicked');
                this.handleUpload();
            });
        } else {
            logger.error('Upload button not found');
        }

        // Process button click
        if (this.processButton) {
            this.processButton.addEventListener('click', (event) => {
                logger.info('Process button clicked');
                this.handleProcess();
            });
        } else {
            logger.error('Process button not found');
        }

        // Close modal button click
        if (this.closeButton) {
            this.closeButton.addEventListener('click', () => {
                this.errorModal.classList.add('hidden');
            });
        }

        // Close modal when clicking outside
        window.addEventListener('click', (event) => {
            if (event.target === this.errorModal) {
                this.errorModal.classList.add('hidden');
            }
        });

        logger.info('Event listeners attached');
    }

    async handleUpload() {
        try {
            logger.info('Handling upload');
            const files = this.uploadInput.files;
            if (!files.length) {
                throw new Error('No files selected');
            }

            // Show progress bar
            this.uploadProgress.classList.remove('hidden');
            this.uploadStatus.textContent = 'Uploading...';
            this.uploadStatus.className = 'status-message';

            // Upload files
            const result = await ragService.uploadDocuments(files);
            const taskId = result.task_id;

            // Start polling for progress
            await this.pollUploadProgress(taskId);

        } catch (error) {
            this.uploadStatus.textContent = `Upload failed: ${error.message}`;
            this.uploadStatus.className = 'status-message error';
            logger.error('Document upload failed', error);
        }
    }

    async pollUploadProgress(taskId) {
        const maxAttempts = 60; // 60 attempts * 1 second = 1 minute timeout
        let attempts = 0;

        while (attempts < maxAttempts) {
            try {
                const response = await fetch(`${ragService.baseUrl}/documents/progress/${taskId}`);
                if (!response.ok) {
                    throw new Error(`Failed to get progress: ${response.statusText}`);
                }

                const progress = await response.json();
                logger.info('Upload progress:', progress);

                // Update progress bar
                const progressFill = this.uploadProgress.querySelector('.progress-fill');
                progressFill.style.width = `${progress.progress}%`;

                // Update status message
                this.uploadStatus.textContent = `Uploading... ${progress.progress}%`;

                if (progress.status === 'completed') {
                    this.uploadStatus.textContent = 'Upload successful!';
                    this.uploadStatus.className = 'status-message success';
                    return;
                }

                if (progress.status === 'failed') {
                    throw new Error(progress.error || 'Upload failed');
                }

                // Wait before next attempt
                await new Promise(resolve => setTimeout(resolve, 1000));
                attempts++;

            } catch (error) {
                logger.error('Error polling upload progress:', error);
                throw error;
            }
        }

        throw new Error('Upload timeout');
    }

    async handleProcess() {
        try {
            // Disable process button and show progress
            this.processButton.disabled = true;
            this.processButton.textContent = "Processing...";
            
            // Show progress bar and status
            this.progressBar.style.display = "block";
            this.progressBar.value = 0;
            this.statusMessage.style.display = "block";
            this.statusMessage.textContent = "Starting query processing...";
            this.statusMessage.className = "status-message";
            
            // Get queries and top_k value
            const queries = this.queryInput.value.split("\n").filter(q => q.trim());
            const top_k = this.topKInput ? parseInt(this.topKInput.value) || 3 : 3;
            
            if (!queries.length) {
                throw new Error('No queries entered');
            }

            // Start processing queries
            const taskInfo = await ragService.processQueries(queries, top_k);
            
            // Poll for progress
            const maxAttempts = 60;  // 1 minute timeout
            let attempt = 0;
            
            const updateProgressWithAnimation = (progress, message) => {
                // Calculate smooth progress steps
                const currentValue = this.progressBar.value;
                const steps = 10;
                const stepSize = (progress - currentValue) / steps;
                
                let currentStep = 0;
                const animate = () => {
                    if (currentStep < steps) {
                        this.progressBar.value += stepSize;
                        currentStep++;
                        requestAnimationFrame(animate);
                    } else {
                        this.progressBar.value = progress;
                    }
                };
                requestAnimationFrame(animate);
                
                // Update status message
                this.statusMessage.textContent = message;
            };
            
            while (attempt < maxAttempts) {
                try {
                    const progress = await ragService.getTaskProgress(taskInfo.task_id);
                    
                    if (progress.queries_progress) {
                        const totalQueries = progress.queries_progress.length;
                        const completedQueries = progress.queries_progress.filter(q => q.completed).length;
                        const currentQuery = progress.queries_progress.find(q => q.in_progress);
                        
                        let statusMessage = `Processing queries (${completedQueries}/${totalQueries})`;
                        
                        if (currentQuery) {
                            const queryIndex = progress.queries_progress.indexOf(currentQuery);
                            statusMessage += ` - Query ${queryIndex + 1}: ${currentQuery.query}`;
                            if (currentQuery.progress > 0) {
                                statusMessage += ` (${currentQuery.progress}%)`;
                            }
                        }
                        
                        updateProgressWithAnimation(progress.progress, statusMessage);
                    }
                    
                    if (progress.status === "completed") {
                        updateProgressWithAnimation(100, "Processing completed!");
                        this.statusMessage.className = "status-message success";
                        
                        // Update results if available
                        if (progress.faiss_db && progress.simple_db) {
                            this.updateResults(progress);
                        }
                        break;
                    } else if (progress.status === "failed") {
                        throw new Error(progress.error || "Task failed");
                    }
                    
                    await new Promise(resolve => setTimeout(resolve, 500)); // Poll every 500ms
                    attempt++;
                    
                } catch (error) {
                    throw error;
                }
            }
            
            if (attempt >= maxAttempts) {
                throw new Error("Request timeout: Processing took too long");
            }
            
        } catch (error) {
            logger.error("Error processing queries:", error);
            this.statusMessage.textContent = `Error: ${error.message}`;
            this.statusMessage.className = "status-message error";
            this.showError(error.message);
        } finally {
            // Re-enable process button
            this.processButton.disabled = false;
            this.processButton.textContent = "Process Queries";
        }
    }

    updateProgress(value, message) {
        // Smoothly update progress bar
        const currentValue = this.progressBar.value;
        const step = (value - currentValue) / 10;
        let current = currentValue;
        
        const animate = () => {
            current += step;
            if ((step > 0 && current <= value) || (step < 0 && current >= value)) {
                this.progressBar.value = current;
                requestAnimationFrame(animate);
            } else {
                this.progressBar.value = value;
            }
        };
        
        requestAnimationFrame(animate);
        
        // Update status message
        this.statusMessage.textContent = message;
    }

    showError(message) {
        const errorMessage = document.getElementById('errorMessage');
        errorMessage.textContent = message;
        this.errorModal.classList.remove('hidden');
    }

    async pollResults() {
        try {
            const maxAttempts = 60; // 60 attempts * 1 second = 1 minute timeout
            let attempts = 0;

            while (attempts < maxAttempts) {
                const result = await ragService.getResults(this.currentTaskId);

                // Update progress bar
                const progressFill = this.uploadProgress.querySelector('.progress-fill');
                progressFill.style.width = `${result.progress}%`;
                this.uploadStatus.textContent = `Processing... ${result.progress}%`;

                if (result.status === 'completed') {
                    // Update visualizations
                    visualization.updateVisualizations(
                        result.faiss_db.results,
                        result.simple_db.results
                    );

                    // Update UI
                    this.uploadStatus.textContent = 'Processing complete!';
                    this.uploadStatus.className = 'status-message success';
                    this.uploadProgress.classList.add('hidden');

                    // Reset UI
                    this.processButton.disabled = false;
                    this.processButton.textContent = 'Process Queries';
                    return;
                }

                if (result.status === 'failed') {
                    throw new Error(result.error || 'Processing failed');
                }

                // Wait before next attempt
                await new Promise(resolve => setTimeout(resolve, 1000));
                attempts++;
            }

            throw new Error('Timeout waiting for results');

        } catch (error) {
            logger.error('Failed to get results', error);
            this.uploadStatus.textContent = `Error: ${error.message}`;
            this.uploadStatus.className = 'status-message error';
            this.processButton.disabled = false;
            this.processButton.textContent = 'Process Queries';
        }
    }

    updateResults(results) {
        try {
            logger.info('Updating results in UI', results);
            
            // Store queries for reference
            this.queries = results.queries_progress.map(q => q.query);
            
            // Clear previous results
            const resultsContainer = document.getElementById('resultsContainer');
            if (!resultsContainer) {
                throw new Error('Results container not found');
            }
            resultsContainer.innerHTML = '';

            // Create charts section
            const chartsSection = document.createElement('div');
            chartsSection.className = 'charts-section';
            chartsSection.innerHTML = '<h3>Performance Comparison</h3>';
            resultsContainer.appendChild(chartsSection);

            // Process each query and create charts
            this.queries.forEach((query, queryIndex) => {
                // Create chart container
                const chartContainer = document.createElement('div');
                chartContainer.className = 'chart-container';
                chartContainer.id = `chart-${queryIndex}`;
                chartsSection.appendChild(chartContainer);

                // Prepare data for the chart
                const faissResults = results.faiss_db.results[queryIndex] || [];
                const simpleResults = results.simple_db.results[queryIndex] || [];

                const traces = [];

                // Add FAISS data
                if (faissResults.length > 0) {
                    traces.push({
                        x: faissResults.map(r => {
                            const method = r.search_method || 'unknown';
                            // Clean up the method name
                            return method.replace('SearchMethod.', '').replace(/_/g, ' ');
                        }),
                        y: faissResults.map(r => Number((r.search_time || 0).toFixed(4))),
                        name: 'FAISS',
                        type: 'bar',
                        marker: {
                            color: '#3498db',
                            line: {
                                color: '#2980b9',
                                width: 1
                            }
                        }
                    });
                }

                // Add SimpleDB data
                if (simpleResults.length > 0) {
                    traces.push({
                        x: simpleResults.map(r => {
                            const method = r.search_method || 'unknown';
                            // Clean up the method name
                            return method.replace('SearchMethod.', '').replace(/_/g, ' ');
                        }),
                        y: simpleResults.map(r => Number((r.search_time || 0).toFixed(4))),
                        name: 'SimpleDB',
                        type: 'bar',
                        marker: {
                            color: '#2ecc71',
                            line: {
                                color: '#27ae60',
                                width: 1
                            }
                        }
                    });
                }

                // Wait for container to be in DOM
                setTimeout(() => {
                    const chartDiv = document.getElementById(`chart-${queryIndex}`);
                    if (chartDiv && traces.length > 0) {
                        // Create the plot
                        const layout = {
                            title: {
                                text: `Query: "${query}"`,
                                font: { 
                                    size: 16,
                                    color: '#2c3e50'
                                }
                            },
                            xaxis: {
                                title: {
                                    text: 'Search Method',
                                    font: {
                                        size: 14,
                                        color: '#34495e'
                                    }
                                },
                                tickangle: -45,
                                tickfont: { 
                                    size: 12,
                                    color: '#2c3e50'
                                },
                                gridcolor: '#ecf0f1'
                            },
                            yaxis: {
                                title: {
                                    text: 'Search Time (seconds)',
                                    font: {
                                        size: 14,
                                        color: '#34495e'
                                    }
                                },
                                tickfont: { 
                                    size: 12,
                                    color: '#2c3e50'
                                },
                                gridcolor: '#ecf0f1',
                                zeroline: true,
                                zerolinecolor: '#95a5a6',
                                zerolinewidth: 1,
                                rangemode: 'tozero'
                            },
                            barmode: 'group',
                            bargap: 0.15,
                            bargroupgap: 0.1,
                            height: 500,
                            margin: {
                                l: 80,
                                r: 30,
                                b: 100,
                                t: 80,
                                pad: 4
                            },
                            plot_bgcolor: '#ffffff',
                            paper_bgcolor: '#ffffff',
                            showlegend: true,
                            legend: {
                                orientation: 'h',
                                yanchor: 'bottom',
                                y: 1.02,
                                xanchor: 'right',
                                x: 1,
                                font: {
                                    size: 12,
                                    color: '#2c3e50'
                                }
                            }
                        };

                        Plotly.newPlot(chartDiv, traces, layout, {
                            responsive: true,
                            displayModeBar: true,
                            displaylogo: false,
                            modeBarButtonsToRemove: ['lasso2d', 'select2d']
                        }).catch(error => {
                            logger.error('Error plotting chart:', error);
                        });
                    }
                }, 100);
            });
            
            // Create results sections
            const faissSection = this.createResultsSection('FAISS Results', results.faiss_db.results);
            const simpleSection = this.createResultsSection('Simple DB Results', results.simple_db.results);
            
            // Add sections to container
            resultsContainer.appendChild(faissSection);
            resultsContainer.appendChild(simpleSection);
            
            // Show results container
            resultsContainer.style.display = 'block';
            
        } catch (error) {
            logger.error('Error updating results:', error);
            this.showError(`Failed to update results: ${error.message}`);
        }
    }
    
    createResultsSection(dbName, results) {
        const section = document.createElement('div');
        section.className = 'results-section';
        section.innerHTML = `<h3>${dbName}</h3>`;

        if (!Array.isArray(results) || results.length === 0) {
            section.innerHTML += '<p>No results available</p>';
            return section;
        }

        // Create a scrollable container for the table
        const tableContainer = document.createElement('div');
        tableContainer.className = 'table-scroll-container';

        // Create a single table for all queries
        const table = document.createElement('table');
        table.className = 'results-table';
        
        // Create header row
        const headerRow = document.createElement('tr');
        headerRow.innerHTML = `
            <th>Query</th>
            <th>Method</th>
            <th>Search Time (s)</th>
            <th>Scores</th>
            <th>Context</th>
            <th>Response</th>
        `;
        table.appendChild(headerRow);

        // Process all queries and their results
        results.forEach((queryResults, queryIndex) => {
            // Get the query text from the progress data
            const queryText = this.queries[queryIndex] || `Query ${queryIndex + 1}`;

            // Handle each method result for this query
            queryResults.forEach(result => {
                const row = document.createElement('tr');
                
                if (result.error) {
                    // Handle error case
                    row.innerHTML = `
                        <td>${this.escapeHtml(queryText)}</td>
                        <td>${result.search_method || 'N/A'}</td>
                        <td colspan="4" class="error-cell">Error: ${this.escapeHtml(result.error)}</td>
                    `;
                } else {
                    // Handle successful result
                    const searchTime = result.search_time ? result.search_time.toFixed(3) : 'N/A';
                    
                    // Format all scores
                    let scoreDisplay = 'N/A';
                    if (Array.isArray(result.scores) && result.scores.length > 0) {
                        scoreDisplay = result.scores.map(s => s.toFixed(3)).join(', ');
                    } else if (result.score !== undefined && result.score !== null) {
                        scoreDisplay = result.score.toFixed(3);
                    }
                    
                    // Format context and response with proper line breaks and styling
                    const formattedContext = this.formatContext(result.context || 'No context');
                    const formattedResponse = this.formatResponse(result.response || 'No response');
                    
                    row.innerHTML = `
                        <td>${this.escapeHtml(queryText)}</td>
                        <td>${result.search_method || 'N/A'}</td>
                        <td>${searchTime}</td>
                        <td class="scores-cell">${scoreDisplay}</td>
                        <td class="context-cell">${formattedContext}</td>
                        <td class="response-cell">${formattedResponse}</td>
                    `;
                }
                
                table.appendChild(row);
            });
        });

        // If no results were added to the table, show a message
        if (table.children.length <= 1) {
            const noResultsRow = document.createElement('tr');
            noResultsRow.innerHTML = `
                <td colspan="6" class="no-results">No results available</td>
            `;
            table.appendChild(noResultsRow);
        }

        // Add table to scrollable container
        tableContainer.appendChild(table);
        section.appendChild(tableContainer);
        return section;
    }

    formatContext(context) {
        if (!context) return 'No context';
        
        // Split the context into sections if it contains a summary header
        const parts = context.split('=== Context Summary ===');
        if (parts.length > 1) {
            const [summary, details] = parts;
            return `
                <div class="context-content">
                    <div class="context-summary">
                        <strong>Summary:</strong>
                        <div>${this.formatTextWithLineBreaks(summary.trim())}</div>
                    </div>
                    <div class="context-details">
                        <strong>Details:</strong>
                        <div>${this.formatTextWithLineBreaks(details.trim())}</div>
                    </div>
                </div>
            `;
        }
        
        // Handle chunk information if present
        const chunkMatch = context.match(/Chunk (\d+)\/(\d+)\s*\|\s*Score:\s*([\d.]+)\s*\|\s*Length:\s*(\d+)/);
        if (chunkMatch) {
            const [fullMatch, chunkNum, totalChunks, score, length] = chunkMatch;
            const mainContent = context.replace(fullMatch, '').trim();
            return `
                <div class="context-content">
                    <div class="chunk-info">
                        <span>Chunk ${chunkNum}/${totalChunks}</span>
                        <span>Score: ${score}</span>
                        <span>Length: ${length}</span>
                    </div>
                    <div class="chunk-content">
                        ${this.formatTextWithLineBreaks(mainContent)}
                    </div>
                </div>
            `;
        }
        
        // Default formatting for other cases
        return `<div class="context-content">${this.formatTextWithLineBreaks(context)}</div>`;
    }

    formatTextWithLineBreaks(text) {
        if (!text) return '';
        return this.escapeHtml(text)
            .replace(/\n/g, '<br>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/===(.*?)===/g, '<mark>$1</mark>');
    }

    escapeHtml(unsafe) {
        if (!unsafe) return '';
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    formatResponse(response) {
        if (!response) return 'No response';
        
        // Create a wrapper div for the response
        return `
            <div class="response-content">
                ${this.formatTextWithLineBreaks(response)}
            </div>
        `;
    }
}

// Initialize UI when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    logger.info('DOM content loaded');
    const ui = new RAGUI();
});
