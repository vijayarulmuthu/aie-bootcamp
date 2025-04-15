import logger from '../utils/logger.js';

class RAGService {
    constructor() {
        this.baseUrl = 'http://localhost:8000/api'; // Update with your API URL
    }

    async uploadDocuments(files) {
        try {
            logger.info('Starting document upload');
            
            const formData = new FormData();
            // Convert FileList to Array and iterate
            Array.from(files).forEach(file => {
                formData.append('file', file);  // Changed from 'files' to 'file' to match FastAPI parameter
            });

            const response = await fetch(`${this.baseUrl}/documents/upload`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorText = await response.text();
                logger.error('Upload failed with response:', errorText);
                throw new Error(`Upload failed: ${response.statusText}`);
            }

            const data = await response.json();
            logger.info('Document upload successful', data);
            return data;

        } catch (error) {
            logger.error('Document upload failed', error);
            throw error;
        }
    }

    async processQueries(queries, top_k = 3) {
        try {
            logger.info('Processing queries', { queries, top_k });
            
            const response = await fetch(`${this.baseUrl}/rag/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ 
                    queries,
                    top_k
                })
            });

            if (!response.ok) {
                throw new Error(`Query processing failed: ${response.statusText}`);
            }

            const data = await response.json();
            logger.info('Query processing successful', data);
            return data;

        } catch (error) {
            logger.error('Query processing failed', error);
            throw error;
        }
    }

    async getResults(taskId) {
        try {
            logger.info('Fetching results', {taskId});
            
            const response = await fetch(`${this.baseUrl}/rag/results/${taskId}`);

            if (!response.ok) {
                throw new Error(`Failed to fetch results: ${response}`);
            }

            const data = await response.json();
            logger.info('Results fetched successfully', data);
            return data;

        } catch (error) {
            logger.error('Failed to fetch results', error);
            throw error;
        }
    }

    async getTaskProgress(taskId) {
        try {
            logger.info('Getting task progress', { taskId });
            
            const response = await fetch(`${this.baseUrl}/rag/${taskId}/progress`);
            
            if (!response.ok) {
                throw new Error(`Failed to get task progress: ${response.statusText}`);
            }

            const data = await response.json();
            
            // Calculate overall progress based on individual query progress
            if (data.queries_progress && Array.isArray(data.queries_progress)) {
                const totalQueries = data.queries_progress.length;
                const completedQueries = data.queries_progress.filter(q => q.completed).length;
                const inProgressQueries = data.queries_progress.filter(q => q.in_progress);
                
                let progress = (completedQueries / totalQueries) * 100;
                
                // Add partial progress for queries in progress
                inProgressQueries.forEach(query => {
                    progress += (query.progress / totalQueries);
                });
                
                data.progress = Math.min(Math.round(progress), 100);
            }
            
            logger.info('Task progress retrieved', data);
            return data;

        } catch (error) {
            logger.error('Error getting task progress:', error);
            throw error;
        }
    }
}

// Create singleton instance
const ragService = new RAGService();

// Export the service instance
export default ragService;
