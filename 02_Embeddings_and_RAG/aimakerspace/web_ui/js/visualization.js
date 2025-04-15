import logger from '../utils/logger.js';

class Visualization {
    constructor() {
        this.chartConfig = {
            displayModeBar: false,
            responsive: true
        };
    }

    updateTable(tableId, data) {
        try {
            const table = document.getElementById(tableId);
            if (!table) {
                throw new Error(`Table ${tableId} not found`);
            }

            const tbody = table.querySelector('tbody');
            tbody.innerHTML = '';

            data.forEach(item => {
                const row = document.createElement('tr');
                
                // Query cell
                const queryCell = document.createElement('td');
                queryCell.textContent = item.query;
                row.appendChild(queryCell);

                // Search method cell
                const methodCell = document.createElement('td');
                methodCell.textContent = item.search_method;
                row.appendChild(methodCell);

                // Response cell
                const responseCell = document.createElement('td');
                responseCell.textContent = item.response;
                row.appendChild(responseCell);

                tbody.appendChild(row);
            });

            logger.info(`Table ${tableId} updated successfully`);

        } catch (error) {
            logger.error(`Failed to update table ${tableId}`, error);
            throw error;
        }
    }

    createBarChart(containerId, data) {
        try {
            const container = document.getElementById(containerId);
            if (!container) {
                throw new Error(`Chart container ${containerId} not found`);
            }

            // Prepare data for Plotly
            const searchMethods = [...new Set(data.map(item => item.search_method))];
            const queries = [...new Set(data.map(item => item.query))];

            const traces = queries.map(query => {
                const queryData = data.filter(item => item.query === query);
                return {
                    x: searchMethods,
                    y: searchMethods.map(method => {
                        const result = queryData.find(item => item.search_method === method);
                        return result ? result.search_time : 0;
                    }),
                    name: query,
                    type: 'bar'
                };
            });

            const layout = {
                title: 'Search Time by Method',
                xaxis: {
                    title: 'Search Method'
                },
                yaxis: {
                    title: 'Search Time (seconds)'
                },
                barmode: 'group'
            };

            Plotly.newPlot(container, traces, layout, this.chartConfig);
            logger.info(`Chart ${containerId} created successfully`);

        } catch (error) {
            logger.error(`Failed to create chart ${containerId}`, error);
            throw error;
        }
    }

    updateVisualizations(faissData, simpleData) {
        try {
            // Update FAISS table and chart
            this.updateTable('faissTable', faissData);
            this.createBarChart('faissChart', faissData);

            // Update Simple DB table and chart
            this.updateTable('simpleTable', simpleData);
            this.createBarChart('simpleChart', simpleData);

            logger.info('All visualizations updated successfully');

        } catch (error) {
            logger.error('Failed to update visualizations', error);
            throw error;
        }
    }
}

// Create singleton instance
const visualization = new Visualization();

// Export the visualization instance
export default visualization;
