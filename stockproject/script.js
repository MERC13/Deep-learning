document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('stock-form');
    let chartInstance = null;
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        const symbol = document.getElementById('stock-symbol').value;
        const startDate = document.getElementById('start-date').value;
        const endDate = document.getElementById('end-date').value;
        
        // Send data to backend
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ symbol, startDate, endDate }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                document.getElementById('prediction-result').textContent = `Error: ${data.error}`;
                return;
            }
            document.getElementById('prediction-result').textContent = `Predicted next close: ${data.prediction.toFixed(2)}`;
            renderChart(data.chartData);
        });
    });
    
    function renderChart(chartData) {
        const ctx = document.getElementById('prediction-chart').getContext('2d');
        if (chartInstance) {
            chartInstance.destroy();
        }
        chartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: chartData.labels,
                datasets: [
                    {
                        label: 'Actual',
                        data: chartData.actual,
                        borderColor: 'rgba(54, 162, 235, 1)',
                        backgroundColor: 'rgba(54, 162, 235, 0.2)',
                        tension: 0.2
                    },
                    {
                        label: 'Predicted',
                        data: chartData.predicted,
                        borderColor: 'rgba(255, 99, 132, 1)',
                        backgroundColor: 'rgba(255, 99, 132, 0.2)',
                        tension: 0.2
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { display: true, title: { display: true, text: 'Date' } },
                    y: { display: true, title: { display: true, text: 'Price' } }
                }
            }
        });
    }
});