let myChart; // Global variable to store the chart instance

async function getRecs() {
    const song = document.getElementById('songInput').value;
    const response = await fetch('/recommend', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({song_name: song})
    });
    const data = await response.json();

    // 1. Draw Radar Chart for the first recommendation
    const ctx = document.getElementById('radarChart').getContext('2d');
    const labels = Object.keys(data.input_song);
    
    if (myChart) myChart.destroy(); // Clear old chart
    
    myChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Selected Song',
                data: Object.values(data.input_song),
                backgroundColor: 'rgba(29, 185, 84, 0.2)',
                borderColor: '#1DB954',
            }, {
                label: 'Top Match',
                data: labels.map(key => data.recommendations[0][key]),
                backgroundColor: 'rgba(255, 255, 255, 0.2)',
                borderColor: '#ffffff',
            }]
        },
        options: { scales: { r: { grid: { color: '#444' }, ticks: { display: false }, min: 0, max: 1 }}}
    });

    // 2. Display the list as before...
}