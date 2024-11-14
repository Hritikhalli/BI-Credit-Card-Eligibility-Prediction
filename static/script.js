document.getElementById("chartForm").addEventListener("submit", function(event) {
    event.preventDefault();

    // Get form data
    const visualizationType = document.getElementById("visualizationType").value;
    const graphType = document.getElementById("graphType").value;

    // Create request payload
    const requestData = {
        visualizationType: visualizationType,
        graphType: graphType
    };

    // Send POST request to Flask app
    fetch('/generate-chart', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
    })
    .then(response => response.blob())
    .then(imageBlob => {
        // Create a local URL for the image and display it
        const chartImageUrl = URL.createObjectURL(imageBlob);
        document.getElementById("chartImage").src = chartImageUrl;
    })
    .catch(error => console.error('Error:', error));
});
