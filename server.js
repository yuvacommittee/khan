const express = require('express');
const axios = require('axios');
const { PythonShell } = require('python-shell');
const app = express();
const port = 3000;

// Middleware to serve static files
app.use(express.static('frontend'));

// Route to get the stock prediction
app.get('/predict', async (req, res) => {
    try {
        // Fetch live stock data from Yahoo Finance (Reliance stock)
        const response = await axios.get('https://query1.finance.yahoo.com/v7/finance/quote?symbols=RELIANCE.NS');
        const stockPrice = response.data.quoteResponse.result[0].regularMarketPrice;

        // Run Python script for prediction
        const options = {
            args: [stockPrice]
        };

        PythonShell.run('backend/stock_prediction.py', options, (err, result) => {
            if (err) throw err;
            res.json({ currentPrice: stockPrice, prediction: result[0] });
        });
    } catch (error) {
        console.error(error);
        res.status(500).send('Error fetching stock data or running prediction');
    }
});

// Start the server
app.listen(port, () => {
    console.log(Server running at http://localhost:${port});
});