import os
from flask import Flask, jsonify, request, render_template
from ml_engine import get_predictions

app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates"
)

# Supported popular tickers list for the frontend quick selectors
POPULAR_TICKERS = [
    {"symbol": "AAPL", "name": "Apple Inc."},
    {"symbol": "MSFT", "name": "Microsoft Corporation"},
    {"symbol": "GOOGL", "name": "Alphabet Inc."},
    {"symbol": "AMZN", "name": "Amazon.com, Inc."},
    {"symbol": "TSLA", "name": "Tesla, Inc."},
    {"symbol": "NVDA", "name": "NVIDIA Corporation"},
    {"symbol": "IBM", "name": "IBM Corporation"},
    {"symbol": "META", "name": "Meta Platforms, Inc."}
]

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
    return response

@app.route('/')
def home():
    """
    Renders the index page.
    """
    return render_template('index.html')

@app.route('/api/tickers')
def get_popular_tickers():
    """
    Returns the list of supported popular tickers.
    """
    return jsonify(POPULAR_TICKERS)

@app.route('/api/predict')
def predict_stock():
    """
    API endpoint that triggers data downloading, feature calculations,
    model training, evaluation, and future forecasting.
    Expects query parameter: ?ticker=SYMBOL
    """
    ticker = request.args.get('ticker', '').strip().upper()
    
    if not ticker:
        return jsonify({
            'status': 'error',
            'message': 'Please provide a valid stock ticker symbol (e.g., AAPL).'
        }), 400
        
    try:
        results = get_predictions(ticker)
        return jsonify({
            'status': 'success',
            'data': results
        })
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error training models for {ticker}: {e}\n{error_trace}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    # Ensure templates and static directories exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    # Run the web server
    print("Starting Flask Stock Price Prediction server on http://127.0.0.1:5000...")
    app.run(host='127.0.0.1', port=5000, debug=True)
