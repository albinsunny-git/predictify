import sys
import os

def test_ml_pipeline():
    print("Testing ML Pipeline initialization...")
    try:
        import ml_engine
    except ImportError as e:
        print(f"FAILED: Cannot import ml_engine. Error: {e}")
        sys.exit(1)
        
    print("ml_engine loaded successfully. Running predictions for 'AAPL' (Apple Inc.)...")
    
    try:
        # Run predictions
        results = ml_engine.get_predictions("AAPL")
        
        # Verify fields in the returned dictionary
        required_keys = ['ticker', 'dates', 'close', 'indicators', 'fitted', 'forecast', 'metrics']
        for key in required_keys:
            assert key in results, f"Missing key '{key}' in prediction response"
            
        print(f"Ticker fetched: {results['ticker']}")
        print(f"Data points collected: {len(results['dates'])}")
        print(f"Dates shape: {len(results['dates'])}, Close shape: {len(results['close'])}")
        
        # Verify Indicators
        indicators = results['indicators']
        for name, values in indicators.items():
            print(f" - Indicator '{name}': {len(values)} points")
            # Verify no entirely null columns
            non_empty = [v for v in values if v != ""]
            assert len(non_empty) > 0, f"Indicator {name} is entirely empty"
            
        # Verify predictions
        fitted = results['fitted']
        print(f"Linear Regression Fitted points: {len(fitted['lr'])}")
        print(f"Random Forest Fitted points: {len(fitted['rf'])}")
        print(f"LSTM Fitted points: {len(fitted['lstm'])}")
        
        forecast = results['forecast']
        print(f"Forecast Days: {len(forecast['dates'])}")
        assert len(forecast['lr']) == 10, "LR forecast does not have 10 values"
        assert len(forecast['rf']) == 10, "RF forecast does not have 10 values"
        assert len(forecast['lstm']) == 10, "LSTM forecast does not have 10 values"
        
        # Verify Metrics
        metrics = results['metrics']
        print("Model Metrics:")
        for model in ['lr', 'rf', 'lstm']:
            print(f" - {model.upper()} - RMSE: {metrics[model]['rmse']:.2f}, MAE: {metrics[model]['mae']:.2f}, R2: {metrics[model]['r2']:.4f}")
            assert metrics[model]['rmse'] >= 0, f"{model} RMSE is negative!"
            
        print("\nSUCCESS: All pipeline and model checks passed successfully!")
        
    except Exception as e:
        import traceback
        print(f"FAILED: Exception occurred during pipeline test: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_ml_pipeline()
