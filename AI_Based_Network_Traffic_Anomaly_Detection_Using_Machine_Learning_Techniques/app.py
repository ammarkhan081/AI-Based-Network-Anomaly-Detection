# app.py - Enhanced Flask Application for AI Traffic Analysis

from flask import Flask, render_template, request, redirect, url_for, send_file, flash, jsonify
import os
import pandas as pd
import uuid
import json
from datetime import datetime

from backend import (
    load_dataset,
    preprocess_training_data,
    preprocess_testing_data,
    train_model,
    evaluate_model,
    predict_sample
)

app = Flask(__name__)
app.secret_key = 'ai-traffic-analysis-secret-key-2024'  # Updated secret key

# Create necessary folders
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
os.makedirs(os.path.join(STATIC_FOLDER, 'css'), exist_ok=True)

# Global storage for models and data
model_store = {}
scaler_store = {}
label_encoder_store = {}
performance_metrics = {}
predicted_csv_path = ""
training_history = []
current_model_name = None  # Add this to track current model

# Model names mapping for display
MODEL_NAMES = {
    'rf': 'Random Forest',
    'lr': 'Logistic Regression',
    'svm': 'Support Vector Machine',
    'dt': 'Decision Tree',
    'xgb': 'XGBoost',
    'lgb': 'LightGBM'
}

# ===================== UTILITY FUNCTIONS =====================

def allowed_file(filename):
    """Check if uploaded file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'

def save_performance_metrics(model_name, metrics):
    """Save performance metrics for later display"""
    global current_model_name
    current_model_name = model_name  # Update current model
    performance_metrics[model_name] = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_name': MODEL_NAMES.get(model_name, model_name),
        'metrics': metrics
    }

def log_training_event(model_name, status, details=None):
    """Log training events for history tracking"""
    event = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model': MODEL_NAMES.get(model_name, model_name),
        'status': status,
        'details': details or {}
    }
    training_history.append(event)
    # Keep only last 50 events
    if len(training_history) > 50:
        training_history.pop(0)

# ===================== MAIN ROUTES =====================

@app.route('/')
def home():
    """Main dashboard page"""
    available_models = list(model_store.keys())
    recent_activity = training_history[-5:] if training_history else []
    
    return render_template('index.html', 
                         title='AI Traffic Analysis Dashboard',
                         available_models=available_models,
                         model_names=MODEL_NAMES,
                         recent_activity=recent_activity)

@app.route('/performance')
def performance():
    """Performance metrics and evaluation page"""
    # Get the latest/current model's performance data
    metrics = None
    confusion_matrix = None
    current_model = None
    last_training = None
    
    if current_model_name and current_model_name in performance_metrics:
        perf_data = performance_metrics[current_model_name]
        metrics_data = perf_data['metrics']
        
        # Format metrics for template (convert to percentages and round)
        metrics = {
            'accuracy': round(metrics_data.get('accuracy', 0) * 100, 2),
            'precision': round(metrics_data.get('precision', 0) * 100, 2),
            'recall': round(metrics_data.get('recall', 0) * 100, 2),
            'f1_score': round(metrics_data.get('f1_score', 0) * 100, 2)
        }
        
        # Get confusion matrix
        confusion_matrix = metrics_data.get('confusion_matrix')
        current_model = MODEL_NAMES.get(current_model_name, current_model_name)
        last_training = perf_data.get('timestamp')
    
    return render_template('performance.html', 
                         title='Model Performance & Metrics',
                         metrics=metrics,
                         confusion_matrix=confusion_matrix,
                         current_model=current_model,
                         last_training=last_training,
                         performance_data=performance_metrics,
                         training_history=training_history)

@app.route('/predict-page')
def predict_page():
    """Prediction interface page"""
    available_models = list(model_store.keys())
    if not available_models:
        flash("⚠️ No trained models available. Please train a model first.", "warning")
        return redirect(url_for('home'))
    
    return render_template('predict_page.html', 
                         title='Make Predictions',
                         available_models=available_models,
                         model_names=MODEL_NAMES)

@app.route('/results')
def results():
    """Display prediction results"""
    global predicted_csv_path

    if not predicted_csv_path or not os.path.exists(predicted_csv_path):
        flash("⚠️ No predictions available. Please make a prediction first.", "warning")
        return redirect(url_for('predict_page'))

    try:
        df = pd.read_csv(predicted_csv_path)
        columns = df.columns.tolist()
        rows = df.values.tolist()
        
        # Calculate predictions counts
        normal_count = 0
        attack_count = 0
        
        if 'prediction' in df.columns:
            # Count predictions based on different possible values
            prediction_counts = df['prediction'].value_counts()
            
            # Handle different formats of predictions (0/1, normal/attack, etc.)
            for value, count in prediction_counts.items():
                # Convert to string for comparison
                str_value = str(value).lower().strip()
                
                # Check if it's normal traffic
                if str_value in ['0', 'normal', 'benign', 'legitimate']:
                    normal_count += count
                # Check if it's attack traffic  
                elif str_value in ['1', 'attack', 'malicious', 'anomaly']:
                    attack_count += count
                else:
                    # If unknown format, try to determine by value
                    try:
                        # If it's numeric, assume 0=normal, 1=attack
                        numeric_value = float(value)
                        if numeric_value == 0:
                            normal_count += count
                        elif numeric_value == 1:
                            attack_count += count
                    except:
                        # If can't determine, count as normal by default
                        normal_count += count
        
        # Calculate risk level based on attack percentage
        total_predictions = len(df)
        if total_predictions > 0:
            attack_percentage = (attack_count / total_predictions) * 100
            if attack_percentage >= 80:
                risk_level = "Critical"
            elif attack_percentage >= 60:
                risk_level = "High"
            elif attack_percentage >= 40:
                risk_level = "Medium"
            elif attack_percentage >= 20:
                risk_level = "Low"
            else:
                risk_level = "Minimal"
        else:
            risk_level = "Unknown"
        
        # Calculate basic statistics
        stats = {
            'total_predictions': total_predictions,
            'normal_count': normal_count,
            'attack_count': attack_count,
            'risk_level': risk_level,
            'unique_classes': df['prediction'].nunique() if 'prediction' in df.columns else 0,
            'features_used': len(columns) - 1 if 'prediction' in columns else len(columns)
        }
        
        return render_template('results.html', 
                             title='Prediction Results',
                             columns=columns,
                             rows=rows,
                             stats=stats,
                             normal_count=normal_count,
                             attack_count=attack_count,
                             risk_level=risk_level,
                             filename=os.path.basename(predicted_csv_path))
    
    except Exception as e:
        flash(f"❌ Error loading results: {str(e)}", "danger")
        return redirect(url_for('predict_page'))
# ===================== TRAINING ROUTES =====================

@app.route('/train', methods=['POST'])
def train():
    """Handle model training"""
    global current_model_name
    
    try:
        # Validate form data
        if 'train_csv' not in request.files or 'test_csv' not in request.files:
            flash("❌ Please upload both training and testing CSV files.", "danger")
            return redirect(url_for('home'))
        
        train_file = request.files['train_csv']
        test_file = request.files['test_csv']
        model_choice = request.form.get('model', 'rf')
        
        # Validate files
        if train_file.filename == '' or test_file.filename == '':
            flash("❌ Please select both training and testing files.", "danger")
            return redirect(url_for('home'))
        
        if not (allowed_file(train_file.filename) and allowed_file(test_file.filename)):
            flash("❌ Please upload CSV files only.", "danger")
            return redirect(url_for('home'))
        
        # Generate unique filenames
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        train_filename = f"train_{model_choice}_{timestamp}_{uuid.uuid4().hex[:8]}.csv"
        test_filename = f"test_{model_choice}_{timestamp}_{uuid.uuid4().hex[:8]}.csv"
        
        train_path = os.path.join(UPLOAD_FOLDER, train_filename)
        test_path = os.path.join(UPLOAD_FOLDER, test_filename)
        
        # Save uploaded files
        train_file.save(train_path)
        test_file.save(test_path)
        
        log_training_event(model_choice, 'started', {'train_file': train_filename, 'test_file': test_filename})
        
        # Load datasets
        train_df = load_dataset(train_path)
        test_df = load_dataset(test_path)
        
        # Validate dataset structure
        if 'label' not in train_df.columns or 'label' not in test_df.columns:
            flash("❌ CSV files must contain a 'label' column for training.", "danger")
            return redirect(url_for('home'))
        
        # Preprocess data
        X_train, y_train, scaler, le = preprocess_training_data(train_df)
        X_test, y_test = preprocess_testing_data(test_df, scaler, le)
        
        # Train model
        model = train_model(X_train, y_train, model_choice)
        
        # Store model and preprocessing tools
        model_store[model_choice] = model
        scaler_store[model_choice] = scaler
        label_encoder_store[model_choice] = le
        current_model_name = model_choice  # Set current model
        
        # Evaluate model and capture metrics
        print(f"🔍 Evaluating {MODEL_NAMES[model_choice]} model...")
        metrics = evaluate_model_with_return(model, X_test, y_test)
        print(f"📊 Metrics calculated: {metrics}")
        
        save_performance_metrics(model_choice, metrics)
        
        log_training_event(model_choice, 'completed', {
            'accuracy': f"{metrics.get('accuracy', 0):.4f}",
            'samples_trained': len(X_train),
            'samples_tested': len(X_test)
        })
        
        # Clean up uploaded files (optional)
        try:
            os.remove(train_path)
            os.remove(test_path)
        except:
            pass  # Files might be in use, ignore cleanup errors
        
        flash(f"✅ {MODEL_NAMES[model_choice]} model trained successfully! "
              f"Accuracy: {metrics.get('accuracy', 0)*100:.2f}%", "success")
        
        return redirect(url_for('performance'))
    
    except Exception as e:
        print(f"❌ Training error: {str(e)}")  # Debug print
        log_training_event(model_choice if 'model_choice' in locals() else 'unknown', 'failed', {'error': str(e)})
        flash(f"❌ Training failed: {str(e)}", "danger")
        return redirect(url_for('home'))

def evaluate_model_with_return(model, X_test, y_test):
    """Enhanced evaluation function that returns metrics"""
    from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                               precision_score, recall_score, f1_score, roc_auc_score)
    
    print("🔄 Making predictions for evaluation...")
    y_pred = model.predict(X_test)
    
    print("📈 Calculating metrics...")
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted')
    }
    
    # Add ROC AUC if model supports probability prediction
    try:
        if hasattr(model, 'predict_proba') and len(set(y_test)) == 2:  # Binary classification
            y_prob = model.predict_proba(X_test)[:, 1]
            metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
    except Exception as e:
        print(f"⚠️ Could not calculate ROC AUC: {e}")
    
    # Add confusion matrix and classification report
    print("📊 Calculating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    print("📋 Generating classification report...")
    try:
        metrics['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
    except Exception as e:
        print(f"⚠️ Could not generate classification report: {e}")
    
    print(f"✅ Evaluation complete. Accuracy: {metrics['accuracy']:.4f}")
    return metrics

# ===================== PREDICTION ROUTES =====================

@app.route('/predict', methods=['POST'])
def predict():
    """Handle batch prediction"""
    global predicted_csv_path
    
    try:
        # Validate form data
        if 'predict_csv' not in request.files:
            flash("❌ Please upload a CSV file for prediction.", "danger")
            return redirect(url_for('predict_page'))
        
        csv_file = request.files['predict_csv']
        model_choice = request.form.get('model')
        
        if csv_file.filename == '':
            flash("❌ Please select a file for prediction.", "danger")
            return redirect(url_for('predict_page'))
        
        if not allowed_file(csv_file.filename):
            flash("❌ Please upload a CSV file only.", "danger")
            return redirect(url_for('predict_page'))
        
        # Check if model exists
        if model_choice not in model_store:
            flash(f"⚠️ {MODEL_NAMES.get(model_choice, model_choice)} model not found. Please train the model first.", "warning")
            return redirect(url_for('home'))
        
        # Save uploaded file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        predict_filename = f"predict_input_{model_choice}_{timestamp}_{uuid.uuid4().hex[:8]}.csv"
        predict_path = os.path.join(UPLOAD_FOLDER, predict_filename)
        csv_file.save(predict_path)
        
        # Get model and preprocessing tools
        model = model_store[model_choice]
        scaler = scaler_store[model_choice]
        le = label_encoder_store[model_choice]
        
        # Load and preprocess prediction data
        df = pd.read_csv(predict_path)
        original_df = df.copy()  # Keep original for reference
        
        # Clean data
        df_clean = df.dropna().drop_duplicates()
        
        if len(df_clean) == 0:
            flash("❌ No valid data found after cleaning. Please check your CSV file.", "danger")
            return redirect(url_for('predict_page'))
        
        # Encode categorical columns
        for col in df_clean.select_dtypes(include='object'):
            try:
                df_clean[col] = le.transform(df_clean[col])
            except ValueError as e:
                # Handle unseen categories
                df_clean[col] = df_clean[col].map(lambda x: 0 if x not in le.classes_ else le.transform([x])[0])
        
        # Scale features
        X = scaler.transform(df_clean)
        
        # Make predictions
        predictions = model.predict(X)
        prediction_proba = None
        
        if hasattr(model, 'predict_proba'):
            try:
                prediction_proba = model.predict_proba(X)
                df_clean['prediction_confidence'] = prediction_proba.max(axis=1)
            except:
                pass
        
        df_clean['prediction'] = predictions
        
        # Save results
        result_filename = f"predictions_{model_choice}_{timestamp}_{uuid.uuid4().hex[:8]}.csv"
        predicted_csv_path = os.path.join(UPLOAD_FOLDER, result_filename)
        df_clean.to_csv(predicted_csv_path, index=False)
        
        # Clean up input file
        try:
            os.remove(predict_path)
        except:
            pass
        
        flash(f"✅ Predictions completed using {MODEL_NAMES[model_choice]}! "
              f"Processed {len(df_clean)} samples.", "success")
        
        return redirect(url_for('results'))
    
    except Exception as e:
        flash(f"❌ Prediction failed: {str(e)}", "danger")
        return redirect(url_for('predict_page'))

# ===================== API ROUTES =====================

@app.route('/api/models')
def api_models():
    """API endpoint to get available models"""
    return jsonify({
        'available_models': list(model_store.keys()),
        'model_names': MODEL_NAMES,
        'total_models': len(model_store)
    })

@app.route('/api/performance/<model_name>')
def api_performance(model_name):
    """API endpoint to get model performance"""
    if model_name in performance_metrics:
        return jsonify(performance_metrics[model_name])
    else:
        return jsonify({'error': 'Model not found'}), 404

@app.route('/download/results')
def download_results():
    """Download prediction results"""
    global predicted_csv_path
    
    if predicted_csv_path and os.path.exists(predicted_csv_path):
        return send_file(predicted_csv_path, 
                        as_attachment=True,
                        download_name=f"ai_traffic_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    else:
        flash("❌ No results file found.", "danger")
        return redirect(url_for('results'))

# ===================== ERROR HANDLERS =====================

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

@app.errorhandler(413)
def too_large(error):
    flash("❌ File too large. Please upload a smaller CSV file.", "danger")
    return redirect(url_for('home'))

# ===================== UTILITY ROUTES =====================

@app.route('/clear-models')
def clear_models():
    """Clear all trained models and data"""
    global model_store, scaler_store, label_encoder_store, performance_metrics, training_history, current_model_name
    
    model_store.clear()
    scaler_store.clear()
    label_encoder_store.clear()
    performance_metrics.clear()
    training_history.clear()
    current_model_name = None
    
    flash("🗑️ All models and data cleared successfully!", "info")
    return redirect(url_for('home'))

@app.route('/model-info/<model_name>')
def model_info(model_name):
    """Get detailed information about a specific model"""
    if model_name not in model_store:
        flash(f"❌ Model {model_name} not found.", "danger")
        return redirect(url_for('home'))
    
    model_data = {
        'name': MODEL_NAMES.get(model_name, model_name),
        'type': model_name,
        'trained': True,
        'performance': performance_metrics.get(model_name, {}),
        'features': getattr(model_store[model_name], 'n_features_in_', 'Unknown')
    }
    
    return render_template('model_info.html', 
                         title=f'{MODEL_NAMES.get(model_name, model_name)} Details',
                         model_data=model_data)

# ===================== CONFIGURATION =====================

# File upload limits
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Session configuration
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour

# ===================== TEMPLATE FILTERS =====================

@app.template_filter('datetime')
def datetime_filter(timestamp):
    """Format datetime for templates"""
    if isinstance(timestamp, str):
        return timestamp
    return timestamp.strftime('%Y-%m-%d %H:%M:%S')

@app.template_filter('percentage')
def percentage_filter(value):
    """Convert decimal to percentage"""
    try:
        return f"{float(value) * 100:.2f}%"
    except:
        return "N/A"

@app.template_filter('round_decimal')
def round_decimal_filter(value, decimals=4):
    """Round decimal values"""
    try:
        return round(float(value), decimals)
    except:
        return value

# ===================== CONTEXT PROCESSORS =====================

@app.context_processor
def inject_global_vars():
    """Inject global variables into all templates"""
    return {
        'current_year': datetime.now().year,
        'app_name': 'AI Traffic Analysis System',
        'version': '2.0.0',
        'total_models': len(model_store),
        'available_model_types': MODEL_NAMES
    }

# ===================== BEFORE REQUEST HANDLERS =====================

@app.before_request
def before_request():
    """Execute before each request"""
    # You can add logging, authentication, etc. here
    pass

# ===================== MAIN EXECUTION =====================

if __name__ == '__main__':
    print("🚀 Starting AI Traffic Analysis System...")
    print("📊 Available Models:", list(MODEL_NAMES.keys()))
    print("🌐 Server starting on http://localhost:5000")
    print("=" * 50)
    
    # Create necessary directories on startup
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)