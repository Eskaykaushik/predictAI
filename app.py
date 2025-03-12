from flask import Flask, render_template, request, send_file
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Folder to store uploaded files
UPLOAD_FOLDER = 'uploads'
PLOT_FOLDER = 'static/plots'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv'}
df = None  # Global variable to store uploaded dataframe

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global df  # Update global dataframe

    if 'datafile' not in request.files:
        return "No file part"

    file = request.files['datafile']
    if file.filename == '':
        return "No selected file"

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Read CSV into DataFrame
        df = pd.read_csv(filepath)

        # Extract basic information
        head_data = df.head().values.tolist()
        num_rows, num_columns = df.shape

        # Get statistics
        column_stats = df.describe(include='all').transpose()
        column_stats['null_count'] = df.isnull().sum()
        column_stats['data_type'] = df.dtypes
        column_stats_dict = column_stats.to_dict(orient='index')

        return render_template('select_columns.html',
                               columns=df.columns,
                               head_data=head_data,
                               num_rows=num_rows,
                               num_columns=num_columns,
                               column_stats=column_stats_dict)
    else:
        return "Invalid file type. Please upload a CSV file."

@app.route("/plot", methods=["POST"])
def generate_plot():
    column = request.form["column"]
    plot_type = request.form["plot_type"]

    plt.figure(figsize=(6, 4))

    if plot_type == "histogram":
        sns.histplot(df[column], bins=20, kde=True)
    elif plot_type == "boxplot":
        sns.boxplot(x=df[column])

    plot_path = "static/plot.png"
    plt.savefig(plot_path)
    plt.close()

    return f'<img src="/{plot_path}" alt="Generated Plot">'




if __name__ == '__main__':
    app.run(debug=True)
