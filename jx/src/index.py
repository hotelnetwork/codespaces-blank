from flask import Flask, render_template
import os

app = Flask(__name__, template_folder='public/templates')


@app.route('/')
def index():
    # Get all HTML files in the templates directory
    files = os.listdir('public/templates')
    html_files = [f for f in files if f.endswith('.html')]

    # Render the index template with the list of HTML files
    return render_template('index.html', html_files=html_files)


if __name__ == "__main__":
    app.run(debug=True)
