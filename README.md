# Endpoint Project

## Overview
This project is a Flask application designed to handle file uploads, parsing, and data processing. It is deployed on Render.

## Project Structure
- `app.py`: The main Flask application file.
- `requirements.txt`: Lists all the dependencies required for the project.
- `Procfile`: Specifies the command to run the application on Render.
- `render.yaml`: Configuration file for Render deployment.
- `db.py`: Database interaction module.
- `wsgi.py`: WSGI entry point for the application.
- `static/`: Directory for static files (CSS, JavaScript, etc.).
- `templates/`: Directory for HTML templates.

## Dependencies
- Flask
- Werkzeug
- Sentry SDK
- Python-dotenv
- OpenAI
- Neo4j
- EZDXF
- PyPDF2
- Gunicorn

## Deployment
The application is deployed on Render. Ensure the root directory is set to `src` in the Render dashboard.

## Running Locally
1. Clone the repository.
2. Navigate to the project directory.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   python app.py
   ```

## License
This project is licensed under the MIT License. 