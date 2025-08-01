# Flask ML API on Google Cloud Run 🚀

A lightweight Flask-based machine learning API deployed serverlessly on **Google Cloud Run**, with automated CI/CD using **GitHub Actions**.

---

## 🔧 Tech Stack

- **Flask** – lightweight Python web framework
- **scikit-learn** – example ML model
- **Docker** – containerized deployment
- **Google Cloud Run** – serverless hosting
- **Artifact Registry** – container image storage
- **GitHub Actions** – CI/CD for automation

---

## 📁 Project Structure

.
├── app/
│ └── main.py # Flask app with ML model
├── requirements.txt # Python dependencies
├── Dockerfile # Container definition
├── .github/
│ └── workflows/
│ └── deploy.yml # CI/CD pipeline


---

## 🚀 Deploying to Cloud Run

This project auto-deploys on every `main` push using GitHub Actions. 

### CI/CD Steps:
1. Build Docker image
2. Push to Artifact Registry
3. Deploy to Cloud Run

---

## 🧪 Sample Usage

```bash
curl -X POST https://<your-cloud-run-url>/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [1.2, 3.4, 5.6, 4]}'
```

🔐 Secrets Setup (GitHub)
In your GitHub repo, set the following secret:
| Name         | Value                       |
| ------------ | --------------------------- |
| `GCP_SA_KEY` | Contents of your `key.json` |


📦 Docker Build (Local)
docker build -t flask-ml-api .
docker run -p 8080:8080 flask-ml-api

🧠 Author
Levan Gafrindashvili
Senior Data Scientist

🏁 License
MIT License

