# LLM-evaluation-platform

A production-ready ML application for comparing Large Language Model responses with experiment tracking and analytics. Fully deployed with automated CI/CD pipeline.

## Live Demo

https://huggingface.co/spaces/Kartheek321/LLM-Comparison-Platform

## Overview

End-to-end deployed application featuring dynamic model selection, real-time LLM comparison, MLflow tracking, persistent storage, and PDF export capabilities.

## Key Features

- Compare 3 LLMs simultaneously from 6 available models
- Dynamic model selection via dropdown menus
- Real-time response comparison with metrics
- MLflow experiment tracking
- Persistent CSV storage for cross-user analytics
- Interactive analytics dashboard with Plotly visualizations
- Professional PDF export with clean formatting
- Automated CI/CD deployment from GitHub to HuggingFace Spaces

## Available Models

- Llama-3.1-8B (Fastest)
- Llama-3.3-70B (Best Quality)
- GPT-OSS-120B (Most Capable)
- GPT-OSS-20B (Balanced)
- Qwen3-32B (Multilingual)
- Llama-4-Scout-17B (Latest)

## Tech Stack

**Frontend:** Streamlit, Plotly  
**Backend:** Python 3.11, Groq API  
**ML Ops:** MLflow, Pandas  
**Storage:** CSV (Persistent)  
**Export:** ReportLab (PDF)  
**Deployment:** HuggingFace Spaces, Docker, GitHub CI/CD

## Architecture

GitHub → HuggingFace Spaces (Docker) → Live App → Groq API + MLflow + CSV Storage

## Local Setup

Clone repository:
git clone https://github.com/Kartheek321/llm-evaluation-platform.git
cd llm-evaluation-platform

Install dependencies:
pip install -r requirements.txt

Add API key:
echo "GROQ_API_KEY=your_key" > .env

Run app:
streamlit run app.py

## Application Pages

**1. Compare Models**  
Select 3 models, enter prompt, view side-by-side responses with time and token metrics

**2. Analytics Dashboard**  
Community statistics, response time distribution, token usage charts, detailed results table

**3. Export**  
Generate and download professional PDF reports of current comparison

## Technical Skills Demonstrated

- End-to-end ML application development and deployment
- Production-grade code with error handling
- RESTful API integration (Groq LLM API)
- Automated CI/CD pipeline setup
- Docker containerization
- MLflow experiment tracking
- Persistent data storage design
- Interactive UI development with Streamlit
- PDF generation with complex formatting
- State management and session handling

## Project Structure

```
LLM-evaluation-platform/
├── app.py                  # Main application
├── requirements.txt        # Dependencies
├── packages.txt            # System packages
└── README.md               # Documentation
```

## Deployment

Automated deployment pipeline from GitHub to HuggingFace Spaces with Docker containerization. Environment variables managed securely through HuggingFace settings.

## Connect

**Live App:** https://huggingface.co/spaces/Kartheek321/LLM-Comparison-Platform

Built to demonstrate end-to-end ML engineering from API integration to production deployment with CI/CD automation.
