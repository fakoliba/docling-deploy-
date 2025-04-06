# Document Q&A Application - Deployment Guide

This document provides instructions for deploying the Document Q&A application to various cloud platforms.

## Prerequisites

- Python 3.8 or higher
- A Streamlit Cloud account, Heroku account, or other cloud platform account
- OpenAI API key
- Tavily API key (optional, for internet search functionality)

## API Keys Setup

This application requires the following API keys:

1. **OpenAI API Key**: Required for embeddings and chat completions
2. **Tavily API Key**: Optional, enables internet search functionality

These keys should be configured as environment variables or secrets depending on your deployment platform.

## Deployment Options

### Streamlit Cloud (Recommended)

1. Push this repository to GitHub
2. Log in to [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app, selecting this repository
4. Set the main file path to `chat.py`
5. Configure the following secrets in the Streamlit Cloud dashboard:
   ```toml
   [api_keys]
   OPENAI_API_KEY = "your-openai-api-key"
   TAVILY_API_KEY = "your-tavily-api-key"
   ```
6. Deploy the app!

### Heroku

1. Make sure you have the Heroku CLI installed
2. Log in to Heroku: `heroku login`
3. Create a new Heroku app: `heroku create your-app-name`
4. Set the required buildpacks:
   ```
   heroku buildpacks:add --index 1 heroku/python
   heroku buildpacks:add --index 2 https://github.com/heroku/heroku-buildpack-apt
   ```
5. Configure environment variables:
   ```
   heroku config:set OPENAI_API_KEY=your-openai-api-key
   heroku config:set TAVILY_API_KEY=your-tavily-api-key
   ```
6. Deploy: `git push heroku main`

### Railway

1. Create an account on [Railway](https://railway.app/)
2. Create a new project and select "Deploy from GitHub repo"
3. Connect your GitHub repository
4. Add the required environment variables in the Railway dashboard:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `TAVILY_API_KEY`: Your Tavily API key
5. Deploy the application

## Advanced Configuration

### Custom Port and Host

By default, Streamlit runs on port 8501. To customize this for local development, create a `.streamlit/config.toml` file:

```toml
[server]
port = 8501
headless = true
enableCORS = false
```

### Memory and Performance Settings

For better performance with large documents, you may need to increase memory limits on your deployment platform.

### Database Setup

The application uses LanceDB for document storage. The database is created automatically when the application runs for the first time.

## Troubleshooting

- **Missing system dependencies**: Check that your deployment platform supports installing system packages via the `packages.txt` file
- **API key errors**: Verify that your API keys are correctly set in your environment variables or secrets
- **Memory errors**: Increase the memory allocation for your deployment if you encounter out-of-memory errors

## Need Help?

If you encounter any issues with deployment, please open an issue on the GitHub repository.
