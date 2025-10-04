# 🚀 Streamlit Cloud Deployment Guide

## Quick Deploy to Streamlit Cloud

### Prerequisites
- GitHub account
- Streamlit Cloud account ([share.streamlit.io](https://share.streamlit.io))

### Step 1: Push to GitHub

```bash
# If not already a git repository
git init
git add .
git commit -m "Add Streamlit app for AI DAO Hedge Fund"

# Push to GitHub
git remote add origin https://github.com/YOUR_USERNAME/AI-DAO-Hedge-Fund.git
git branch -M main
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Select your repository: `YOUR_USERNAME/AI-DAO-Hedge-Fund`
4. Set main file path: `streamlit_app/app.py`
5. Click "Deploy"

Your app will be live at: `https://YOUR_USERNAME-ai-dao-hedge-fund.streamlit.app`

### Step 3: Configuration (Optional)

#### Use Minimal Requirements
For faster deployment, rename files:
```bash
cd streamlit_app
mv requirements.txt requirements-full.txt
mv requirements-minimal.txt requirements.txt
```

#### Add Secrets (if needed)
In Streamlit Cloud dashboard:
1. Go to app settings
2. Click "Secrets"
3. Add secrets in TOML format:
```toml
[api]
url = "https://your-api.com"

[contracts]
dao_governance = "0x..."
```

## Local Testing

### Using Full Requirements
```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

### Using Minimal Requirements
```bash
cd streamlit_app
pip install -r requirements-minimal.txt
streamlit run app.py
```

Access at: `http://localhost:8501`

## Troubleshooting

### Port Already in Use
```bash
streamlit run app.py --server.port 8502
```

### Module Import Errors
```bash
# Ensure you're in streamlit_app directory
cd streamlit_app
python -c "import streamlit; import plotly; import pandas; import numpy"
```

### Clear Cache
```bash
streamlit cache clear
```

## Custom Domain (Optional)

After deployment, you can:
1. Go to app settings in Streamlit Cloud
2. Add custom domain (requires DNS configuration)

## Performance Tips

1. **Use caching**: Add `@st.cache_data` to expensive functions
2. **Lazy loading**: Import heavy libraries only when needed
3. **Minimal requirements**: Use requirements-minimal.txt for faster deploys
4. **Optimize data**: Use smaller datasets for demo purposes

## Live Demo URLs

Once deployed, update README with:
```markdown
## 🌐 Live Demo

**Streamlit App**: https://ai-dao-hedge-fund.streamlit.app

Try the interactive dashboard with:
- Real-time portfolio monitoring
- AI agent control interface
- DAO governance voting
- SHAP explainability
- Trading simulations
```

## Security Notes

- Never commit `.streamlit/secrets.toml`
- Use Streamlit Cloud secrets management
- Don't hardcode API keys or private keys
- Use environment variables for sensitive data

## Support

If deployment fails:
1. Check Streamlit Cloud logs
2. Verify requirements.txt syntax
3. Test locally first
4. Check [Streamlit docs](https://docs.streamlit.io)

---

**Ready to deploy?** Follow the 3 steps above and your AI DAO Hedge Fund app will be live in minutes!
