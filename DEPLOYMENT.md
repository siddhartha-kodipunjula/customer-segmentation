# Deployment Guide - Streamlit Cloud

Your Customer Segmentation Dashboard is ready to be deployed to **Streamlit Cloud**! Follow these steps to get your live link:

## Quick Deployment Steps

### 1. **Go to Streamlit Cloud**
   - Visit: https://streamlit.io/cloud
   - Click "Sign Up" or "Get Started"
   - Click "GitHub sign in" and authorize Streamlit

### 2. **Connect Your Repository**
   - Click "New app"
   - Repository: Select **siddhartha-kodipunjula/customer-segmentation**
   - Branch: Select **master**
   - Main file path: **dashboard/app.py**

### 3. **Deploy**
   - Click "Deploy"
   - Streamlit will build and deploy your app (takes 1-3 minutes)
   - Your live link will appear on the screen

## Your Repository
- **GitHub URL**: https://github.com/siddhartha-kodipunjula/customer-segmentation
- **Main App File**: dashboard/app.py

## Expected Live URL Format
After deployment, your app will be available at:
```
https://customer-segmentation-<random-id>.streamlit.app
```

## Features Available on Streamlit Cloud
✅ Free hosting  
✅ Automatic updates when you push to GitHub  
✅ SSL certificate included  
✅ Shareable public link  
✅ Support for all Python libraries (pandas, scikit-learn, plotly, etc.)  

## Troubleshooting
If deployment fails:
1. Ensure all dependencies in `requirements.txt` are correct
2. Check that `dashboard/app.py` exists and is the entry point
3. Verify data files are in the correct path (`data/mall_customers.csv`)
4. Check Streamlit Cloud logs for errors

## Share Your Link
Once deployed, copy the Streamlit Cloud link from your app URL and add it to your GitHub repository:
- Add to **README.md**: `🚀 [Live Dashboard](your-app-url)`
- Add to **GitHub About**: Paste the URL in the repository description

---

**Get started now**: https://streamlit.io/cloud
