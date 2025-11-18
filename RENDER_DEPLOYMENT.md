# Deploying to Render - Quick Start Guide

## Option 1: Automatic Deployment (Recommended)

1. **Push to GitHub:**
   ```bash
   git add render.yaml render-requirements.txt
   git commit -m "Add Render deployment config"
   git push
   ```

2. **Connect to Render:**
   - Go to https://render.com
   - Sign up/Login
   - Click "New +" → "Blueprint"
   - Connect your GitHub repository
   - Render will automatically detect `render.yaml` and set everything up!

3. **Done!** Render will build and deploy automatically.

## Option 2: Manual Setup

1. **Go to** https://render.com → "New +" → "Web Service"

2. **Connect GitHub** repository

3. **Configure:**
   - **Name:** `fake-logo-detection`
   - **Build Command:** 
     ```
     pip install -r render-requirements.txt
     ```
   - **Start Command:**
     ```
     streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
     ```
     **Note:** Additional settings like CORS are configured in `.streamlit/config.toml`

4. **Environment Variables:**
   - `PYTHON_VERSION` = `3.11.0`
   - `SESSION_SECRET` = (auto-generate)

5. **Deploy!**

## Environment Variables (Optional)

For email/SMS alerts, add these later:
- `SENDGRID_API_KEY` - Email alerts
- `TWILIO_ACCOUNT_SID` - SMS
- `TWILIO_AUTH_TOKEN` - SMS  
- `TWILIO_PHONE_NUMBER` - SMS

## Free Tier Info

✅ **512 MB RAM** - Enough for this app
✅ **750 hours/month** - Runs 24/7 for one app
⚠️ **Sleeps after 15 min inactivity** - First request takes ~30-60s to wake

## Monitoring

- **View Logs:** Render Dashboard → Your Service → Logs
- **Live URL:** Render provides `https://fake-logo-detection.onrender.com`
- **Custom Domain:** Free SSL with your own domain

## Troubleshooting

**Build fails?**
- Check Python version is 3.11
- Verify all packages in render-requirements.txt

**App crashes?**
- Check logs in Render dashboard
- Verify start command is correct
- Check memory usage (512MB limit)

**JavaScript/Module Loading Errors?**
- "TypeError: Failed to fetch dynamically imported module"
- **Fix:** Ensure `.streamlit/config.toml` does NOT hardcode serverAddress or serverPort in [browser] section
- Let Streamlit auto-detect the correct URL based on Render's dynamic $PORT
- CORS and XSRF settings should be in [server] section only

**Slow to wake up?**
- Normal on free tier
- Use UptimeRobot to ping every 14 min to keep awake
