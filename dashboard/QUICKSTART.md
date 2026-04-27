# Aegis Health Dashboard - Quick Start Guide

## Prerequisites

- Python 3.10+ with Aegis backend installed
- Node.js 16+ and npm
- Aegis health database with some data

## Step 1: Start the Backend Server

From the project root directory:

```bash
python start_dashboard_server.py
```

You should see:
```
INFO:     Starting Aegis Health Dashboard Server...
INFO:     API will be available at: http://127.0.0.1:8000
INFO:     WebSocket endpoint at: ws://127.0.0.1:8000/ws/health/live
```

Keep this terminal open.

## Step 2: Install Frontend Dependencies

Open a new terminal and navigate to the dashboard directory:

```bash
cd dashboard
npm install
```

This will install all required dependencies (React, Chart.js, etc.).

## Step 3: Start the Frontend

```bash
npm start
```

The dashboard will automatically open in your browser at `http://localhost:3000`.

## Step 4: Login

Use the default credentials:
- **Username**: `admin`
- **Password**: `admin`

## What You'll See

### Overview Tab
- Health statistics summary
- Mood trend chart
- Sleep duration chart
- Energy level chart
- Emotion distribution pie chart

### Trends Tab
- Detailed trend charts for mood, sleep, energy
- Vital signs chart (heart rate, SpO2, temperature)
- Interactive zoom and pan (scroll to zoom, drag to pan)

### Correlations Tab
- Sleep vs Mood scatter plot with correlation coefficient
- Energy vs Sleep correlation analysis
- Statistical insights

### Calendar Tab
- Visual calendar with daily health indicators
- Color-coded days (low mood, insufficient sleep)
- Selected date details

### Alerts Tab
- Proactive health alerts
- Filter by status (all, unacknowledged, acknowledged)
- Acknowledge alerts with one click

## Features to Try

### Interactive Charts
- **Zoom**: Scroll on any trend chart to zoom in/out
- **Pan**: Click and drag to pan left/right
- **Reset**: Click "Reset Zoom" button to restore original view
- **Hover**: Hover over data points for detailed information

### Date Range Selection
- Use the dropdown at the top to switch between 7, 30, and 90-day views
- All charts update automatically

### Export Data
- **PNG Export**: Click the 📷 button on any chart to save as image
- **CSV Export**: Click "📥 Export CSV" in the header to download all data

### Accessibility
- **High Contrast**: Click "🌙 High Contrast" for better visibility
- **Large Text**: Click "🔍 Large Text" for increased text size
- **Keyboard Navigation**: Use Tab to navigate, Enter to activate
- **Screen Reader**: Full ARIA support for screen readers

### Live Updates
- Dashboard automatically updates when new health data is recorded
- WebSocket connection shows real-time changes
- No page refresh needed

## Troubleshooting

### Backend Not Starting
- Check that port 8000 is not in use
- Verify Python dependencies are installed: `pip install -r requirements.txt`
- Check that the health database exists: `data/db/aegis_health.db`

### Frontend Not Starting
- Delete `node_modules` and run `npm install` again
- Check that port 3000 is not in use
- Clear npm cache: `npm cache clean --force`

### CORS Errors
- Ensure backend is running on `http://127.0.0.1:8000`
- Check that frontend is on `http://localhost:3000`
- Verify CORS middleware is configured in `core/dashboard_api.py`

### WebSocket Not Connecting
- Check browser console for errors
- Verify backend WebSocket endpoint is accessible
- Try refreshing the page

### Charts Not Rendering
- Check browser console for errors
- Verify Chart.js dependencies are installed
- Try clearing browser cache

### No Data Showing
- Ensure you have health data in the database
- Check that the API endpoints are returning data
- Visit `http://127.0.0.1:8000/docs` to test API directly

## API Documentation

Visit `http://127.0.0.1:8000/docs` when the backend is running to see:
- All available endpoints
- Request/response schemas
- Interactive API testing

## Development Mode

The frontend runs in development mode with:
- Hot reload (changes appear immediately)
- Detailed error messages
- React DevTools support

## Production Build

To create a production build:

```bash
cd dashboard
npm run build
```

This creates an optimized build in the `build/` directory.

To serve the production build:

```bash
npm install -g serve
serve -s build -l 3000
```

## Next Steps

- Explore all tabs and features
- Try the interactive chart controls
- Acknowledge some alerts
- Export data as PNG and CSV
- Test accessibility features
- Check the correlation analysis

## Support

For issues or questions:
- Check the main README: `dashboard/README.md`
- Review the implementation summary: `task_implementations/EPIC_14_COMPLETE_IMPLEMENTATION_SUMMARY.md`
- Check API docs: `http://127.0.0.1:8000/docs`

## Keyboard Shortcuts

- **Tab**: Navigate between elements
- **Enter/Space**: Activate buttons
- **Escape**: Close modals (if any)
- **Arrow Keys**: Navigate calendar

Enjoy your Aegis Health Dashboard! 🎉
