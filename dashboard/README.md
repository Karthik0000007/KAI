# Aegis Health Dashboard - React Frontend

A comprehensive health monitoring dashboard for the Aegis Offline Health AI system.

## Features

### ✅ Implemented (Epic 14 Complete)

#### Task 14.3: Complete Charts
- **Mood Trend Chart**: 7/30/90 day views with line chart
- **Sleep Duration Chart**: Track sleep patterns over time
- **Energy Level Chart**: Monitor energy trends
- **Emotion Distribution**: Pie chart showing emotion breakdown
- **Vital Signs Charts**: Heart rate, SpO2, temperature tracking

#### Task 14.4: Interactive Features
- **Zoom & Pan**: Scroll to zoom, drag to pan on all trend charts
- **Date Range Selection**: Switch between 7, 30, and 90-day views
- **Hover Tooltips**: Detailed data on hover for all charts

#### Task 14.5: Correlation Analysis
- **Sleep vs Mood**: Scatter plot with correlation coefficient
- **Energy vs Sleep**: Correlation analysis
- **Statistical Insights**: Correlation strength and direction indicators

#### Task 14.6: Calendar View & Alerts
- **Calendar View**: Visual calendar with daily health summaries
- **Day Highlighting**: Low mood and insufficient sleep days highlighted
- **Alerts History**: Complete proactive alerts with acknowledgment
- **Alert Filtering**: View all, unacknowledged, or acknowledged alerts
- **Conversation Statistics**: Total check-ins and conversations tracked

#### Task 14.7: Export Functionality
- **PNG Export**: Export any chart as PNG image
- **CSV Export**: Export all health data as CSV file

#### Task 14.8: WebSocket Live Updates
- **Real-time Updates**: Dashboard updates automatically when new data arrives
- **Auto-reconnect**: WebSocket reconnects automatically on disconnect

### 🎨 Accessibility Features (Epic 20 Integration)
- **High Contrast Theme**: Toggle for better visibility
- **Large Text Mode**: Increase text size for readability
- **Keyboard Navigation**: Full keyboard support
- **Screen Reader Support**: ARIA labels and semantic HTML
- **Skip Links**: Skip to main content
- **Focus Indicators**: Clear focus outlines

## Installation

```bash
cd dashboard
npm install
```

## Running the Dashboard

### Development Mode
```bash
npm start
```

The dashboard will open at `http://localhost:3000`

### Production Build
```bash
npm run build
```

## Backend Requirements

The dashboard requires the Aegis FastAPI backend to be running:

```bash
# From the project root
python -m uvicorn core.dashboard_api:app --reload --host 127.0.0.1 --port 8000
```

Or use the provided startup script:

```python
# start_dashboard_server.py
from core.dashboard_api import create_app
from core.health_db import HealthDatabase
import uvicorn

db = HealthDatabase()
app = create_app(db)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
```

## Default Credentials

For testing purposes:
- **Username**: `admin`
- **Password**: `admin`

## Architecture

### Components

- **App.js**: Main application component with authentication
- **Login.js**: Login form with session management
- **Dashboard.js**: Main dashboard with tabs and data loading
- **TrendChart.js**: Line charts with zoom/pan support
- **EmotionPieChart.js**: Pie chart for emotion distribution
- **VitalSignsChart.js**: Multi-line chart for vital signs
- **CorrelationChart.js**: Scatter plots with correlation analysis
- **CalendarView.js**: Calendar with daily health summaries
- **AlertsPanel.js**: Proactive alerts with acknowledgment
- **StatisticsPanel.js**: Health statistics overview

### Services

- **api.js**: API service for backend communication
  - REST API calls
  - WebSocket connection management
  - Authentication handling

## Features by Requirement

| Requirement | Feature | Status |
|-------------|---------|--------|
| 13.1 | FastAPI backend with REST endpoints | ✅ |
| 13.2 | Mood trend chart (7/30/90 days) | ✅ |
| 13.3 | Sleep trend chart (7/30/90 days) | ✅ |
| 13.4 | Energy trend chart (7/30/90 days) | ✅ |
| 13.5 | Emotion distribution pie chart | ✅ |
| 13.6 | Medication compliance calendar | ✅ |
| 13.7 | Vital signs line charts | ✅ |
| 13.8 | Calendar view with daily summaries | ✅ |
| 13.9 | Proactive alerts history | ✅ |
| 13.10 | Zoom, pan, hover tooltips | ✅ |
| 13.11 | Export to PNG | ✅ |
| 13.12 | Export to CSV | ✅ |
| 13.13 | Correlation analysis views | ✅ |
| 13.14 | Conversation statistics | ✅ |
| 13.15 | Session-based authentication | ✅ |
| 13.16 | Session timeout (30 minutes) | ✅ |
| 13.17 | WebSocket live updates | ✅ |

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)

## Dependencies

- **react**: ^18.2.0
- **react-dom**: ^18.2.0
- **chart.js**: ^4.4.0 - Chart rendering
- **react-chartjs-2**: ^5.2.0 - React wrapper for Chart.js
- **axios**: ^1.6.0 - HTTP client
- **date-fns**: ^2.30.0 - Date formatting
- **react-calendar**: ^4.6.0 - Calendar component
- **html2canvas**: ^1.4.1 - PNG export
- **chartjs-plugin-zoom**: For zoom/pan functionality

## Development Notes

### Adding New Charts

1. Create a new component in `src/components/charts/`
2. Import Chart.js components as needed
3. Add export PNG functionality
4. Include in Dashboard.js

### Styling

- Uses CSS modules for component-specific styles
- Global styles in `index.css`
- Accessibility themes in `App.css`

### Testing

```bash
npm test
```

## Troubleshooting

### CORS Issues
Ensure the backend has CORS middleware configured for `http://localhost:3000`

### WebSocket Connection Failed
Check that the backend is running and WebSocket endpoint is accessible

### Charts Not Rendering
Verify Chart.js and react-chartjs-2 are properly installed

## Future Enhancements

- Dark mode theme
- Mobile app version
- Offline support with service workers
- More chart types (heatmaps, radar charts)
- Custom date range picker
- Data comparison between time periods
