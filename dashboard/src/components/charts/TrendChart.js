import React, { useRef } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import zoomPlugin from 'chartjs-plugin-zoom';
import html2canvas from 'html2canvas';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  zoomPlugin
);

function TrendChart({ data, label, color }) {
  const chartRef = useRef(null);

  if (!data || !data.data || data.data.length === 0) {
    return <div className="no-data">No data available</div>;
  }

  const chartData = {
    labels: data.data.map(d => new Date(d.date).toLocaleDateString()),
    datasets: [
      {
        label: label,
        data: data.data.map(d => d.value),
        borderColor: color,
        backgroundColor: `${color}33`,
        fill: true,
        tension: 0.4,
        pointRadius: 4,
        pointHoverRadius: 6,
      }
    ]
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: true,
        position: 'top',
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        callbacks: {
          label: function(context) {
            return `${context.dataset.label}: ${context.parsed.y.toFixed(1)}`;
          }
        }
      },
      zoom: {
        zoom: {
          wheel: {
            enabled: true,
          },
          pinch: {
            enabled: true
          },
          mode: 'x',
        },
        pan: {
          enabled: true,
          mode: 'x',
        }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        grid: {
          color: '#f0f0f0'
        }
      },
      x: {
        grid: {
          display: false
        }
      }
    },
    interaction: {
      mode: 'nearest',
      axis: 'x',
      intersect: false
    }
  };

  const handleExportPNG = async () => {
    if (chartRef.current) {
      const canvas = await html2canvas(chartRef.current);
      const url = canvas.toDataURL('image/png');
      const a = document.createElement('a');
      a.href = url;
      a.download = `${label.toLowerCase().replace(' ', '-')}-chart.png`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
  };

  const handleResetZoom = () => {
    if (chartRef.current) {
      chartRef.current.resetZoom();
    }
  };

  return (
    <div className="chart-container" ref={chartRef}>
      <div className="chart-controls">
        <button onClick={handleResetZoom} className="btn-small">
          Reset Zoom
        </button>
        <button onClick={handleExportPNG} className="btn-small">
          📷 Export PNG
        </button>
      </div>
      <div style={{ height: '300px' }}>
        <Line ref={chartRef} data={chartData} options={options} />
      </div>
      <p className="chart-hint">💡 Scroll to zoom, drag to pan</p>
    </div>
  );
}

export default TrendChart;
