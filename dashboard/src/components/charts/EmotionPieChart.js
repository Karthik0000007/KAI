import React, { useRef } from 'react';
import { Pie } from 'react-chartjs-2';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';
import html2canvas from 'html2canvas';

ChartJS.register(ArcElement, Tooltip, Legend);

function EmotionPieChart({ data }) {
  const chartRef = useRef(null);

  if (!data || !data.emotions || data.emotions.length === 0) {
    return <div className="no-data">No emotion data available</div>;
  }

  const emotionColors = {
    happy: '#f39c12',
    sad: '#3498db',
    angry: '#e74c3c',
    stressed: '#e67e22',
    calm: '#2ecc71',
    anxious: '#9b59b6',
    neutral: '#95a5a6',
    excited: '#f1c40f',
    tired: '#34495e'
  };

  const chartData = {
    labels: data.emotions.map(e => e.emotion.charAt(0).toUpperCase() + e.emotion.slice(1)),
    datasets: [
      {
        data: data.emotions.map(e => e.count),
        backgroundColor: data.emotions.map(e => emotionColors[e.emotion] || '#95a5a6'),
        borderColor: '#fff',
        borderWidth: 2,
      }
    ]
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'right',
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            const label = context.label || '';
            const value = context.parsed || 0;
            const total = context.dataset.data.reduce((a, b) => a + b, 0);
            const percentage = ((value / total) * 100).toFixed(1);
            return `${label}: ${value} (${percentage}%)`;
          }
        }
      }
    }
  };

  const handleExportPNG = async () => {
    if (chartRef.current) {
      const canvas = await html2canvas(chartRef.current);
      const url = canvas.toDataURL('image/png');
      const a = document.createElement('a');
      a.href = url;
      a.download = 'emotion-distribution-chart.png';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
  };

  return (
    <div className="chart-container" ref={chartRef}>
      <div className="chart-controls">
        <button onClick={handleExportPNG} className="btn-small">
          📷 Export PNG
        </button>
      </div>
      <div style={{ height: '300px' }}>
        <Pie data={chartData} options={options} />
      </div>
    </div>
  );
}

export default EmotionPieChart;
