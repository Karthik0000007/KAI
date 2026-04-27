import React, { useRef } from 'react';
import { Scatter } from 'react-chartjs-2';
import { Chart as ChartJS, LinearScale, PointElement, Tooltip, Legend } from 'chart.js';
import html2canvas from 'html2canvas';

ChartJS.register(LinearScale, PointElement, Tooltip, Legend);

function CorrelationChart({ xData, yData, xLabel, yLabel }) {
  const chartRef = useRef(null);

  if (!xData || !yData || !xData.data || !yData.data) {
    return <div className="no-data">Insufficient data for correlation analysis</div>;
  }

  // Match data points by date
  const points = [];
  xData.data.forEach(xPoint => {
    const yPoint = yData.data.find(y => y.date === xPoint.date);
    if (yPoint) {
      points.push({
        x: xPoint.value,
        y: yPoint.value
      });
    }
  });

  if (points.length === 0) {
    return <div className="no-data">No matching data points for correlation</div>;
  }

  // Calculate correlation coefficient
  const n = points.length;
  const sumX = points.reduce((sum, p) => sum + p.x, 0);
  const sumY = points.reduce((sum, p) => sum + p.y, 0);
  const sumXY = points.reduce((sum, p) => sum + p.x * p.y, 0);
  const sumX2 = points.reduce((sum, p) => sum + p.x * p.x, 0);
  const sumY2 = points.reduce((sum, p) => sum + p.y * p.y, 0);

  const correlation = (n * sumXY - sumX * sumY) / 
    Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));

  const chartData = {
    datasets: [
      {
        label: `${xLabel} vs ${yLabel}`,
        data: points,
        backgroundColor: '#3498db',
        pointRadius: 6,
        pointHoverRadius: 8,
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
        callbacks: {
          label: function(context) {
            return `${xLabel}: ${context.parsed.x.toFixed(1)}, ${yLabel}: ${context.parsed.y.toFixed(1)}`;
          }
        }
      }
    },
    scales: {
      x: {
        title: {
          display: true,
          text: xLabel
        },
        grid: {
          color: '#f0f0f0'
        }
      },
      y: {
        title: {
          display: true,
          text: yLabel
        },
        grid: {
          color: '#f0f0f0'
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
      a.download = `correlation-${xLabel.toLowerCase()}-${yLabel.toLowerCase()}.png`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
  };

  const getCorrelationStrength = (r) => {
    const abs = Math.abs(r);
    if (abs >= 0.7) return 'Strong';
    if (abs >= 0.4) return 'Moderate';
    if (abs >= 0.2) return 'Weak';
    return 'Very Weak';
  };

  const getCorrelationDirection = (r) => {
    return r > 0 ? 'Positive' : 'Negative';
  };

  return (
    <div className="chart-container" ref={chartRef}>
      <div className="chart-controls">
        <button onClick={handleExportPNG} className="btn-small">
          📷 Export PNG
        </button>
      </div>
      <div style={{ height: '300px' }}>
        <Scatter data={chartData} options={options} />
      </div>
      <div className="correlation-stats">
        <p>
          <strong>Correlation:</strong> {correlation.toFixed(3)} 
          ({getCorrelationStrength(correlation)} {getCorrelationDirection(correlation)})
        </p>
        <p className="correlation-hint">
          {Math.abs(correlation) >= 0.4 
            ? `There is a ${getCorrelationStrength(correlation).toLowerCase()} ${getCorrelationDirection(correlation).toLowerCase()} relationship between ${xLabel.toLowerCase()} and ${yLabel.toLowerCase()}.`
            : `There is little to no correlation between ${xLabel.toLowerCase()} and ${yLabel.toLowerCase()}.`
          }
        </p>
      </div>
    </div>
  );
}

export default CorrelationChart;
