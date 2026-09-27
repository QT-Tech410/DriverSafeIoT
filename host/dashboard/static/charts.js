/**
 * Realtime Charts Component (DASH-02).
 * Quản lý 2 đồ thị chuỗi thời gian thực Chart.js: Fatigue Trend và Cabin & Alcohol Trend.
 */

class DashboardCharts {
  constructor(fatigueCanvasId, cabinCanvasId) {
    this.maxPoints = 300; // Lưu 300 giây (5 phút gần nhất) theo spec DASH-04

    this.fatigueChart = this.initFatigueChart(fatigueCanvasId);
    this.cabinChart = this.initCabinChart(cabinCanvasId);
  }

  initFatigueChart(canvasId) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;

    return new Chart(ctx, {
      type: "line",
      data: {
        labels: [],
        datasets: [
          {
            label: "PERCLOS (%)",
            data: [],
            borderColor: "#f59e0b",
            backgroundColor: "rgba(245, 158, 11, 0.1)",
            borderWidth: 2,
            tension: 0.3,
            yAxisID: "y1",
            fill: true,
          },
          {
            label: "Risk Score (0-100)",
            data: [],
            borderColor: "#ef4444",
            borderWidth: 2,
            tension: 0.3,
            yAxisID: "y1",
          },
          {
            label: "EAR (Mắt)",
            data: [],
            borderColor: "#3b82f6",
            borderWidth: 1.5,
            tension: 0.3,
            yAxisID: "y2",
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {
          legend: { labels: { color: "#9ca3af", font: { size: 11 } } },
        },
        scales: {
          x: {
            grid: { color: "rgba(255, 255, 255, 0.05)" },
            ticks: { color: "#6b7280", font: { size: 10 }, maxTicksLimit: 8 },
          },
          y1: {
            type: "linear",
            position: "left",
            min: 0,
            max: 100,
            grid: { color: "rgba(255, 255, 255, 0.05)" },
            ticks: { color: "#f59e0b", font: { size: 10 } },
          },
          y2: {
            type: "linear",
            position: "right",
            min: 0,
            max: 0.45,
            grid: { drawOnChartArea: false },
            ticks: { color: "#3b82f6", font: { size: 10 } },
          },
        },
      },
    });
  }

  initCabinChart(canvasId) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;

    return new Chart(ctx, {
      type: "line",
      data: {
        labels: [],
        datasets: [
          {
            label: "Nhiệt độ Cabin (°C)",
            data: [],
            borderColor: "#10b981",
            backgroundColor: "rgba(16, 185, 129, 0.1)",
            borderWidth: 2,
            tension: 0.3,
            yAxisID: "y1",
            fill: true,
          },
          {
            label: "Cồn BrAC (g/L)",
            data: [],
            borderColor: "#ef4444",
            borderWidth: 2,
            tension: 0.3,
            yAxisID: "y2",
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {
          legend: { labels: { color: "#9ca3af", font: { size: 11 } } },
        },
        scales: {
          x: {
            grid: { color: "rgba(255, 255, 255, 0.05)" },
            ticks: { color: "#6b7280", font: { size: 10 }, maxTicksLimit: 8 },
          },
          y1: {
            type: "linear",
            position: "left",
            min: 15,
            max: 45,
            grid: { color: "rgba(255, 255, 255, 0.05)" },
            ticks: { color: "#10b981", font: { size: 10 } },
          },
          y2: {
            type: "linear",
            position: "right",
            min: 0.0,
            max: 0.5,
            grid: { drawOnChartArea: false },
            ticks: { color: "#ef4444", font: { size: 10 } },
          },
        },
      },
    });
  }

  pushPoint(t) {
    const timeLabel = new Date(t.ts).toLocaleTimeString("vi-VN", {
      hour12: false,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });

    // 1. Cập nhật Fatigue chart (PERCLOS, Risk Score, EAR)
    if (this.fatigueChart) {
      const fc = this.fatigueChart;
      fc.data.labels.push(timeLabel);
      fc.data.datasets[0].data.push((t.perclos_60s * 100).toFixed(1));
      fc.data.datasets[1].data.push(Number(t.risk || 0).toFixed(1));
      fc.data.datasets[2].data.push(t.ear.toFixed(3));

      if (fc.data.labels.length > this.maxPoints) {
        fc.data.labels.shift();
        fc.data.datasets[0].data.shift();
        fc.data.datasets[1].data.shift();
        fc.data.datasets[2].data.shift();
      }
      fc.update();
    }

    // 2. Cập nhật Cabin chart
    if (this.cabinChart) {
      const cc = this.cabinChart;
      cc.data.labels.push(timeLabel);
      cc.data.datasets[0].data.push(t.temp_c.toFixed(1));
      cc.data.datasets[1].data.push(t.alcohol_g_l.toFixed(3));

      if (cc.data.labels.length > this.maxPoints) {
        cc.data.labels.shift();
        cc.data.datasets[0].data.shift();
        cc.data.datasets[1].data.shift();
      }
      cc.update();
    }
  }

  loadHistory(records) {
    if (!Array.isArray(records)) return;
    for (const r of records.slice(-this.maxPoints)) {
      this.pushPoint(r);
    }
  }
}

window.DashboardCharts = DashboardCharts;
