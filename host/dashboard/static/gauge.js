/**
 * Risk Gauge Component (DASH-02).
 * Vẽ đồng hồ bán nguyệt đo Risk Score 0–100 với 4 dải màu phân tầng.
 */

class RiskGauge {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext("2d");
    this.currentScore = 0;
    this.targetScore = 0;
    this.band = "SAFE";
    this.animating = false;

    // Màu sắc theo 4 dải quy định tại spec §8
    this.bands = [
      { max: 25, color: "#10b981", name: "SAFE" },       // Xanh
      { max: 50, color: "#f59e0b", name: "WARN" },       // Vàng
      { max: 75, color: "#f97316", name: "ALARM" },      // Cam
      { max: 100, color: "#ef4444", name: "CRITICAL" },  // Đỏ
    ];

    this.draw();
  }

  setScore(score, band) {
    this.targetScore = Math.max(0, Math.min(100, Number(score) || 0));
    this.band = band || this.getBandName(this.targetScore);
    if (!this.animating) {
      this.animate();
    }
  }

  getBandName(val) {
    if (val < 25) return "SAFE";
    if (val < 50) return "WARN";
    if (val <= 75) return "ALARM";
    return "CRITICAL";
  }

  getBandColor(val) {
    for (const b of this.bands) {
      if (val <= b.max) return b.color;
    }
    return "#ef4444";
  }

  animate() {
    this.animating = true;
    const diff = this.targetScore - this.currentScore;
    if (Math.abs(diff) < 0.2) {
      this.currentScore = this.targetScore;
      this.draw();
      this.animating = false;
      return;
    }

    this.currentScore += diff * 0.15;
    this.draw();
    requestAnimationFrame(() => this.animate());
  }

  draw() {
    if (!this.canvas || !this.ctx) return;
    const ctx = this.ctx;
    const width = this.canvas.width;
    const height = this.canvas.height;
    ctx.clearRect(0, 0, width, height);

    const centerX = width / 2;
    const centerY = height - 25;
    const radius = Math.min(centerX, centerY) - 20;
    const startAngle = Math.PI;
    const endAngle = 2 * Math.PI;
    const arcWidth = 22;

    // 1. Vẽ 4 cung màu phân dải
    let prevAngle = startAngle;
    const ranges = [
      { start: 0, end: 25, color: "#10b981" },
      { start: 25, end: 50, color: "#f59e0b" },
      { start: 50, end: 75, color: "#f97316" },
      { start: 75, end: 100, color: "#ef4444" },
    ];

    for (const r of ranges) {
      const segAngle = startAngle + (r.end / 100) * Math.PI;
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius, prevAngle, segAngle, false);
      ctx.lineWidth = arcWidth;
      ctx.strokeStyle = r.color;
      ctx.stroke();
      prevAngle = segAngle;
    }

    // 2. Vẽ kim chỉ thị (Needle)
    const needleAngle = startAngle + (this.currentScore / 100) * Math.PI;
    const needleLength = radius - 10;
    const needleColor = this.getBandColor(this.currentScore);

    ctx.save();
    ctx.translate(centerX, centerY);
    ctx.rotate(needleAngle);

    ctx.beginPath();
    ctx.moveTo(0, -6);
    ctx.lineTo(needleLength, 0);
    ctx.lineTo(0, 6);
    ctx.closePath();
    ctx.fillStyle = needleColor;
    ctx.shadowColor = needleColor;
    ctx.shadowBlur = 10;
    ctx.fill();
    ctx.restore();

    // 3. Trục kim tâm (Hub circle)
    ctx.beginPath();
    ctx.arc(centerX, centerY, 12, 0, 2 * Math.PI);
    ctx.fillStyle = "#1f2937";
    ctx.fill();
    ctx.lineWidth = 3;
    ctx.strokeStyle = needleColor;
    ctx.stroke();

    // 4. In nhãn số Risk Score trung tâm
    ctx.font = "bold 32px system-ui, sans-serif";
    ctx.fillStyle = needleColor;
    ctx.textAlign = "center";
    ctx.fillText(this.currentScore.toFixed(1), centerX, centerY - 45);

    // 5. In tên Band
    ctx.font = "600 14px system-ui, sans-serif";
    ctx.fillStyle = "#9ca3af";
    ctx.fillText(this.band.toUpperCase(), centerX, centerY - 25);
  }
}

// Khởi tạo toàn cục cho frontend
window.RiskGauge = RiskGauge;
