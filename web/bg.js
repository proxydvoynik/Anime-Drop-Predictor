// ===== Interactive Multi-Color Particle Constellation Background =====
(function() {
  const canvas = document.getElementById('bg-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  // Config - High Visibility & Vibrant Multi-Color Palette
  const CONFIG = {
    particleCount: 110,
    connectionDistance: 150,
    mouseRadius: 220,
    mouseForce: 0.1,
    baseSpeed: 0.4,
    colors: [
      { r: 0,   g: 242, b: 254 }, // Electric Cyan
      { r: 168, g: 85,  b: 247 }, // Neon Violet
      { r: 244, g: 63,  b: 94  }, // Neon Pink / Rose
      { r: 16,  g: 185, b: 129 }, // Mint / Emerald
      { r: 245, g: 158, b: 11  }, // Radiant Gold
      { r: 59,  g: 130, b: 246 }, // Sky Blue
      { r: 255, g: 94,  b: 58  }, // Bright Coral
    ],
    shootingStarInterval: 3000,
  };

  let W, H;
  let mouse = { x: -9999, y: -9999, active: false };
  let particles = [];
  let shootingStars = [];
  let lastShootingStar = 0;

  function resize() {
    W = canvas.width = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }

  window.addEventListener('resize', resize);
  resize();

  // Mouse tracking
  document.addEventListener('mousemove', (e) => {
    mouse.x = e.clientX;
    mouse.y = e.clientY;
    mouse.active = true;
  });
  document.addEventListener('mouseleave', () => {
    mouse.active = false;
  });

  // Particle class
  class Particle {
    constructor() {
      this.reset();
    }

    reset() {
      this.x = Math.random() * W;
      this.y = Math.random() * H;
      this.vx = (Math.random() - 0.5) * CONFIG.baseSpeed;
      this.vy = (Math.random() - 0.5) * CONFIG.baseSpeed;
      this.radius = Math.random() * 2.5 + 1.5; // Brighter & larger
      this.baseRadius = this.radius;
      this.color = CONFIG.colors[Math.floor(Math.random() * CONFIG.colors.length)];
      this.alpha = Math.random() * 0.4 + 0.55; // High opacity: 0.55 to 0.95
      this.baseAlpha = this.alpha;
      // Pulse animation
      this.pulseSpeed = Math.random() * 0.03 + 0.01;
      this.pulsePhase = Math.random() * Math.PI * 2;
    }

    update(time) {
      // Pulse size and alpha smoothly
      const pulse = Math.sin(time * this.pulseSpeed + this.pulsePhase);
      this.radius = this.baseRadius + pulse * 0.8;
      this.alpha = Math.min(1, Math.max(0.3, this.baseAlpha + pulse * 0.25));

      // Mouse repulsion
      if (mouse.active) {
        const dx = this.x - mouse.x;
        const dy = this.y - mouse.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < CONFIG.mouseRadius) {
          const force = (1 - dist / CONFIG.mouseRadius) * CONFIG.mouseForce;
          this.vx += (dx / dist) * force;
          this.vy += (dy / dist) * force;
        }
      }

      // Smooth damping
      this.vx *= 0.985;
      this.vy *= 0.985;

      // Drift back toward base speed if slowing down
      const speed = Math.sqrt(this.vx * this.vx + this.vy * this.vy);
      if (speed < CONFIG.baseSpeed * 0.4) {
        this.vx += (Math.random() - 0.5) * 0.04;
        this.vy += (Math.random() - 0.5) * 0.04;
      }

      this.x += this.vx;
      this.y += this.vy;

      // Wrap around screen edges
      const pad = 30;
      if (this.x < -pad) this.x = W + pad;
      if (this.x > W + pad) this.x = -pad;
      if (this.y < -pad) this.y = H + pad;
      if (this.y > H + pad) this.y = -pad;
    }

    draw() {
      // Outer intense colorful halo
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.radius * 3.5, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${this.color.r}, ${this.color.g}, ${this.color.b}, ${this.alpha * 0.35})`;
      ctx.fill();

      // Main core particle
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${this.color.r}, ${this.color.g}, ${this.color.b}, ${this.alpha})`;
      ctx.fill();

      // Bright white center highlight for extra pop
      if (this.radius > 2.0) {
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.radius * 0.4, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255, 255, 255, ${this.alpha * 0.9})`;
        ctx.fill();
      }
    }
  }

  // Shooting Star class
  class ShootingStar {
    constructor() {
      this.x = Math.random() * W * 0.7;
      this.y = Math.random() * H * 0.4;
      const angle = Math.PI * 0.18 + Math.random() * 0.25;
      const speed = 7 + Math.random() * 5;
      this.vx = Math.cos(angle) * speed;
      this.vy = Math.sin(angle) * speed;
      this.life = 1;
      this.decay = 0.012 + Math.random() * 0.01;
      this.length = 60 + Math.random() * 70;
      this.color = CONFIG.colors[Math.floor(Math.random() * CONFIG.colors.length)];
    }

    update() {
      this.x += this.vx;
      this.y += this.vy;
      this.life -= this.decay;
    }

    draw() {
      if (this.life <= 0) return;
      const speedNorm = Math.sqrt(this.vx * this.vx + this.vy * this.vy);
      const tailX = this.x - (this.vx / speedNorm) * this.length;
      const tailY = this.y - (this.vy / speedNorm) * this.length;

      const gradient = ctx.createLinearGradient(tailX, tailY, this.x, this.y);
      gradient.addColorStop(0, `rgba(${this.color.r}, ${this.color.g}, ${this.color.b}, 0)`);
      gradient.addColorStop(0.7, `rgba(${this.color.r}, ${this.color.g}, ${this.color.b}, ${this.life * 0.5})`);
      gradient.addColorStop(1, `rgba(255, 255, 255, ${this.life * 0.95})`);

      ctx.beginPath();
      ctx.moveTo(tailX, tailY);
      ctx.lineTo(this.x, this.y);
      ctx.strokeStyle = gradient;
      ctx.lineWidth = 2.2;
      ctx.stroke();

      // Head bright glow
      ctx.beginPath();
      ctx.arc(this.x, this.y, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(255, 255, 255, ${this.life})`;
      ctx.fill();
    }

    get alive() { return this.life > 0; }
  }

  // Init particles
  for (let i = 0; i < CONFIG.particleCount; i++) {
    particles.push(new Particle());
  }

  // Draw vibrant connecting lines between nearby particles
  function drawConnections() {
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < CONFIG.connectionDistance) {
          const alpha = (1 - dist / CONFIG.connectionDistance) * 0.35; // Brighter lines!
          const c1 = particles[i].color;
          const c2 = particles[j].color;
          const mr = Math.round((c1.r + c2.r) / 2);
          const mg = Math.round((c1.g + c2.g) / 2);
          const mb = Math.round((c1.b + c2.b) / 2);

          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = `rgba(${mr}, ${mg}, ${mb}, ${alpha})`;
          ctx.lineWidth = 0.8;
          ctx.stroke();
        }
      }
    }
  }

  // Draw vibrant multi-color interactive mouse glow
  function drawMouseGlow() {
    if (!mouse.active) return;
    const gradient = ctx.createRadialGradient(mouse.x, mouse.y, 0, mouse.x, mouse.y, CONFIG.mouseRadius);
    gradient.addColorStop(0, 'rgba(0, 242, 254, 0.18)');   // Electric Cyan core
    gradient.addColorStop(0.35, 'rgba(244, 63, 94, 0.12)'); // Magenta mid
    gradient.addColorStop(0.7, 'rgba(168, 85, 247, 0.06)'); // Violet outer
    gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');

    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(mouse.x, mouse.y, CONFIG.mouseRadius, 0, Math.PI * 2);
    ctx.fill();
  }

  // Animation loop
  function animate(time) {
    ctx.clearRect(0, 0, W, H);

    // Mouse interactive radial multi-color glow
    drawMouseGlow();

    // Update and draw particles
    for (const p of particles) {
      p.update(time);
      p.draw();
    }

    // Draw constellation network lines
    drawConnections();

    // Trigger shooting stars
    if (time - lastShootingStar > CONFIG.shootingStarInterval + Math.random() * 2500) {
      shootingStars.push(new ShootingStar());
      lastShootingStar = time;
    }

    // Update and draw shooting stars
    for (const star of shootingStars) {
      star.update();
      star.draw();
    }
    shootingStars = shootingStars.filter(s => s.alive);

    requestAnimationFrame(animate);
  }

  requestAnimationFrame(animate);
})();
