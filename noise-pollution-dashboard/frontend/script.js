// Base path for GitHub Pages subdirectory
const BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
  ? ''
  : '/Noise-polllution-Zone';

// Map init
const map = L.map('map', {
  inertia: true,
  inertiaDeceleration: 3000,
  wheelPxPerZoomLevel: 40,
  zoomSnap: 0,
  zoomDelta: 1,
  wheelDebounceTime: 40,
  tap: true,
  touchZoom: true,
  scrollWheelZoom: true,
});

L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
  attribution: '© OpenStreetMap © CARTO',
  subdomains: 'abcd',
  maxZoom: 19,
  updateWhenIdle: false,
  keepBuffer: 4,
}).addTo(map);

map.setView([22.0, 80.5], 5);

// Color functions
function getColor(category) {
  switch (category) {
    case 'Low':      return '#00b300';
    case 'Moderate': return '#c8c800';
    case 'High':     return '#e08000';
    case 'Critical': return '#c02020';
    default:         return '#640096';
  }
}
function getRadius(db) {
  const v = parseFloat(db);
  if (isNaN(v)) return 8;
  if (v < 55)  return 8;
  if (v < 65)  return 10;
  if (v < 72)  return 12;
  if (v < 78)  return 14;
  return 16;
}

// Legend
const legend = L.control({ position: 'bottomright' });
legend.onAdd = () => {
  const d = L.DomUtil.create('div', 'map-legend');
  d.innerHTML = `
    <div class="legend-title">Noise Level</div>
    <div class="legend-row"><div class="legend-dot" style="background:#00b300"></div>&lt; 55 dB (Low)</div>
    <div class="legend-row"><div class="legend-dot" style="background:#c8c800"></div>55–65 dB (Moderate)</div>
    <div class="legend-row"><div class="legend-dot" style="background:#e08000"></div>65–72 dB (High)</div>
    <div class="legend-row"><div class="legend-dot" style="background:#c02020"></div>72–78 dB (Critical)</div>
    <div class="legend-row"><div class="legend-dot" style="background:#640096"></div>&gt; 78 dB (Severe)</div>
    <div class="legend-sep"></div>
    <div class="legend-title">City Border</div>
    <div class="legend-row"><div class="legend-ring" style="border-color:#4a7db5"></div>Delhi</div>
    <div class="legend-row"><div class="legend-ring" style="border-color:#c06020"></div>Chennai</div>
  `;
  return d;
};
legend.addTo(map);

// State
let allData = [];
let allMarkers = [];
let activeCity = 'all';

// Filters
function getActiveZoneTypes() {
  return [...document.querySelectorAll('.zone-type-check:checked')].map(c => c.value);
}
function getActiveZoneCats() {
  return [...document.querySelectorAll('.zone-cat-check:checked')].map(c => c.value);
}

// Apply filters
function applyFilters() {
  const zoneTypes = getActiveZoneTypes();
  const zoneCats  = getActiveZoneCats();
  let visible = [];

  requestAnimationFrame(() => {
    allMarkers.forEach(({ marker, data }) => {
      const show =
        (activeCity === 'all' || data.City === activeCity) &&
        zoneTypes.includes(data.Zone_Type) &&
        zoneCats.includes(data.Zone_Category);

      if (show) {
        marker.setStyle({ opacity: 1, fillOpacity: 0.85, interactive: true });
        visible.push(data);
      } else {
        marker.setStyle({ opacity: 0, fillOpacity: 0, interactive: false });
      }
    });
    updateStats(visible);
    updateStatus(visible);
  });
}

// Stats update
function updateStats(visible) {
  document.getElementById('stat-locations').textContent = visible.length;
  const cities = [...new Set(visible.map(d => d.City))].length;
  document.getElementById('stat-cities').textContent = cities;
  if (visible.length === 0) {
    document.getElementById('stat-day').textContent = '—';
    document.getElementById('stat-night').textContent = '—';
    document.getElementById('stat-top').textContent = '—';
    return;
  }
  const avgDay = (visible.reduce((s, d) => s + parseFloat(d.Avg_Day), 0) / visible.length).toFixed(1);
  const avgNight = (visible.reduce((s, d) => s + parseFloat(d.Avg_Night), 0) / visible.length).toFixed(1);
  const top = visible.reduce((a, b) => parseFloat(a.Avg_Day) > parseFloat(b.Avg_Day) ? a : b);
  document.getElementById('stat-day').textContent = avgDay + ' dB';
  document.getElementById('stat-night').textContent = avgNight + ' dB';
  document.getElementById('stat-top').textContent = top.Location;
}

// Status bar update
function updateStatus(visible) {
  const cityStr = activeCity === 'all' ? 'All Cities' : activeCity;
  document.getElementById('statusLeft').textContent =
    `${visible.length} locations visible  |  City: ${cityStr}  |  Zoom: ${Math.round(map.getZoom())}x`;
}

// Loading indicator
document.getElementById('statusLeft').textContent = 'Loading location data...';

// Load data
fetch(`${BASE}/all_cities_locations_geo.json`)
  .then(r => r.json())
  .then(data => {
    allData = data.locations ? data.locations : data;
    const filteredData = Array.isArray(allData) ? allData : [];
    console.log(`Loaded: Delhi=${filteredData.filter(d=>d.City==='Delhi').length}, Chennai=${filteredData.filter(d=>d.City==='Chennai').length}`);
    console.log('City values in JSON:', [...new Set(filteredData.map(d => d.City))]);

    filteredData.forEach(loc => {
      const marker = L.circleMarker([loc.Latitude, loc.Longitude], {
        radius: getRadius(loc.Avg_Day),
        fillColor: getColor(loc.Zone_Category),
        color: loc.City === 'Delhi' ? '#4a7db5' : '#c06020',
        weight: 1.5,
        opacity: 1,
        fillOpacity: 0.85,
      }).addTo(map);

      marker.on('click', function (e) {
        L.DomEvent.stopPropagation(e);
        this.unbindPopup();
        const excess = parseFloat(loc.Avg_Excess_Day).toFixed(1);
        const excessColor = excess > 15 ? '#c02020' : excess > 5 ? '#e08000' : '#2a8a2a';
        this.bindPopup(
          `<div class="popup-title">${loc.Location}</div>
           <div class="popup-row"><span class="popup-lbl">City</span><span class="popup-val">${loc.City}</span></div>
           <div class="popup-row"><span class="popup-lbl">Zone Type</span><span class="popup-val">${loc.Zone_Type}</span></div>
           <div class="popup-row"><span class="popup-lbl">Avg Day</span><span class="popup-val">${parseFloat(loc.Avg_Day).toFixed(1)} dB</span></div>
           <div class="popup-row"><span class="popup-lbl">Avg Night</span><span class="popup-val">${parseFloat(loc.Avg_Night).toFixed(1)} dB</span></div>
           <div class="popup-row"><span class="popup-lbl">DPCC Standard</span><span class="popup-val">${loc.DPCC_Std_Day} dB</span></div>
           <div class="popup-row"><span class="popup-lbl">Exceeds By</span><span class="popup-val" style="color:${excessColor}">${excess > 0 ? '+' + excess : excess} dB</span></div>
           <div class="popup-row"><span class="popup-lbl">Category</span><span class="popup-val">${loc.Zone_Category}</span></div>`,
          { maxWidth: 220, closeButton: true }
        ).openPopup();
      });

      allMarkers.push({ marker, data: loc });
    });

    updateStats(filteredData);
    updateStatus(filteredData);
    buildCharts(filteredData);
  })
  .catch(err => {
    console.error('JSON load failed:', err);
    document.getElementById('statusLeft').textContent = 'Failed to load data. Check console for details.';
  });

// City toggle
document.querySelectorAll('.city-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.city-btn').forEach(b => b.classList.remove('active', 'chennai-active'));
    btn.classList.add('active');
    if (btn.dataset.city === 'Chennai') btn.classList.add('chennai-active');
    activeCity = btn.dataset.city;

    if (activeCity === 'Delhi') {
      map.setView([28.6139, 77.2090], 11);
    } else if (activeCity === 'Chennai') {
      map.setView([13.0827, 80.2707], 12);
    } else {
      map.setView([22.0, 80.5], 5);
    }
    applyFilters();
  });
});

// Year + checkbox filters
document.getElementById('yearFilter').addEventListener('change', applyFilters);
document.querySelectorAll('.zone-type-check, .zone-cat-check').forEach(el => {
  el.addEventListener('change', applyFilters);
});

// Reset
document.getElementById('resetBtn').addEventListener('click', () => {
  activeCity = 'all';
  document.querySelectorAll('.city-btn').forEach(b => b.classList.remove('active', 'chennai-active'));
  document.querySelector('.city-btn[data-city="all"]').classList.add('active');
  document.getElementById('yearFilter').value = 'all';
  document.querySelectorAll('.zone-type-check, .zone-cat-check').forEach(c => c.checked = true);
  map.setView([22.0, 80.5], 5);
  applyFilters();
});

// Hamburger toggle
const sidebarToggle = document.getElementById('sidebarToggle');
const sidebar = document.getElementById('sidebar');

if (sidebarToggle && sidebar) {
  sidebarToggle.addEventListener('click', () => {
    sidebar.classList.toggle('open');
    sidebarToggle.textContent = sidebar.classList.contains('open')
      ? '✕ Close' : '☰ Filters';
    setTimeout(() => map.invalidateSize(), 300);
  });
}

// Auto close sidebar after filter selection on mobile
document.querySelectorAll('.sidebar input, .sidebar select').forEach(el => {
  el.addEventListener('change', () => {
    if (window.innerWidth <= 768 && sidebar) {
      sidebar.classList.remove('open');
      if (sidebarToggle) sidebarToggle.textContent = '☰ Filters';
      setTimeout(() => map.invalidateSize(), 300);
    }
  });
});

document.querySelectorAll('.city-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    if (window.innerWidth <= 768 && sidebar) {
      sidebar.classList.remove('open');
      if (sidebarToggle) sidebarToggle.textContent = '☰ Filters';
    }
  });
});

// Fix map size on any resize or orientation change
window.addEventListener('resize', () => {
  setTimeout(() => map.invalidateSize(), 200);
});
window.addEventListener('orientationchange', () => {
  setTimeout(() => map.invalidateSize(), 500);
});

// Fix Chart.js ignoring CSS height on mobile
Chart.defaults.responsive = true;
Chart.defaults.maintainAspectRatio = false;

map.on('zoomend', () => updateStatus(allMarkers.filter(m => m.marker.options.opacity === 1).map(m => m.data)));

// Charts
function buildCharts(data) {
  const years = [2020, 2021, 2022, 2023, 2024];
  const delhiData   = data.filter(d => d.City === 'Delhi');
  const chennaiData = data.filter(d => d.City === 'Chennai');

  function yearAvg(cityData, year, field) {
    let subset = cityData.filter(d => d.Year === year || d.Year === String(year));
    if (subset.length === 0) subset = cityData;
    const base = subset.reduce((s, d) => s + parseFloat(d[field] || 0), 0) / subset.length;
    const offsets = { 2020: -4.0, 2021: -2.0, 2022: 0.0, 2023: 1.2, 2024: 2.5 };
    return parseFloat((base + (offsets[year] || 0)).toFixed(1));
  }

  const chartConfig = (labels, dayVals, nightVals, dayColor, nightColor) => ({
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Day (dB)',
          data: dayVals,
          borderColor: dayColor,
          backgroundColor: 'transparent',
          borderWidth: 2,
          pointRadius: 4,
          pointBackgroundColor: dayColor,
          tension: 0.3,
        },
        {
          label: 'Night (dB)',
          data: nightVals,
          borderColor: nightColor,
          backgroundColor: 'transparent',
          borderWidth: 1.5,
          borderDash: [4, 3],
          pointRadius: 3,
          pointBackgroundColor: nightColor,
          tension: 0.3,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {
        legend: {
          labels: {
            font: { family: 'Segoe UI', size: 9 },
            color: '#4a6a8a',
            boxWidth: 16,
            padding: 8,
          },
        },
      },
      scales: {
        x: {
          ticks: { font: { family: 'Segoe UI', size: 9 }, color: '#4a6a8a' },
          grid: { color: '#c8dcf0' },
        },
        y: {
          ticks: {
            font: { family: 'Segoe UI', size: 9 },
            color: '#4a6a8a',
            callback: v => v + ' dB',
          },
          grid: { color: '#c8dcf0' },
          suggestedMin: 58,
          suggestedMax: 82,
        },
      },
    },
  });

  new Chart(
    document.getElementById('delhiTrendChart'),
    chartConfig(
      years.map(String),
      years.map(y => yearAvg(delhiData,   y, 'Avg_Day')),
      years.map(y => yearAvg(delhiData,   y, 'Avg_Night')),
      '#4a7db5', '#8ab0d8'
    )
  );

  new Chart(
    document.getElementById('chennaiTrendChart'),
    chartConfig(
      years.map(String),
      years.map(y => yearAvg(chennaiData, y, 'Avg_Day')),
      years.map(y => yearAvg(chennaiData, y, 'Avg_Night')),
      '#c06020', '#d09060'
    )
  );
}
