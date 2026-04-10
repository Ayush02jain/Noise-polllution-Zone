/* ===========================================================
   Delhi Noise Pollution Zone Map - Dashboard Logic
   Leaflet.js + Chart.js integration
   =========================================================== */

(function () {
    'use strict';

    /* ---- State ---- */
    let allData = null;         // Full JSON data
    let map = null;             // Leaflet map instance
    let markersLayer = null;    // Leaflet layer group for markers
    let trendChart = null;      // Chart.js instance

    /* ---- Constants ---- */
    const DELHI_CENTER = [28.6139, 77.2090];
    const ZOOM_DEFAULT = 11;

    /* ---- Color Scale (by Zone Category) ---- */
    function getColor(zone) {
        switch (zone) {
            case 'Low':      return '#00b300';
            case 'Moderate': return '#ffff00';
            case 'High':     return '#ff8c00';
            case 'Critical': return '#b40000';
            default:         return '#640096';
        }
    }

    function getRadius(avgDay) {
        return Math.max(7, Math.min(16, 5 + (avgDay - 45) * 0.35));
    }

    /* ---- Init ---- */
    document.addEventListener('DOMContentLoaded', function () {
        initMap();
        loadData();
        bindEvents();
    });

    /* ---- Map Initialization ---- */
    function initMap() {
        map = L.map('map', {
            center: DELHI_CENTER,
            zoom: ZOOM_DEFAULT,
            zoomControl: true,
            attributionControl: true
        });

        // Dark tile layer
        L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org">OSM</a> &copy; <a href="https://carto.com">CARTO</a>',
            subdomains: 'abcd',
            maxZoom: 19
        }).addTo(map);

        markersLayer = L.layerGroup().addTo(map);
    }

    /* ---- Data Loading ---- */
    function loadData() {
        fetch('locations_geo.json')
            .then(function (res) {
                if (!res.ok) throw new Error('Failed to load JSON');
                return res.json();
            })
            .then(function (data) {
                allData = data;
                renderMarkers();
                renderTrendChart();
                updateStats();
            })
            .catch(function (err) {
                console.error('Error loading data:', err);
            });
    }

    /* ---- Get Active Filters ---- */
    function getFilters() {
        var year = document.getElementById('yearFilter').value;
        var zones = [];
        if (document.getElementById('zoneLow').checked) zones.push('Low');
        if (document.getElementById('zoneModerate').checked) zones.push('Moderate');
        if (document.getElementById('zoneHigh').checked) zones.push('High');
        if (document.getElementById('zoneCritical').checked) zones.push('Critical');
        return { year: year, zones: zones };
    }

    /* ---- Get Location Data for Current Year ---- */
    function getLocationDataForYear(loc, year) {
        if (year === 'all') {
            return {
                avgDay: loc.Avg_Day,
                avgNight: loc.Avg_Night,
                zone: loc.Zone_Category
            };
        }
        var yearNum = parseInt(year, 10);
        var match = null;
        for (var i = 0; i < loc.Yearly.length; i++) {
            if (loc.Yearly[i].Year === yearNum) {
                match = loc.Yearly[i];
                break;
            }
        }
        if (match) {
            return {
                avgDay: match.Avg_Day,
                avgNight: match.Avg_Night,
                zone: match.Zone_Category
            };
        }
        return {
            avgDay: loc.Avg_Day,
            avgNight: loc.Avg_Night,
            zone: loc.Zone_Category
        };
    }

    /* ---- Render Markers ---- */
    function renderMarkers() {
        if (!allData) return;

        markersLayer.clearLayers();

        var filters = getFilters();
        var locations = allData.locations;
        var visibleCount = 0;
        var totalDay = 0;
        var totalNight = 0;

        for (var i = 0; i < locations.length; i++) {
            var loc = locations[i];
            var data = getLocationDataForYear(loc, filters.year);

            // Zone filter
            if (filters.zones.indexOf(data.zone) === -1) continue;

            var color = getColor(data.zone);
            var radius = getRadius(data.avgDay);

            var marker = L.circleMarker([loc.Latitude, loc.Longitude], {
                radius: radius,
                fillColor: color,
                color: '#ffffff',
                weight: 1.5,
                opacity: 0.9,
                fillOpacity: 0.75
            });

            var yearLabel = filters.year === 'all' ? 'All Years' : filters.year;

            var popupContent =
                '<div class="popup-content">' +
                    '<div class="popup-header" style="border-bottom: 2px solid ' + color + ';">' +
                        '<strong>' + loc.Location + '</strong>' +
                    '</div>' +
                    '<table class="popup-table">' +
                        '<tr><td>Zone Type</td><td class="popup-val">' + loc.Zone_Type + '</td></tr>' +
                        '<tr><td>Avg Day</td><td class="popup-val">' + data.avgDay.toFixed(1) + ' dB</td></tr>' +
                        '<tr><td>Avg Night</td><td class="popup-val">' + data.avgNight.toFixed(1) + ' dB</td></tr>' +
                        '<tr><td>Category</td><td class="popup-val" style="color:' + color + ';">' + data.zone + '</td></tr>' +
                        '<tr><td>Year</td><td class="popup-val">' + yearLabel + '</td></tr>' +
                    '</table>' +
                '</div>';

            marker.bindPopup(popupContent, { maxWidth: 260 });
            marker.bindTooltip(loc.Location, {
                direction: 'top',
                offset: [0, -radius],
                className: 'dark-tooltip'
            });

            markersLayer.addLayer(marker);

            visibleCount++;
            totalDay += data.avgDay;
            totalNight += data.avgNight;
        }

        // Update stats
        document.getElementById('statTotal').textContent = visibleCount;
        document.getElementById('statDay').textContent = visibleCount > 0
            ? (totalDay / visibleCount).toFixed(1) + ' dB'
            : '—';
        document.getElementById('statNight').textContent = visibleCount > 0
            ? (totalNight / visibleCount).toFixed(1) + ' dB'
            : '—';
    }

    /* ---- Update Stats ---- */
    function updateStats() {
        renderMarkers();
    }

    /* ---- Trend Chart ---- */
    function renderTrendChart() {
        if (!allData || !allData.yearly_trend) return;

        var ctx = document.getElementById('trendChart').getContext('2d');
        var trend = allData.yearly_trend;

        var labels = [];
        var dayData = [];
        var nightData = [];

        for (var i = 0; i < trend.length; i++) {
            labels.push(trend[i].Year.toString());
            dayData.push(trend[i].Avg_Day);
            nightData.push(trend[i].Avg_Night);
        }

        if (trendChart) {
            trendChart.destroy();
        }

        trendChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Day Noise (dB)',
                        data: dayData,
                        borderColor: '#ff6b6b',
                        backgroundColor: 'rgba(255, 107, 107, 0.1)',
                        borderWidth: 2.5,
                        pointBackgroundColor: '#ff6b6b',
                        pointBorderColor: '#fff',
                        pointBorderWidth: 1.5,
                        pointRadius: 5,
                        pointHoverRadius: 8,
                        fill: false,
                        tension: 0.3
                    },
                    {
                        label: 'Night Noise (dB)',
                        data: nightData,
                        borderColor: '#00d4ff',
                        backgroundColor: 'rgba(0, 212, 255, 0.1)',
                        borderWidth: 2.5,
                        pointBackgroundColor: '#00d4ff',
                        pointBorderColor: '#fff',
                        pointBorderWidth: 1.5,
                        pointRadius: 5,
                        pointHoverRadius: 8,
                        fill: false,
                        tension: 0.3
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                plugins: {
                    legend: {
                        position: 'top',
                        align: 'end',
                        labels: {
                            color: '#8b949e',
                            font: { family: 'Inter', size: 11 },
                            boxWidth: 12,
                            boxHeight: 2,
                            usePointStyle: false,
                            padding: 16
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(22, 27, 34, 0.95)',
                        titleColor: '#e6edf3',
                        bodyColor: '#8b949e',
                        borderColor: '#30363d',
                        borderWidth: 1,
                        cornerRadius: 8,
                        titleFont: { family: 'Inter', weight: '600', size: 12 },
                        bodyFont: { family: 'Inter', size: 11 },
                        padding: 10,
                        callbacks: {
                            label: function (ctx) {
                                return ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(1) + ' dB';
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(48, 54, 61, 0.4)' },
                        ticks: { color: '#8b949e', font: { family: 'Inter', size: 11 } }
                    },
                    y: {
                        grid: { color: 'rgba(48, 54, 61, 0.4)' },
                        ticks: {
                            color: '#8b949e',
                            font: { family: 'Inter', size: 11 },
                            callback: function (v) { return v + ' dB'; }
                        }
                    }
                }
            }
        });
    }

    /* ---- Event Bindings ---- */
    function bindEvents() {
        // Year filter
        document.getElementById('yearFilter').addEventListener('change', function () {
            renderMarkers();
        });

        // Zone checkboxes
        var checkboxes = ['zoneLow', 'zoneModerate', 'zoneHigh', 'zoneCritical'];
        for (var i = 0; i < checkboxes.length; i++) {
            document.getElementById(checkboxes[i]).addEventListener('change', function () {
                renderMarkers();
            });
        }

        // Reset
        document.getElementById('resetFilters').addEventListener('click', function () {
            document.getElementById('yearFilter').value = 'all';
            document.getElementById('zoneLow').checked = true;
            document.getElementById('zoneModerate').checked = true;
            document.getElementById('zoneHigh').checked = true;
            document.getElementById('zoneCritical').checked = true;

            map.setView(DELHI_CENTER, ZOOM_DEFAULT, { animate: true });
            renderMarkers();
        });
    }

})();
