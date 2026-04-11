/* ===========================================================
   India Noise Pollution Zone Map — Multi-City Dashboard Logic
   Leaflet.js + Chart.js  |  Delhi + Chennai
   =========================================================== */

(function () {
    'use strict';

    var allData = null;
    var map = null;
    var markersLayer = null;
    var delhiChart = null;
    var chennaiChart = null;
    var activeCity = 'all';

    var INDIA_CENTER = [22.5, 78.5];
    var INDIA_ZOOM = 5;
    var CITY_VIEWS = {
        'Delhi':   { center: [28.6139, 77.2090], zoom: 11 },
        'Chennai': { center: [13.0827, 80.2707], zoom: 12 }
    };
    var CITY_COLORS = { 'Delhi': '#00d4ff', 'Chennai': '#ff6b35' };

    /* ---- Color by Zone Category ---- */
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
        return Math.max(6, Math.min(16, 4 + (avgDay - 45) * 0.35));
    }

    /* ---- Init ---- */
    document.addEventListener('DOMContentLoaded', function () {
        initMap();
        loadData();
        bindEvents();
    });

    function initMap() {
        map = L.map('map', { center: INDIA_CENTER, zoom: INDIA_ZOOM, zoomControl: true });
        L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
            attribution: '&copy; OSM &copy; CARTO',
            subdomains: 'abcd', maxZoom: 19
        }).addTo(map);
        markersLayer = L.layerGroup().addTo(map);
    }

    function loadData() {
        fetch('../data/geo/all_cities_locations_geo.json')
            .then(function (r) { if (!r.ok) throw new Error(); return r.json(); })
            .then(function (data) {
                allData = data;
                renderMarkers();
                renderTrendCharts();
            })
            .catch(function (e) { console.error('Load error:', e); });
    }

    /* ---- Filters ---- */
    function getFilters() {
        var year = document.getElementById('yearFilter').value;
        var zt = [];
        if (document.getElementById('ztCommercial').checked) zt.push('Commercial');
        if (document.getElementById('ztIndustrial').checked) zt.push('Industrial');
        if (document.getElementById('ztResidential').checked) zt.push('Residential');
        if (document.getElementById('ztSilence').checked) zt.push('Silence');
        var cat = [];
        if (document.getElementById('catLow').checked) cat.push('Low');
        if (document.getElementById('catModerate').checked) cat.push('Moderate');
        if (document.getElementById('catHigh').checked) cat.push('High');
        if (document.getElementById('catCritical').checked) cat.push('Critical');
        return { city: activeCity, year: year, zoneTypes: zt, categories: cat };
    }

    function getYearData(loc, year) {
        if (year === 'all') {
            return { avgDay: loc.Avg_Day, avgNight: loc.Avg_Night, zone: loc.Zone_Category };
        }
        var yn = parseInt(year, 10);
        for (var i = 0; i < loc.Yearly.length; i++) {
            if (loc.Yearly[i].Year === yn) {
                return {
                    avgDay: loc.Yearly[i].Avg_Day,
                    avgNight: loc.Yearly[i].Avg_Night,
                    zone: loc.Yearly[i].Zone_Category
                };
            }
        }
        return { avgDay: loc.Avg_Day, avgNight: loc.Avg_Night, zone: loc.Zone_Category };
    }

    /* ---- Render Markers ---- */
    function renderMarkers() {
        if (!allData) return;
        markersLayer.clearLayers();
        var f = getFilters();
        var locs = allData.locations;
        var count = 0, tDay = 0, tNight = 0;
        var cities = {};
        var maxLoc = '', maxDay = 0;
        var visibleLatLngs = [];

        for (var i = 0; i < locs.length; i++) {
            var loc = locs[i];
            if (f.city !== 'all' && loc.City !== f.city) continue;
            if (f.zoneTypes.indexOf(loc.Zone_Type) === -1) continue;

            var d = getYearData(loc, f.year);
            if (f.categories.indexOf(d.zone) === -1) continue;

            var fill = getColor(d.zone);
            var border = CITY_COLORS[loc.City] || '#ffffff';
            var r = getRadius(d.avgDay);
            var yrLabel = f.year === 'all' ? 'All Years' : f.year;

            var popup =
                '<div style="font-family:Inter,sans-serif;min-width:230px;">' +
                '<div style="font-size:14px;font-weight:700;color:' + border + ';margin-bottom:6px;' +
                    'border-bottom:2px solid ' + border + ';padding-bottom:5px;">' +
                    loc.Location + ' <span style="font-size:11px;color:#8b949e;">(' + loc.City + ')</span></div>' +
                '<table style="width:100%;font-size:12px;color:#c9d1d9;">' +
                '<tr><td style="padding:3px 0;">Zone Type</td><td style="text-align:right;font-weight:600;">' + loc.Zone_Type + '</td></tr>' +
                '<tr><td style="padding:3px 0;">Avg Day</td><td style="text-align:right;font-weight:600;">' + d.avgDay.toFixed(1) + ' dB</td></tr>' +
                '<tr><td style="padding:3px 0;">Avg Night</td><td style="text-align:right;font-weight:600;">' + d.avgNight.toFixed(1) + ' dB</td></tr>' +
                '<tr><td style="padding:3px 0;">Category</td><td style="text-align:right;font-weight:700;color:' + fill + ';">' + d.zone + '</td></tr>' +
                '<tr><td style="padding:3px 0;">Year</td><td style="text-align:right;font-weight:600;">' + yrLabel + '</td></tr>' +
                '</table></div>';

            var mk = L.circleMarker([loc.Latitude, loc.Longitude], {
                radius: r,
                fillColor: fill,
                color: border,
                weight: 2.5,
                opacity: 0.95,
                fillOpacity: 0.7
            });
            mk.bindPopup(popup, { maxWidth: 280 });
            mk.bindTooltip(loc.Location + ' (' + loc.City + ')', {
                direction: 'top', offset: [0, -r]
            });
            markersLayer.addLayer(mk);

            count++;
            tDay += d.avgDay;
            tNight += d.avgNight;
            cities[loc.City] = true;
            visibleLatLngs.push([loc.Latitude, loc.Longitude]);
            if (d.avgDay > maxDay) { maxDay = d.avgDay; maxLoc = loc.Location; }
        }

        // Stats
        document.getElementById('statTotal').textContent = count;
        document.getElementById('statCities').textContent = Object.keys(cities).join(', ') || '—';
        document.getElementById('statDay').textContent = count ? (tDay / count).toFixed(1) + ' dB' : '—';
        document.getElementById('statNight').textContent = count ? (tNight / count).toFixed(1) + ' dB' : '—';
        document.getElementById('statPolluted').textContent = maxLoc || '—';

        // Auto-zoom
        if (f.city !== 'all' && CITY_VIEWS[f.city]) {
            map.setView(CITY_VIEWS[f.city].center, CITY_VIEWS[f.city].zoom, { animate: true });
        } else if (f.city === 'all' && visibleLatLngs.length > 1) {
            try {
                map.fitBounds(L.latLngBounds(visibleLatLngs), { padding: [30, 30], animate: true });
            } catch (e) {
                map.setView(INDIA_CENTER, INDIA_ZOOM, { animate: true });
            }
        }
    }

    /* ---- Trend Charts ---- */
    function renderTrendCharts() {
        if (!allData || !allData.trends) return;
        renderOneChart('delhiTrendChart', allData.trends['Delhi'] || [], '#00d4ff', 'Delhi');
        renderOneChart('chennaiTrendChart', allData.trends['Chennai'] || [], '#ff6b35', 'Chennai');
    }

    function renderOneChart(canvasId, trend, color, city) {
        var ctx = document.getElementById(canvasId).getContext('2d');
        var labels = [], dayD = [], nightD = [];
        for (var i = 0; i < trend.length; i++) {
            labels.push(trend[i].Year.toString());
            dayD.push(trend[i].Avg_Day);
            nightD.push(trend[i].Avg_Night);
        }

        var existing = (canvasId === 'delhiTrendChart') ? delhiChart : chennaiChart;
        if (existing) existing.destroy();

        var chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Day (dB)',
                        data: dayD,
                        borderColor: color,
                        backgroundColor: color.replace(')', ',0.1)').replace('rgb', 'rgba'),
                        borderWidth: 2.5,
                        pointBackgroundColor: color,
                        pointBorderColor: '#fff',
                        pointBorderWidth: 1,
                        pointRadius: 4,
                        pointHoverRadius: 7,
                        fill: false,
                        tension: 0.3
                    },
                    {
                        label: 'Night (dB)',
                        data: nightD,
                        borderColor: '#8b949e',
                        borderWidth: 1.5,
                        borderDash: [5, 3],
                        pointBackgroundColor: '#8b949e',
                        pointBorderColor: '#fff',
                        pointBorderWidth: 1,
                        pointRadius: 3,
                        pointHoverRadius: 6,
                        fill: false,
                        tension: 0.3
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: {
                        position: 'top', align: 'end',
                        labels: {
                            color: '#8b949e', font: { family: 'Inter', size: 10 },
                            boxWidth: 10, boxHeight: 2, padding: 12
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(22,27,34,0.95)',
                        titleColor: '#e6edf3', bodyColor: '#8b949e',
                        borderColor: '#30363d', borderWidth: 1, cornerRadius: 8,
                        titleFont: { family: 'Inter', weight: '600', size: 11 },
                        bodyFont: { family: 'Inter', size: 10 }, padding: 8,
                        callbacks: {
                            title: function (items) { return city + ' — ' + items[0].label; },
                            label: function (c) { return c.dataset.label + ': ' + c.parsed.y.toFixed(1) + ' dB'; }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(48,54,61,0.3)' },
                        ticks: { color: '#8b949e', font: { family: 'Inter', size: 10 } }
                    },
                    y: {
                        grid: { color: 'rgba(48,54,61,0.3)' },
                        ticks: {
                            color: '#8b949e', font: { family: 'Inter', size: 10 },
                            callback: function (v) { return v + ' dB'; }
                        }
                    }
                }
            }
        });

        if (canvasId === 'delhiTrendChart') delhiChart = chart;
        else chennaiChart = chart;
    }

    /* ---- Events ---- */
    function bindEvents() {
        // City toggles
        var cityBtns = document.querySelectorAll('.city-btn');
        cityBtns.forEach(function (btn) {
            btn.addEventListener('click', function () {
                cityBtns.forEach(function (b) { b.classList.remove('active'); });
                btn.classList.add('active');
                activeCity = btn.getAttribute('data-city');
                renderMarkers();
            });
        });

        // Year
        document.getElementById('yearFilter').addEventListener('change', renderMarkers);

        // Checkboxes
        var cbIds = [
            'ztCommercial', 'ztIndustrial', 'ztResidential', 'ztSilence',
            'catLow', 'catModerate', 'catHigh', 'catCritical'
        ];
        cbIds.forEach(function (id) {
            document.getElementById(id).addEventListener('change', renderMarkers);
        });

        // Reset
        document.getElementById('resetFilters').addEventListener('click', function () {
            document.getElementById('yearFilter').value = 'all';
            cbIds.forEach(function (id) { document.getElementById(id).checked = true; });
            var cityBtns2 = document.querySelectorAll('.city-btn');
            cityBtns2.forEach(function (b) { b.classList.remove('active'); });
            document.getElementById('btnAllCities').classList.add('active');
            activeCity = 'all';
            map.setView(INDIA_CENTER, INDIA_ZOOM, { animate: true });
            renderMarkers();
        });
    }

})();
