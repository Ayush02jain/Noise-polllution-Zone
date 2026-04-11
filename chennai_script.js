/* ===========================================================
   Chennai Noise Pollution Zone Map - Dashboard Logic
   Leaflet.js + Chart.js
   =========================================================== */

(function () {
    'use strict';

    var allData = null;
    var map = null;
    var markersLayer = null;
    var trendChart = null;

    var CHENNAI_CENTER = [13.0827, 80.2707];
    var ZOOM_DEFAULT = 12;

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
        return Math.max(8, Math.min(18, 6 + (avgDay - 45) * 0.35));
    }

    /* ---- Init ---- */
    document.addEventListener('DOMContentLoaded', function () {
        initMap();
        loadData();
        bindEvents();
    });

    function initMap() {
        map = L.map('map', {
            center: CHENNAI_CENTER,
            zoom: ZOOM_DEFAULT,
            zoomControl: true
        });
        L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org">OSM</a> &copy; <a href="https://carto.com">CARTO</a>',
            subdomains: 'abcd',
            maxZoom: 19
        }).addTo(map);
        markersLayer = L.layerGroup().addTo(map);
    }

    function loadData() {
        fetch('chennai_locations_geo.json')
            .then(function (res) {
                if (!res.ok) throw new Error('Failed to load JSON');
                return res.json();
            })
            .then(function (data) {
                allData = data;
                renderMarkers();
                renderTrendChart();
            })
            .catch(function (err) {
                console.error('Error loading data:', err);
            });
    }

    /* ---- Filters ---- */
    function getFilters() {
        var year = document.getElementById('yearFilter').value;
        var zoneTypes = [];
        if (document.getElementById('zoneCommercial').checked) zoneTypes.push('Commercial');
        if (document.getElementById('zoneIndustrial').checked) zoneTypes.push('Industrial');
        if (document.getElementById('zoneResidential').checked) zoneTypes.push('Residential');
        if (document.getElementById('zoneSilence').checked) zoneTypes.push('Silence');
        var categories = [];
        if (document.getElementById('catLow').checked) categories.push('Low');
        if (document.getElementById('catModerate').checked) categories.push('Moderate');
        if (document.getElementById('catHigh').checked) categories.push('High');
        if (document.getElementById('catCritical').checked) categories.push('Critical');
        return { year: year, zoneTypes: zoneTypes, categories: categories };
    }

    function getLocationDataForYear(loc, year) {
        if (year === 'all') {
            return { avgDay: loc.Avg_Day, avgNight: loc.Avg_Night, zone: loc.Zone_Category };
        }
        var yearNum = parseInt(year, 10);
        for (var i = 0; i < loc.Yearly.length; i++) {
            if (loc.Yearly[i].Year === yearNum) {
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

        var filters = getFilters();
        var locations = allData.locations;
        var count = 0, totalDay = 0, totalNight = 0;

        for (var i = 0; i < locations.length; i++) {
            var loc = locations[i];
            var data = getLocationDataForYear(loc, filters.year);

            // Zone Type filter
            if (filters.zoneTypes.indexOf(loc.Zone_Type) === -1) continue;
            // Zone Category filter
            if (filters.categories.indexOf(data.zone) === -1) continue;

            var color = getColor(data.zone);
            var radius = getRadius(data.avgDay);
            var yearLabel = filters.year === 'all' ? 'All Years' : filters.year;

            var popup =
                '<div style="font-family:Inter,sans-serif;min-width:220px;">' +
                    '<div style="font-size:15px;font-weight:700;color:' + color + ';margin-bottom:8px;border-bottom:2px solid ' + color + ';padding-bottom:6px;">' +
                        loc.Location +
                    '</div>' +
                    '<table style="width:100%;font-size:12px;color:#c9d1d9;">' +
                        '<tr><td style="padding:3px 0;">Zone Type</td><td style="text-align:right;font-weight:600;">' + loc.Zone_Type + '</td></tr>' +
                        '<tr><td style="padding:3px 0;">Avg Day</td><td style="text-align:right;font-weight:600;">' + data.avgDay.toFixed(1) + ' dB</td></tr>' +
                        '<tr><td style="padding:3px 0;">Avg Night</td><td style="text-align:right;font-weight:600;">' + data.avgNight.toFixed(1) + ' dB</td></tr>' +
                        '<tr><td style="padding:3px 0;">Category</td><td style="text-align:right;font-weight:700;color:' + color + ';">' + data.zone + '</td></tr>' +
                        '<tr><td style="padding:3px 0;">Year</td><td style="text-align:right;font-weight:600;">' + yearLabel + '</td></tr>' +
                    '</table>' +
                '</div>';

            var marker = L.circleMarker([loc.Latitude, loc.Longitude], {
                radius: radius,
                fillColor: color,
                color: '#ffffff',
                weight: 1.5,
                opacity: 0.9,
                fillOpacity: 0.75
            });
            marker.bindPopup(popup, { maxWidth: 280 });
            marker.bindTooltip(loc.Location, {
                direction: 'top',
                offset: [0, -radius],
                className: 'dark-tooltip'
            });
            markersLayer.addLayer(marker);

            count++;
            totalDay += data.avgDay;
            totalNight += data.avgNight;
        }

        document.getElementById('statTotal').textContent = count;
        document.getElementById('statDay').textContent = count > 0
            ? (totalDay / count).toFixed(1) + ' dB' : '—';
        document.getElementById('statNight').textContent = count > 0
            ? (totalNight / count).toFixed(1) + ' dB' : '—';
    }

    /* ---- Trend Chart ---- */
    function renderTrendChart() {
        if (!allData || !allData.yearly_trend) return;
        var ctx = document.getElementById('trendChart').getContext('2d');
        var trend = allData.yearly_trend;
        var labels = [], dayData = [], nightData = [];

        for (var i = 0; i < trend.length; i++) {
            labels.push(trend[i].Year.toString());
            dayData.push(trend[i].Avg_Day);
            nightData.push(trend[i].Avg_Night);
        }

        if (trendChart) trendChart.destroy();

        trendChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Day Noise (dB)',
                        data: dayData,
                        borderColor: '#ff6b35',
                        backgroundColor: 'rgba(255,107,53,0.1)',
                        borderWidth: 2.5,
                        pointBackgroundColor: '#ff6b35',
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
                        borderColor: '#00b4d8',
                        backgroundColor: 'rgba(0,180,216,0.1)',
                        borderWidth: 2.5,
                        pointBackgroundColor: '#00b4d8',
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
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: {
                        position: 'top', align: 'end',
                        labels: {
                            color: '#8b949e',
                            font: { family: 'Inter', size: 11 },
                            boxWidth: 12, boxHeight: 2, padding: 16
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(22,27,34,0.95)',
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
                        grid: { color: 'rgba(48,54,61,0.4)' },
                        ticks: { color: '#8b949e', font: { family: 'Inter', size: 11 } }
                    },
                    y: {
                        grid: { color: 'rgba(48,54,61,0.4)' },
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

    /* ---- Events ---- */
    function bindEvents() {
        document.getElementById('yearFilter').addEventListener('change', renderMarkers);

        var checkboxIds = [
            'zoneCommercial', 'zoneIndustrial', 'zoneResidential', 'zoneSilence',
            'catLow', 'catModerate', 'catHigh', 'catCritical'
        ];
        for (var i = 0; i < checkboxIds.length; i++) {
            document.getElementById(checkboxIds[i]).addEventListener('change', renderMarkers);
        }

        document.getElementById('resetFilters').addEventListener('click', function () {
            document.getElementById('yearFilter').value = 'all';
            var allChecks = [
                'zoneCommercial', 'zoneIndustrial', 'zoneResidential', 'zoneSilence',
                'catLow', 'catModerate', 'catHigh', 'catCritical'
            ];
            for (var i = 0; i < allChecks.length; i++) {
                document.getElementById(allChecks[i]).checked = true;
            }
            map.setView(CHENNAI_CENTER, ZOOM_DEFAULT, { animate: true });
            renderMarkers();
        });
    }

})();
