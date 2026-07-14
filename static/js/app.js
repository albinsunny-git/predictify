// Predictify ML: Frontend Core Logic & Charting

document.addEventListener("DOMContentLoaded", () => {
    // UI Elements
    const searchForm = document.getElementById("search-form");
    const tickerInput = document.getElementById("ticker-input");
    const searchBtn = document.getElementById("search-btn");
    const loadingIcon = searchBtn.querySelector(".loading-icon");
    const searchBtnText = searchBtn.querySelector("span");
    const quickTickersList = document.getElementById("quick-tickers-list");
    const dashboardLoader = document.getElementById("dashboard-loader");
    const dashboardContent = document.getElementById("dashboard-content");

    // Stat Cards
    const stockPriceEl = document.getElementById("stock-price");
    const stockChangeEl = document.getElementById("stock-change");
    const signalBadgeEl = document.getElementById("signal-badge");
    const signalReasonEl = document.getElementById("signal-reason");
    const rsiValueEl = document.getElementById("rsi-value");
    const rsiProgressEl = document.getElementById("rsi-progress");
    const rsiStatusEl = document.getElementById("rsi-status");
    const macdStatusEl = document.getElementById("macd-status");
    const macdValueEl = document.getElementById("macd-value");

    // Overlays & Models checkboxes
    const toggles = {
        sma20: document.getElementById("toggle-sma20"),
        sma50: document.getElementById("toggle-sma50"),
        ema20: document.getElementById("toggle-ema20"),
        bb: document.getElementById("toggle-bb"),
        lr: document.getElementById("toggle-lr"),
        rf: document.getElementById("toggle-rf"),
        lstm: document.getElementById("toggle-lstm"),
        chartType: document.getElementById("toggle-chart-type")
    };

    // Theme Switcher Logic
    const themeToggleBtn = document.getElementById("theme-toggle-btn");
    const themeIcon = themeToggleBtn.querySelector("i");
    
    // Check local storage for theme
    const savedTheme = localStorage.getItem("theme") || "dark";
    if (savedTheme === "light") {
        document.body.classList.add("light-mode");
        themeIcon.className = "fa-solid fa-sun";
    } else {
        document.body.classList.remove("light-mode");
        themeIcon.className = "fa-solid fa-moon";
    }

    themeToggleBtn.addEventListener("click", () => {
        if (document.body.classList.contains("light-mode")) {
            document.body.classList.remove("light-mode");
            themeIcon.className = "fa-solid fa-moon";
            localStorage.setItem("theme", "dark");
        } else {
            document.body.classList.add("light-mode");
            themeIcon.className = "fa-solid fa-sun";
            localStorage.setItem("theme", "light");
        }
        // If chart exists, re-render it with new theme colors
        if (mainChart && currentStockData) {
            renderChart(currentStockData);
        }
    });

    // Table bodies
    const forecastTableBody = document.getElementById("forecast-table-body");
    
    // Chart Globals
    let mainChart = null;
    let currentStockData = null; // Cache for current ticker responses

    // 1. Fetch and render popular tickers list on startup
    fetchPopularTickers();

    // 2. Initial analysis of AAPL as default
    loadTickerData("AAPL");

    // 3. Search Form Submit handler
    searchForm.addEventListener("submit", (e) => {
        e.preventDefault();
        const symbol = tickerInput.value.trim().toUpperCase();
        if (symbol) {
            loadTickerData(symbol);
        }
    });

    // 4. Set up toggles behavior
    Object.keys(toggles).forEach(key => {
        toggles[key].addEventListener("change", () => {
            if (mainChart) {
                updateChartDatasetVisibility();
            }
        });
    });

    // Fetch popular tickers config
    async function fetchPopularTickers() {
        try {
            const response = await fetch("/api/tickers");
            const tickers = await response.json();
            
            quickTickersList.innerHTML = "";
            tickers.forEach(t => {
                const pill = document.createElement("button");
                pill.className = "ticker-pill";
                pill.textContent = t.symbol;
                pill.title = t.name;
                pill.addEventListener("click", () => {
                    tickerInput.value = t.symbol;
                    loadTickerData(t.symbol);
                });
                quickTickersList.appendChild(pill);
            });
        } catch (err) {
            console.error("Failed to load popular tickers", err);
        }
    }

    // Call API and handle UI state transitions
    async function loadTickerData(ticker) {
        // Toggle loader UI
        showLoader(true);
        setActiveTickerPill(ticker);

        try {
            const response = await fetch(`/api/predict?ticker=${ticker}`);
            const payload = await response.json();

            if (payload.status === "success") {
                currentStockData = payload.data;
                renderDashboard(currentStockData);
                showLoader(false);
            } else {
                alert(`Error: ${payload.message}`);
                showLoader(false);
            }
        } catch (error) {
            console.error("API error:", error);
            alert("An error occurred while communicating with the model server.");
            showLoader(false);
        }
    }

    function showLoader(isLoading) {
        if (isLoading) {
            dashboardLoader.style.display = "flex";
            dashboardContent.style.opacity = "0.3";
            dashboardContent.style.pointerEvents = "none";
            loadingIcon.style.display = "inline-block";
            searchBtnText.textContent = "Analyzing";
            searchBtn.disabled = true;
        } else {
            dashboardLoader.style.display = "none";
            dashboardContent.style.opacity = "1";
            dashboardContent.style.pointerEvents = "all";
            loadingIcon.style.display = "none";
            searchBtnText.textContent = "Analyze";
            searchBtn.disabled = false;
        }
    }

    function setActiveTickerPill(ticker) {
        const pills = document.querySelectorAll(".ticker-pill");
        pills.forEach(p => {
            if (p.textContent === ticker) {
                p.classList.add("active");
            } else {
                p.classList.remove("active");
            }
        });
    }

    // Main layout renderer
    function renderDashboard(data) {
        // Set chart title
        document.getElementById("chart-main-title").textContent = `${data.ticker} Price Forecast Chart`;

        // 1. Render Stats
        renderStats(data);
        
        // 2. Render Chart
        renderChart(data);

        // 3. Render Metrics Table
        renderMetrics(data);

        // 4. Render Forecasts Table
        renderForecastsTable(data);
    }

    function renderStats(data) {
        const len = data.close.length;
        if (len === 0) return;

        const currentPrice = data.close[len - 1];
        const prevPrice = data.close[len - 2];
        const change = currentPrice - prevPrice;
        const changePct = (change / prevPrice) * 100;

        stockPriceEl.textContent = `$${currentPrice.toFixed(2)}`;
        
        // Show change
        if (change >= 0) {
            stockChangeEl.className = "stat-change up";
            stockChangeEl.innerHTML = `<i class="fa-solid fa-caret-up"></i> +$${change.toFixed(2)} (+${changePct.toFixed(2)}%)`;
        } else {
            stockChangeEl.className = "stat-change down";
            stockChangeEl.innerHTML = `<i class="fa-solid fa-caret-down"></i> -$${Math.abs(change).toFixed(2)} (-${changePct.toFixed(2)}%)`;
        }

        // RSI Assessment
        const rsiArray = data.indicators.rsi;
        const rsiVal = rsiArray[rsiArray.length - 1];
        
        if (rsiVal !== null && rsiVal !== "") {
            const val = parseFloat(rsiVal);
            rsiValueEl.textContent = val.toFixed(1);
            rsiProgressEl.style.width = `${val}%`;

            if (val >= 70) {
                rsiStatusEl.className = "stat-desc price-text-down";
                rsiStatusEl.textContent = "Overbought (Bearish)";
            } else if (val <= 30) {
                rsiStatusEl.className = "stat-desc price-text-up";
                rsiStatusEl.textContent = "Oversold (Bullish)";
            } else {
                rsiStatusEl.className = "stat-desc";
                rsiStatusEl.textContent = "Neutral";
            }
        } else {
            rsiValueEl.textContent = "N/A";
            rsiProgressEl.style.width = "0%";
            rsiStatusEl.textContent = "Insufficient Data";
        }

        // MACD Crossover Assessment
        const macdArr = data.indicators.macd;
        const macdSigArr = data.indicators.macd_signal;
        const macdVal = macdArr[macdArr.length - 1];
        const macdSigVal = macdSigArr[macdSigArr.length - 1];

        if (macdVal !== "" && macdSigVal !== "") {
            const mVal = parseFloat(macdVal);
            const sVal = parseFloat(macdSigVal);
            
            macdValueEl.textContent = `MACD: ${mVal.toFixed(3)} | Signal: ${sVal.toFixed(3)}`;
            
            if (mVal > sVal) {
                macdStatusEl.className = "stat-value price-text-up";
                macdStatusEl.textContent = "Bullish";
            } else {
                macdStatusEl.className = "stat-value price-text-down";
                macdStatusEl.textContent = "Bearish";
            }
        } else {
            macdStatusEl.className = "stat-value";
            macdStatusEl.textContent = "N/A";
            macdValueEl.textContent = "Insufficient Data";
        }

        // Dynamic Signal Recommendation System
        calculateTradingSignal(data, currentPrice);
    }

    function calculateTradingSignal(data, currentPrice) {
        let bullishPoints = 0;
        let bearishPoints = 0;
        let reasons = [];

        // 1. RSI Rule
        const rsiVal = parseFloat(data.indicators.rsi[data.indicators.rsi.length - 1]);
        if (!isNaN(rsiVal)) {
            if (rsiVal <= 35) {
                bullishPoints += 1.5;
                reasons.push("RSI oversold indicates turning point");
            } else if (rsiVal >= 65) {
                bearishPoints += 1.5;
                reasons.push("RSI overbought signals correction risk");
            }
        }

        // 2. MACD Crossover Rule
        const macd = parseFloat(data.indicators.macd[data.indicators.macd.length - 1]);
        const signal = parseFloat(data.indicators.macd_signal[data.indicators.macd_signal.length - 1]);
        if (!isNaN(macd) && !isNaN(signal)) {
            if (macd > signal) {
                bullishPoints += 1.0;
                reasons.push("MACD bullish crossover");
            } else {
                bearishPoints += 1.0;
                reasons.push("MACD bearish crossover");
            }
        }

        // 3. Price vs MA Rules
        const sma20 = parseFloat(data.indicators.sma_20[data.indicators.sma_20.length - 1]);
        if (!isNaN(sma20)) {
            if (currentPrice > sma20) {
                bullishPoints += 0.8;
                reasons.push("Price above 20-day SMA");
            } else {
                bearishPoints += 0.8;
                reasons.push("Price below 20-day SMA");
            }
        }

        const ema20 = parseFloat(data.indicators.ema_20[data.indicators.ema_20.length - 1]);
        if (!isNaN(ema20)) {
            if (currentPrice > ema20) {
                bullishPoints += 0.7;
            } else {
                bearishPoints += 0.7;
            }
        }

        // Compile verdict
        const netScore = bullishPoints - bearishPoints;
        let signalClass = "";
        let signalText = "";
        
        if (netScore >= 1.0) {
            signalText = "BUY";
            signalClass = "buy";
        } else if (netScore <= -1.0) {
            signalText = "SELL";
            signalClass = "sell";
        } else {
            signalText = "HOLD";
            signalClass = "hold";
        }

        signalBadgeEl.textContent = signalText;
        signalBadgeEl.className = `stat-value signal-value ${signalClass}`;
        
        if (reasons.length > 0) {
            signalReasonEl.textContent = reasons.slice(0, 2).join(" & ");
        } else {
            signalReasonEl.textContent = "Indicators are in alignment";
        }
    }

    // ApexCharts implementation
    function renderChart(data) {
        const histLen = data.dates.length;
        const lastHistClose = data.close[histLen - 1];
        const isCandle = toggles.chartType.checked;
        const isLightMode = document.body.classList.contains("light-mode");

        // 1. Historical series
        let historicalSeriesData = [];
        for (let i = 0; i < histLen; i++) {
            if (isCandle) {
                historicalSeriesData.push({
                    x: data.dates[i],
                    y: [
                        parseFloat(data.open[i]),
                        parseFloat(data.high[i]),
                        parseFloat(data.low[i]),
                        parseFloat(data.close[i])
                    ]
                });
            } else {
                historicalSeriesData.push({
                    x: data.dates[i],
                    y: parseFloat(data.close[i])
                });
            }
        }

        // 2. Overlays helper
        function buildOverlayData(indicatorList) {
            let list = [];
            for (let i = 0; i < histLen; i++) {
                const val = indicatorList[i];
                if (val !== null && val !== "") {
                    list.push({ x: data.dates[i], y: parseFloat(val) });
                }
            }
            return list;
        }

        // 3. Fitted helper
        function buildFittedData(fittedList) {
            let list = [];
            for (let i = 0; i < histLen; i++) {
                const val = fittedList[i];
                if (val !== null && val !== undefined) {
                    list.push({ x: data.dates[i], y: parseFloat(val) });
                }
            }
            return list;
        }

        // 4. Forecast helper
        function buildForecastData(forecastList) {
            let list = [{ x: data.dates[histLen - 1], y: lastHistClose }];
            for (let i = 0; i < forecastList.length; i++) {
                list.push({ x: data.forecast.dates[i], y: parseFloat(forecastList[i]) });
            }
            return list;
        }

        const series = [];

        // Historical Close is always visible
        series.push({
            name: 'Historical Close',
            type: isCandle ? 'candlestick' : 'area',
            data: historicalSeriesData
        });

        // Add indicator overlays if checked
        if (toggles.sma20.checked) {
            series.push({
                name: 'SMA 20',
                type: 'line',
                data: buildOverlayData(data.indicators.sma_20)
            });
        }
        if (toggles.sma50.checked) {
            series.push({
                name: 'SMA 50',
                type: 'line',
                data: buildOverlayData(data.indicators.sma_50)
            });
        }
        if (toggles.ema20.checked) {
            series.push({
                name: 'EMA 20',
                type: 'line',
                data: buildOverlayData(data.indicators.ema_20)
            });
        }
        if (toggles.bb.checked) {
            series.push({
                name: 'BB Upper',
                type: 'line',
                data: buildOverlayData(data.indicators.bb_upper)
            });
            series.push({
                name: 'BB Lower',
                type: 'line',
                data: buildOverlayData(data.indicators.bb_lower)
            });
        }

        // Add fitted lines if checked
        if (toggles.lr.checked) {
            series.push({
                name: 'Lin Reg (Fitted)',
                type: 'line',
                data: buildFittedData(data.fitted.lr)
            });
            series.push({
                name: 'Lin Reg Forecast',
                type: 'line',
                data: buildForecastData(data.forecast.lr)
            });
        }
        if (toggles.rf.checked) {
            series.push({
                name: 'Random Forest (Fitted)',
                type: 'line',
                data: buildFittedData(data.fitted.rf)
            });
            series.push({
                name: 'Random Forest Forecast',
                type: 'line',
                data: buildForecastData(data.forecast.rf)
            });
        }
        if (toggles.lstm.checked) {
            series.push({
                name: 'LSTM (Fitted)',
                type: 'line',
                data: buildFittedData(data.fitted.lstm)
            });
            series.push({
                name: 'LSTM Forecast',
                type: 'line',
                data: buildForecastData(data.forecast.lstm)
            });
        }

        const options = {
            chart: {
                height: '100%',
                type: 'line',
                fontFamily: 'Inter, sans-serif',
                toolbar: {
                    show: true,
                    tools: {
                        download: true,
                        selection: true,
                        zoom: true,
                        zoomin: true,
                        zoomout: true,
                        pan: true,
                        reset: true
                    }
                },
                background: 'transparent',
                animations: {
                    enabled: true,
                    easing: 'easeinout',
                    speed: 800,
                    animateGradually: {
                        enabled: true,
                        delay: 150
                    },
                    dynamicAnimation: {
                        enabled: true,
                        speed: 350
                    }
                }
            },
            theme: {
                mode: isLightMode ? 'light' : 'dark'
            },
            series: series,
            stroke: {
                width: series.map(s => {
                    if (s.name === 'Historical Close') return isCandle ? 1.5 : 2.5;
                    if (s.name.includes('Forecast')) return 2;
                    if (s.name.includes('Fitted')) return 1.2;
                    return 1.5; // Indicators
                }),
                dashArray: series.map(s => {
                    if (s.name === 'SMA 20') return 3;
                    if (s.name === 'SMA 50') return 5;
                    if (s.name === 'BB Upper' || s.name === 'BB Lower') return 4;
                    if (s.name.includes('Forecast')) return 5;
                    return 0;
                })
            },
            colors: series.map(s => {
                if (s.name === 'Historical Close') return '#a78bfa';
                if (s.name === 'SMA 20') return '#eab308';
                if (s.name === 'SMA 50') return '#f97316';
                if (s.name === 'EMA 20') return '#06b6d4';
                if (s.name === 'BB Upper' || s.name === 'BB Lower') return isLightMode ? 'rgba(0, 0, 0, 0.25)' : 'rgba(255, 255, 255, 0.25)';
                if (s.name.includes('Lin Reg')) return '#3b82f6';
                if (s.name.includes('Random Forest')) return '#10b981';
                if (s.name.includes('LSTM')) return '#ec4899';
                return '#a78bfa';
            }),
            plotOptions: {
                candlestick: {
                    colors: {
                        upward: '#089981',   // Bullish green
                        downward: '#f23645'  // Bearish red
                    },
                    wick: {
                        useFillColor: true
                    }
                }
            },
            fill: {
                type: series.map(s => s.name === 'Historical Close' && !isCandle ? 'gradient' : 'solid'),
                gradient: {
                    shade: isLightMode ? 'light' : 'dark',
                    type: 'vertical',
                    shadeIntensity: 0.5,
                    inverseColors: false,
                    opacityFrom: 0.4,
                    opacityTo: 0.05,
                    stops: [0, 90, 100]
                }
            },
            markers: {
                size: series.map(s => s.name.includes('Forecast') ? 4 : 0),
                strokeWidth: 0,
                hover: {
                    size: 6
                }
            },
            grid: {
                borderColor: isLightMode ? 'rgba(0,0,0,0.06)' : 'rgba(255,255,255,0.05)',
                xaxis: {
                    lines: {
                        show: true
                    }
                },
                yaxis: {
                    lines: {
                        show: true
                    }
                }
            },
            xaxis: {
                type: 'category',
                labels: {
                    style: {
                        colors: isLightMode ? '#4b5563' : '#9ca3af',
                        fontFamily: 'Inter',
                        fontSize: '10px'
                    },
                    maxTicksLimit: 12,
                    formatter: function(val) {
                        if (!val) return '';
                        const parts = val.split('-');
                        if (parts.length === 3) {
                            return `${parts[1]}/${parts[0].substring(2)}`;
                        }
                        return val;
                    }
                },
                axisBorder: { show: false },
                axisTicks: { show: false }
            },
            yaxis: {
                labels: {
                    style: {
                        colors: isLightMode ? '#4b5563' : '#9ca3af',
                        fontFamily: 'Inter',
                        fontSize: '11px'
                    },
                    formatter: function(value) {
                        return '$' + value.toFixed(2);
                    }
                }
            },
            tooltip: {
                theme: isLightMode ? 'light' : 'dark',
                shared: true,
                intersect: false,
                x: {
                    show: true
                },
                y: {
                    formatter: function (val) {
                        if (val === undefined || val === null) return '';
                        if (Array.isArray(val)) {
                            return `<br/>O: $${val[0].toFixed(2)}<br/>H: $${val[1].toFixed(2)}<br/>L: $${val[2].toFixed(2)}<br/>C: $${val[3].toFixed(2)}`;
                        }
                        return '$' + val.toFixed(2);
                    }
                }
            },
            legend: {
                show: true,
                position: 'top',
                horizontalAlign: 'center',
                labels: {
                    colors: isLightMode ? '#1f2937' : '#f3f4f6'
                },
                itemMargin: {
                    horizontal: 10,
                    vertical: 5
                }
            }
        };

        if (mainChart) {
            mainChart.destroy();
        }
        mainChart = new ApexCharts(document.getElementById('mainChart'), options);
        mainChart.render();
    }

    function updateChartDatasetVisibility() {
        if (mainChart && currentStockData) {
            renderChart(currentStockData);
        }
    }

    // Render comparison metrics table
    function renderMetrics(data) {
        const models = ['lr', 'rf', 'lstm'];
        
        let bestModel = 'lr';
        let bestR2 = -999;
        
        models.forEach(m => {
            const container = document.getElementById(`metric-row-${m}`);
            const metrics = data.metrics[m];
            
            container.querySelector(".metric-rmse").textContent = `$${metrics.rmse.toFixed(2)}`;
            container.querySelector(".metric-mae").textContent = `$${metrics.mae.toFixed(2)}`;
            container.querySelector(".metric-r2").textContent = metrics.r2.toFixed(4);

            if (metrics.r2 > bestR2) {
                bestR2 = metrics.r2;
                bestModel = m;
            }
        });

        // Set evaluation badges
        models.forEach(m => {
            const container = document.getElementById(`metric-row-${m}`);
            const badge = container.querySelector(".eval-badge");
            if (m === bestModel) {
                badge.className = "eval-badge badge-success";
                badge.textContent = "Top Performer";
            } else if (m === 'lr') {
                badge.className = "eval-badge badge-neutral";
                badge.textContent = "Baseline";
            } else {
                badge.className = "eval-badge badge-neutral";
                badge.textContent = "Alternative";
            }
        });

        // Render recommendation box
        const verdictBox = document.getElementById("best-model-verdict");
        const verdictTextP = verdictBox.querySelector(".verdict-text p");
        
        let modelNiceName = "";
        let details = "";
        
        if (bestModel === 'lr') {
            modelNiceName = "Linear Regression";
            details = `This baseline model shows the highest correlation coefficient (R2: ${bestR2.toFixed(4)}) on test data, suggesting stable trend continuation.`;
        } else if (bestModel === 'rf') {
            modelNiceName = "Random Forest Regressor";
            details = `The ensemble decision tree algorithm performed best (R2: ${bestR2.toFixed(4)}), indicating strong adaptation to non-linear stock fluctuations.`;
        } else {
            modelNiceName = "LSTM Neural Network";
            details = `Deep Learning Long Short-Term Memory network outperformed other models (R2: ${bestR2.toFixed(4)}), demonstrating superior performance in retaining long-term historical context.`;
        }

        verdictBox.querySelector("h4").innerHTML = `Recommended Model: <span style="color:var(--success);">${modelNiceName}</span>`;
        verdictTextP.textContent = details;
    }

    // Render future prediction table lists
    function renderForecastsTable(data) {
        forecastTableBody.innerHTML = "";
        
        const len = data.forecast.dates.length;
        for (let i = 0; i < len; i++) {
            const tr = document.createElement("tr");
            
            // Format dates slightly nicer
            const dateStr = data.forecast.dates[i];
            
            // Linear regression forecast change direction
            const prevLR = i === 0 ? data.close[data.close.length - 1] : data.forecast.lr[i - 1];
            const currLR = data.forecast.lr[i];
            const lrClass = currLR >= prevLR ? "price-text-up" : "price-text-down";
            
            // Random Forest forecast direction
            const prevRF = i === 0 ? data.close[data.close.length - 1] : data.forecast.rf[i - 1];
            const currRF = data.forecast.rf[i];
            const rfClass = currRF >= prevRF ? "price-text-up" : "price-text-down";

            // LSTM forecast direction
            const prevLSTM = i === 0 ? data.close[data.close.length - 1] : data.forecast.lstm[i - 1];
            const currLSTM = data.forecast.lstm[i];
            const lstmClass = currLSTM >= prevLSTM ? "price-text-up" : "price-text-down";

            tr.innerHTML = `
                <td style="font-weight: 500;">Day ${i+1} <span style="font-size:0.75rem; color:var(--text-muted); font-weight:400;">(${dateStr})</span></td>
                <td class="${lrClass}">$${currLR.toFixed(2)}</td>
                <td class="${rfClass}">$${currRF.toFixed(2)}</td>
                <td class="${lstmClass}">$${currLSTM.toFixed(2)}</td>
            `;
            forecastTableBody.appendChild(tr);
        }
    }
});
