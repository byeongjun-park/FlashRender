/* Stage by stage - metric charts over an NFE sweep.

   The encoding is two-fold: colour is RETA on/off, line style is the model
   (FlashRender / FlashRender-MS). Four colours for four series would bury this chart's
   whole point, the difference between RETA on and off, under colour contrast. This way
   only two colours are needed, and identity never rests on colour alone.

   The palette is slots 1 and 2 of the documented default. It passes all six checks
   (all-pairs: CVD dE 24.7 / normal vision 33.6 / contrast above 3:1). Do not change it
   without re-validating. */
(function () {
  'use strict';

  var NFE = [2, 4, 8, 16, 32];

  var SERIES = [
    { key: 'fr_reta',    label: 'FlashRender w/ RETA (Stage 2)',      color: '#2a78d6', dash: '' },
    { key: 'ms_reta',    label: 'FlashRender-MS w/ RETA',   color: '#2a78d6', dash: '5 4' },
    { key: 'fr_noreta',  label: 'FlashRender w/o RETA (Stage 2)',     color: '#eb6834', dash: '' },
    { key: 'ms_noreta',  label: 'FlashRender-MS w/o RETA',  color: '#eb6834', dash: '5 4' }
  ];

  var GROUPS = [
    {
      title: 'Visual Quality',
      panels: [
        { name: 'Aesthetic Quality', better: 'up', dp: 4, data: {
            fr_reta:   [0.5045, 0.5140, 0.5161, 0.5160, 0.5137],
            fr_noreta: [0.4875, 0.5071, 0.5143, 0.5149, 0.5115],
            ms_reta:   [0.4650, 0.4945, 0.5109, 0.5133, 0.5140],
            ms_noreta: [0.4572, 0.4953, 0.5093, 0.5103, 0.5110] } },
        { name: 'Imaging Quality', better: 'up', dp: 4, data: {
            fr_reta:   [0.6472, 0.6580, 0.6623, 0.6607, 0.6642],
            fr_noreta: [0.6100, 0.6474, 0.6588, 0.6605, 0.6629],
            ms_reta:   [0.5713, 0.6370, 0.6494, 0.6589, 0.6640],
            ms_noreta: [0.5334, 0.6097, 0.6327, 0.6449, 0.6510] } }
      ]
    },
    {
      title: 'Geometric Consistency',
      panels: [
        { name: 'Dyn-MEt3R', better: 'up', dp: 4, data: {
            fr_reta:   [0.850279, 0.8564, 0.858738, 0.859142, 0.858231],
            fr_noreta: [0.8405, 0.8509, 0.8522, 0.8527, 0.8521],
            ms_reta:   [0.787165, 0.821520, 0.831588, 0.8361, 0.8450],
            ms_noreta: [0.7627, 0.8130, 0.8203, 0.8265, 0.8320] } },
        { name: 'MEt3R', better: 'down', dp: 4, data: {
            fr_reta:   [0.319688, 0.3119, 0.309509, 0.308572, 0.309536],
            fr_noreta: [0.3503, 0.3256, 0.3188, 0.3159, 0.3149],
            ms_reta:   [0.394280, 0.342000, 0.323370, 0.31941, 0.31150],
            ms_noreta: [0.4335, 0.3800, 0.3586, 0.3420, 0.3380] } }
      ]
    },
    {
      title: 'Camera Accuracy',
      panels: [
        { name: 'TransErr', better: 'down', dp: 4, data: {
            fr_reta:   [0.016760, 0.0143, 0.014860, 0.015420, 0.016020],
            fr_noreta: [0.0179, 0.0176, 0.0174, 0.0180, 0.0182],
            ms_reta:   [0.025160, 0.018860, 0.016360, 0.01607, 0.01490],
            ms_noreta: [0.0300, 0.0205, 0.0188, 0.0165, 0.0152] } },
        { name: 'RotErr', better: 'down', dp: 3, data: {
            fr_reta:   [1.330200, 1.311, 1.371340, 1.416600, 1.434640],
            fr_noreta: [1.62, 1.671, 1.620, 1.55, 1.560],
            ms_reta:   [1.696820, 1.630840, 1.570, 1.490, 1.480],
            ms_noreta: [1.820, 1.760, 1.670, 1.620, 1.600] } }
      ]
    }
  ];

  /* ---- Curvature curves ---------------------------------------------------
     Five series, so slots 1-5 of the documented palette in order. Three of them fall
     under 3:1 against white, which makes the validator demand relief - hence the names
     labelled directly at the end of each line (a legend alone is not relief). */
  var CURV = {"sigmas":[0.09259,0.17241,0.24194,0.30303,0.35714,0.40541,0.44872,0.4878,0.52326,0.55556,0.58511,0.61224,0.63725,0.66038,0.68182,0.70175,0.72034,0.7377,0.75397,0.76923,0.78358,0.7971,0.80986,0.82192,0.83333,0.84416,0.85443,0.8642,0.87349,0.88235,0.8908,0.89888,0.90659,0.91398,0.92105,0.92784,0.93434,0.94059,0.9466,0.95238,0.95794,0.9633,0.96847,0.97345,0.97826,0.98291,0.98739,0.99174,0.99593,1.0],"series":{"ours":[0.081294,0.029248,0.016163,0.011642,0.009493,0.00844,0.007944,0.007858,0.007911,0.0078,0.008172,0.008406,0.00863,0.008921,0.0093,0.009632,0.009948,0.010257,0.010644,0.010965,0.011836,0.012012,0.012418,0.013066,0.013519,0.014177,0.014643,0.015413,0.016188,0.017058,0.017607,0.018307,0.019216,0.02001,0.021002,0.021577,0.023075,0.02423,0.025436,0.027461,0.029572,0.032374,0.036193,0.038116,0.041312,0.045097,0.049513,0.054428,0.070544,0.090671],"noreta":[0.103125,0.037958,0.022085,0.016242,0.013048,0.011528,0.010622,0.010627,0.010493,0.010322,0.010715,0.011268,0.011518,0.012299,0.01307,0.013781,0.014435,0.015024,0.015992,0.017114,0.017774,0.019015,0.020023,0.021265,0.022422,0.023743,0.025041,0.026956,0.028535,0.030533,0.032595,0.033876,0.035655,0.036647,0.038506,0.039778,0.041353,0.042064,0.043942,0.045417,0.047368,0.051245,0.053177,0.05617,0.060741,0.066046,0.069938,0.076768,0.091988,0.113121],"recam":[0.097864,0.037186,0.021223,0.01505,0.011997,0.010256,0.009508,0.009245,0.009232,0.009299,0.009823,0.010222,0.010879,0.011405,0.012086,0.012714,0.013708,0.014394,0.015328,0.016408,0.017143,0.017903,0.019031,0.020186,0.021103,0.02249,0.023705,0.025292,0.027304,0.028807,0.030453,0.032508,0.034905,0.037508,0.039837,0.042269,0.045436,0.047011,0.0501,0.052558,0.054309,0.057481,0.059933,0.062941,0.066004,0.069454,0.075993,0.082432,0.09349,0.094419],"redirector":[0.096959,0.035194,0.020444,0.014432,0.011672,0.010406,0.009849,0.009712,0.009844,0.010052,0.010379,0.01103,0.0117,0.012419,0.013125,0.013864,0.014325,0.014972,0.015547,0.016328,0.017278,0.01827,0.01957,0.020525,0.021789,0.023216,0.024799,0.025674,0.027198,0.027851,0.029256,0.030568,0.031879,0.034284,0.035575,0.03623,0.037379,0.038355,0.039418,0.041396,0.044774,0.050737,0.057625,0.065174,0.065231,0.064323,0.074611,0.100178,0.123762,0.107281],"geoalign":[0.092857,0.034358,0.019896,0.014114,0.011452,0.010126,0.009612,0.009456,0.009552,0.009806,0.010104,0.010542,0.011245,0.011895,0.012508,0.013214,0.013817,0.014502,0.015207,0.016094,0.017127,0.018033,0.019245,0.020213,0.021098,0.022153,0.023748,0.02473,0.02567,0.026673,0.02797,0.02937,0.030586,0.032518,0.033907,0.035424,0.037064,0.038565,0.041023,0.041643,0.045197,0.050308,0.057642,0.064768,0.064239,0.06429,0.071313,0.093414,0.11761,0.10457]}};

  var CURV_SERIES = [
    { key: 'ours',       label: 'FlashRender-MS (w/ RETA)',  color: '#2a78d6' },
    { key: 'noreta',     label: 'FlashRender-MS (w/o RETA)', color: '#eb6834' },
    { key: 'recam',      label: 'ReCamMaster',               color: '#1baf7a' },
    { key: 'redirector', label: 'ReDirector',                color: '#eda100' },
    { key: 'geoalign',   label: 'GeoAlign',                  color: '#e87ba4' }
  ];

  var W = 340, H = 210;
  var M = { top: 12, right: 14, bottom: 40, left: 48 };
  var PW = W - M.left - M.right;
  var PH = H - M.top - M.bottom;
  var NS = 'http://www.w3.org/2000/svg';

  function el(name, attrs) {
    var n = document.createElementNS(NS, name);
    for (var k in attrs) if (attrs[k] !== undefined) n.setAttribute(k, attrs[k]);
    return n;
  }
  function fmt(v, dp) { return v.toFixed(dp); }

  /* The tooltip follows the cursor inside the plot. Floating it above would overflow the
     card and get clipped; leaving it parked loses track of which point is being read.
     Near an edge it folds to the opposite side. */
  function placeTip(tip, wrap, clientX, clientY) {
    var box = wrap.getBoundingClientRect();
    var tw = tip.offsetWidth, th = tip.offsetHeight;
    var x = (clientX === undefined ? box.left + box.width / 2 : clientX) - box.left;
    var y = (clientY === undefined ? box.top + 12 : clientY) - box.top;

    var left = x + 16;
    if (left + tw > box.width - 4) left = x - tw - 16;
    left = Math.max(4, Math.min(box.width - tw - 4, left));

    var top = y - th - 12;
    if (top < 4) top = y + 18;
    top = Math.max(4, Math.min(box.height - th - 4, top));

    tip.style.left = left + 'px';
    tip.style.top = top + 'px';
  }

  /* Four tick intervals, with 8% of headroom around the value range. */
  function scaleY(panel) {
    var lo = Infinity, hi = -Infinity;
    SERIES.forEach(function (s) {
      panel.data[s.key].forEach(function (v) { lo = Math.min(lo, v); hi = Math.max(hi, v); });
    });
    var pad = (hi - lo) * 0.08 || Math.abs(hi) * 0.05 || 1;
    lo -= pad; hi += pad;
    return {
      lo: lo, hi: hi,
      y: function (v) { return M.top + PH - (v - lo) / (hi - lo) * PH; },
      ticks: [0, 1, 2, 3].map(function (i) { return lo + (hi - lo) * i / 3; })
    };
  }
  /* x is log scaled. NFE doubles each step, so log2 positions come out evenly spaced. */
  var LX = NFE.map(function (n) { return Math.log2(n); });
  var LX0 = LX[0], LX1 = LX[LX.length - 1];
  function xAt(i) { return M.left + (LX[i] - LX0) / (LX1 - LX0) * PW; }

  function buildPanel(panel, showX) {
    var fig = document.createElement('figure');
    fig.className = 'chart';

    var cap = document.createElement('figcaption');
    cap.innerHTML = '<span class="chart-name">' + panel.name + '</span>' +
      '<span class="chart-dir">' + (panel.better === 'up' ? 'higher is better' : 'lower is better') + '</span>';
    fig.appendChild(cap);

    var wrap = document.createElement('div');
    wrap.className = 'chart-plot';

    var svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img',
                          'aria-label': panel.name + ' against the number of function evaluations' });
    var sc = scaleY(panel);

    // Grid and axes sit one step above the background, as solid hairlines.
    sc.ticks.forEach(function (t) {
      var y = sc.y(t);
      svg.appendChild(el('line', { x1: M.left, x2: M.left + PW, y1: y, y2: y,
                                   stroke: '#E3ECEB', 'stroke-width': 1 }));
      var lab = el('text', { x: M.left - 8, y: y + 3.5, 'text-anchor': 'end', class: 'chart-tick' });
      lab.textContent = fmt(t, panel.dp);
      svg.appendChild(lab);
    });

    NFE.forEach(function (n, i) {
      var lab = el('text', { x: xAt(i), y: M.top + PH + 18, 'text-anchor': 'middle', class: 'chart-tick' });
      lab.textContent = n;
      svg.appendChild(lab);
    });
    svg.appendChild(el('line', { x1: M.left, x2: M.left + PW, y1: M.top + PH, y2: M.top + PH,
                                 stroke: '#CBDAD8', 'stroke-width': 1 }));
    if (showX) {
      var xt = el('text', { x: M.left + PW / 2, y: M.top + PH + 31, 'text-anchor': 'middle', class: 'chart-axis' });
      xt.textContent = 'sampling steps (NFE, log scale)';
      svg.appendChild(xt);
    }

    SERIES.forEach(function (s) {
      var vals = panel.data[s.key];
      var d = vals.map(function (v, i) { return (i ? 'L' : 'M') + xAt(i) + ' ' + sc.y(v); }).join(' ');
      svg.appendChild(el('path', { d: d, fill: 'none', stroke: s.color, 'stroke-width': 2,
                                   'stroke-linejoin': 'round', 'stroke-linecap': 'round',
                                   'stroke-dasharray': s.dash || undefined }));
    });

    // Points go on top of the lines; a 2px white ring separates overlapping ones.
    SERIES.forEach(function (s) {
      panel.data[s.key].forEach(function (v, i) {
        svg.appendChild(el('circle', { cx: xAt(i), cy: sc.y(v), r: 4, fill: s.color,
                                       stroke: '#fff', 'stroke-width': 2 }));
      });
    });

    wrap.appendChild(svg);
    fig.appendChild(wrap);

    return fig;
  }

  /* The curvature panel. Unlike the small multiples it takes one wide figure: five
     50-point curves overlap here, and nothing separates at a narrow width. */
  function buildCurvature() {
    var CW = 900, CH = 330;
    var CM = { top: 16, right: 188, bottom: 46, left: 62 };   // the right margin holds the direct labels
    var CPW = CW - CM.left - CM.right, CPH = CH - CM.top - CM.bottom;

    var xs = CURV.sigmas;
    var lo = Infinity, hi = -Infinity;
    CURV_SERIES.forEach(function (s2) {
      CURV.series[s2.key].forEach(function (v) { lo = Math.min(lo, v); hi = Math.max(hi, v); });
    });
    // Starting at zero squashes the crowded 0.01-0.04 band until the differences vanish.
    // Pad the data range instead.
    var pad = (hi - lo) * 0.07;
    lo = Math.max(0, lo - pad);   // curvature cannot go negative; without this the ticks read -0.000
    hi += pad;

    // 0 to 1 is the natural domain for timestep, so the axis opens at 0 even though the
    // data starts at 0.09.
    var X = function (v) { return CM.left + v * CPW; };
    var Y = function (v) { return CM.top + CPH - (v - lo) / (hi - lo) * CPH; };

    var fig = document.createElement('figure');
    fig.className = 'chart chart--wide';
    var cap = document.createElement('figcaption');
    cap.innerHTML = '<span class="chart-name">Denoising trajectory curvature</span>' +
                    '<span class="chart-dir">lower is better &middot; 50-step sampling</span>';
    fig.appendChild(cap);

    var wrap = document.createElement('div');
    wrap.className = 'chart-plot';
    var svg = el('svg', { viewBox: '0 0 ' + CW + ' ' + CH, role: 'img',
                          'aria-label': 'Denoising trajectory curvature against timestep for five models' });

    [0, 1, 2, 3, 4].forEach(function (i) {
      var v = lo + (hi - lo) * i / 4, y = Y(v);
      svg.appendChild(el('line', { x1: CM.left, x2: CM.left + CPW, y1: y, y2: y,
                                   stroke: '#E3ECEB', 'stroke-width': 1 }));
      var t = el('text', { x: CM.left - 10, y: y + 3.5, 'text-anchor': 'end', class: 'chart-tick' });
      t.textContent = v.toFixed(3);
      svg.appendChild(t);
    });
    [0, 0.2, 0.4, 0.6, 0.8, 1].forEach(function (v) {
      var x = X(v);
      var t = el('text', { x: x, y: CM.top + CPH + 20, 'text-anchor': 'middle', class: 'chart-tick' });
      t.textContent = v.toFixed(1);
      svg.appendChild(t);
    });
    svg.appendChild(el('line', { x1: CM.left, x2: CM.left + CPW, y1: CM.top + CPH, y2: CM.top + CPH,
                                 stroke: '#CBDAD8', 'stroke-width': 1 }));
    var xt = el('text', { x: CM.left + CPW / 2, y: CM.top + CPH + 38, 'text-anchor': 'middle', class: 'chart-axis' });
    xt.textContent = 'timestep';
    svg.appendChild(xt);

    var cross = el('line', { y1: CM.top, y2: CM.top + CPH, stroke: '#9FB3B0',
                             'stroke-width': 1, opacity: 0 });
    svg.appendChild(cross);

    CURV_SERIES.forEach(function (s2) {
      var vals = CURV.series[s2.key];
      var d = vals.map(function (v, i) { return (i ? 'L' : 'M') + X(xs[i]).toFixed(1) + ' ' + Y(v).toFixed(1); }).join(' ');
      svg.appendChild(el('path', { d: d, fill: 'none', stroke: s2.color, 'stroke-width': 2,
                                   'stroke-linejoin': 'round', 'stroke-linecap': 'round' }));
    });

    /* Direct labels at the right edge, pushed down when they collide - the five values
       crowd into a 0.02 band and would otherwise stack on top of each other. */
    var last = CURV_SERIES.map(function (s2) {
      var vals = CURV.series[s2.key];
      return { s: s2, y: Y(vals[vals.length - 1]), v: vals[vals.length - 1] };
    }).sort(function (a, b) { return a.y - b.y; });
    var GAP = 17;
    for (var i = 1; i < last.length; i++) {
      if (last[i].y - last[i - 1].y < GAP) last[i].y = last[i - 1].y + GAP;
    }
    /* Those right-hand names are the legend. On hover they travel to the value at that
       point - which ties line to name far more directly than a separate tooltip box. */
    var RIGHT = CM.left + CPW;
    var marks = CURV_SERIES.map(function (s2) {
      var g = el('g', {});
      var leader = el('path', { stroke: s2.color, 'stroke-width': 1, fill: 'none', opacity: 0.5 });
      var name = el('text', { x: RIGHT + 15, class: 'chart-endlabel', fill: s2.color });
      name.textContent = s2.label;
      var dot = el('circle', { r: 4.5, fill: s2.color, stroke: '#fff', 'stroke-width': 2, opacity: 0 });
      g.appendChild(leader); g.appendChild(dot); g.appendChild(name);
      svg.appendChild(g);
      return { s: s2, leader: leader, name: name, dot: dot };
    });

    /* When the five values crowd together the labels overlap, so sort by value and push down. */
    function layout(i, hovering) {
      var gap = 17;
      var rows = marks.map(function (m) {
        var v = CURV.series[m.s.key][i];
        return { m: m, v: v, y: Y(v) };
      }).sort(function (a, c) { return a.y - c.y; });

      for (var k = 1; k < rows.length; k++) {
        if (rows[k].y - rows[k - 1].y < gap) rows[k].y = rows[k - 1].y + gap;
      }
      // Pushing down alone runs the last one off the axis; if it overflows, shift the set back up.
      var over = rows[rows.length - 1].y - (CM.top + CPH - 4);
      if (over > 0) rows.forEach(function (r) { r.y -= over; });

      var px = X(xs[i]);
      rows.forEach(function (r) {
        var py = Y(r.v);
        r.m.leader.setAttribute('d', 'M' + px.toFixed(1) + ' ' + py.toFixed(1) +
                                     ' L' + (RIGHT + 11) + ' ' + r.y.toFixed(1));
        r.m.name.setAttribute('y', (r.y + 4).toFixed(1));
        r.m.dot.setAttribute('cx', px);
        r.m.dot.setAttribute('cy', py);
        r.m.dot.setAttribute('opacity', hovering ? 1 : 0);
      });
    }

    var LAST = xs.length - 1;
    layout(LAST, false);

    var cross = el('line', { y1: CM.top, y2: CM.top + CPH, stroke: '#9FB3B0',
                             'stroke-width': 1, opacity: 0 });
    svg.insertBefore(cross, svg.firstChild.nextSibling);

    var hit = el('rect', { x: CM.left, y: CM.top, width: CPW, height: CPH,
                           fill: 'transparent', style: 'cursor:crosshair' });
    svg.appendChild(hit);
    wrap.appendChild(svg);
    fig.appendChild(wrap);

    function show(i) {
      cross.setAttribute('x1', X(xs[i]));
      cross.setAttribute('x2', X(xs[i]));
      cross.setAttribute('opacity', 1);
      layout(i, true);
    }
    function hide() {
      cross.setAttribute('opacity', 0);
      layout(LAST, false);
    }
    function nearest(clientX) {
      var box = svg.getBoundingClientRect();
      var vx = (clientX - box.left) / box.width * CW;
      var best = 0, bd = Infinity;
      xs.forEach(function (v, i) { var d = Math.abs(X(v) - vx); if (d < bd) { bd = d; best = i; } });
      return best;
    }
    hit.addEventListener('pointermove', function (e) { show(nearest(e.clientX)); });
    hit.addEventListener('pointerleave', hide);

    var cursor = LAST;
    fig.tabIndex = 0;
    fig.addEventListener('focus', function () { show(cursor); });
    fig.addEventListener('blur', hide);
    fig.addEventListener('keydown', function (e) {
      if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
      e.preventDefault();
      cursor = Math.max(0, Math.min(LAST, cursor + (e.key === 'ArrowRight' ? 1 : -1)));
      show(cursor);
    });

    return fig;
  }

  function build(root) {
    var legend = document.createElement('div');
    legend.className = 'chart-legend';
    legend.innerHTML = SERIES.map(function (s) {
      return '<span class="chart-legend-item"><span class="chart-swatch' + (s.dash ? ' is-dashed' : '') +
        '" style="--c:' + s.color + '"></span>' + s.label + '</span>';
    }).join('');
    root.appendChild(legend);

    var grid = document.createElement('div');
    grid.className = 'chart-grid';
    GROUPS.forEach(function (g) {
      var col = document.createElement('div');
      col.className = 'chart-col';
      g.panels.forEach(function (p, i) { col.appendChild(buildPanel(p, i === g.panels.length - 1)); });
      grid.appendChild(col);
    });
    root.appendChild(grid);
  }

  /* ------------------------------------------------------------ runtime -- */

  /* Runtime, pulled out of the table into its own view. The values span 380x, from 0.2 to
     75.9 minutes, so on a linear axis only Vista4D is visible and everything else collapses.
     Hence the log axis. Note: bar length on a log axis is not proportional to the value
     (75.9 is 380x of 0.2 but its bar looks about 3x). That is why every bar carries its
     number at the end and the caption says log scale. */
  var RUNTIME = [
    { name: 'Vista4D',           v: 75.9 },
    { name: 'ReCamMaster',       v: 6.5 },
    { name: 'ReDirector',        v: 6.5 },
    { name: 'GeoAlign',          v: 6.5 },
    { name: 'GCD',               v: 2.4 },
    { name: 'CogNVS',            v: 2.2 },
    { name: 'TrajectoryCrafter', v: 1.9 },
    { name: 'NeoVerse',          v: 0.5 },
    { name: 'FlashRender',       v: 0.2, ours: true }
  ];

  function buildRuntime() {
    var RW = 900, ROW = 30;
    var RM = { top: 14, right: 74, bottom: 44, left: 150 };
    var RH = RM.top + RM.bottom + RUNTIME.length * ROW;
    var RPW = RW - RM.left - RM.right;

    var TICKS = [0.1, 1, 10, 100];
    var L0 = Math.log10(TICKS[0]), L1 = Math.log10(TICKS[TICKS.length - 1]);
    var X = function (v) { return RM.left + (Math.log10(v) - L0) / (L1 - L0) * RPW; };
    var rowY = function (i) { return RM.top + i * ROW + ROW / 2; };

    var fig = document.createElement('figure');
    fig.className = 'chart chart--wide chart--runtime';
    var cap = document.createElement('figcaption');
    cap.innerHTML = '<span class="chart-name">Runtime per video</span>' +
                    '<span class="chart-dir">lower is better &middot; minutes (log scale)</span>';
    fig.appendChild(cap);

    var wrap = document.createElement('div');
    wrap.className = 'chart-plot';
    var svg = el('svg', { viewBox: '0 0 ' + RW + ' ' + RH, role: 'img',
                          'aria-label': 'Runtime per video in minutes for nine methods, log scale' });

    TICKS.forEach(function (t) {
      var x = X(t);
      svg.appendChild(el('line', { x1: x, x2: x, y1: RM.top, y2: RM.top + RUNTIME.length * ROW,
                                   stroke: '#E3ECEB', 'stroke-width': 1 }));
      var lb = el('text', { x: x, y: RM.top + RUNTIME.length * ROW + 20,
                            'text-anchor': 'middle', class: 'chart-tick' });
      lb.textContent = t < 1 ? String(t) : String(t);
      svg.appendChild(lb);
    });
    var xt = el('text', { x: RM.left + RPW / 2, y: RM.top + RUNTIME.length * ROW + 38,
                          'text-anchor': 'middle', class: 'chart-axis' });
    xt.textContent = 'minutes per video';
    svg.appendChild(xt);

    var BAR = 17;                       // bar thickness; rows are 30 apart, leaving gaps above and below
    RUNTIME.forEach(function (m, i) {
      var y = rowY(i), x = X(m.v);
      var color = m.ours ? '#07BEB8' : '#8FA7A4';

      var nm = el('text', { x: RM.left - 12, y: y + 4, 'text-anchor': 'end',
                            class: 'rt-name' + (m.ours ? ' is-ours' : '') });
      nm.textContent = m.name;
      svg.appendChild(nm);

      svg.appendChild(el('rect', { x: RM.left, y: y - BAR / 2, width: Math.max(2, x - RM.left),
                                   height: BAR, rx: 4, fill: color }));

      var val = el('text', { x: x + 10, y: y + 4, class: 'rt-value' + (m.ours ? ' is-ours' : '') });
      val.textContent = m.v.toFixed(1);
      svg.appendChild(val);
    });

    wrap.appendChild(svg);
    fig.appendChild(wrap);
    return fig;
  }

  var root = document.querySelector('[data-charts]');
  if (root) build(root);

  var curvRoot = document.querySelector('[data-curvature]');
  if (curvRoot) curvRoot.appendChild(buildCurvature());

  var rtRoot = document.querySelector('[data-runtime]');
  if (rtRoot) rtRoot.appendChild(buildRuntime());
})();
