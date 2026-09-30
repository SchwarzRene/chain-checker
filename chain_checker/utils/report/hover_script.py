# Appended to a summary report's body - the epoch report never embeds this
# since it has no line/area charts to hover.
HOVER_SCRIPT = """<script>
(function () {
  function init() {
    document.querySelectorAll('script[data-chart-points]').forEach(function (dataEl) {
      var chartId = dataEl.getAttribute('data-chart-points');
      var svg = document.getElementById(chartId);
      if (!svg) return;

      var data;
      try {
        data = JSON.parse(dataEl.textContent);
      } catch (e) {
        return;
      }
      if (!data.xs || !data.xs.length) return;

      var card = svg.closest('.card');
      if (!card) return;

      var svgNS = 'http://www.w3.org/2000/svg';
      var guide = document.createElementNS(svgNS, 'line');
      guide.setAttribute('class', 'chart-crosshair');
      guide.setAttribute('y1', data.top);
      guide.setAttribute('y2', data.bottom);
      svg.appendChild(guide);

      var tooltip = document.createElement('div');
      tooltip.className = 'chart-tooltip';
      card.appendChild(tooltip);

      function nearestIndex(svgX) {
        var xs = data.xs, best = 0, bestDist = Infinity;
        for (var i = 0; i < xs.length; i++) {
          var d = Math.abs(xs[i] - svgX);
          if (d < bestDist) { bestDist = d; best = i; }
        }
        return best;
      }

      function showAt(idx) {
        var x = data.xs[idx];
        guide.setAttribute('x1', x);
        guide.setAttribute('x2', x);
        guide.style.display = '';

        var entry = data.tooltips[idx];
        tooltip.textContent = '';

        var title = document.createElement('div');
        title.className = 'chart-tooltip-title';
        title.textContent = entry.title;
        tooltip.appendChild(title);

        entry.rows.forEach(function (row) {
          var rowEl = document.createElement('div');
          rowEl.className = 'chart-tooltip-row' + (row.strong ? ' chart-tooltip-row-strong' : '');
          if (row.color) {
            var swatch = document.createElement('span');
            swatch.className = 'chart-tooltip-swatch';
            swatch.style.background = row.color;
            rowEl.appendChild(swatch);
          }
          rowEl.appendChild(document.createTextNode(row.label + ': '));
          var valueEl = document.createElement('span');
          valueEl.className = 'chart-tooltip-value';
          valueEl.textContent = row.value;
          rowEl.appendChild(valueEl);
          tooltip.appendChild(rowEl);
        });

        tooltip.style.display = 'block';

        var rect = svg.getBoundingClientRect();
        var cardRect = card.getBoundingClientRect();
        var vb = svg.viewBox.baseVal;
        if (!rect.width || !vb || !vb.width) return;

        var pointClientX = rect.left + (x / vb.width) * rect.width;
        var left = pointClientX - cardRect.left;
        var tw = tooltip.offsetWidth;
        left = Math.max(4, Math.min(left, cardRect.width - tw - 4));
        tooltip.style.left = left + 'px';

        var topClientY = rect.top + (data.top / vb.height) * rect.height;
        var top = topClientY - cardRect.top - tooltip.offsetHeight - 10;
        if (top < 4) {
          var bottomClientY = rect.top + (data.bottom / vb.height) * rect.height;
          top = bottomClientY - cardRect.top + 10;
        }
        tooltip.style.top = top + 'px';
      }

      function hide() {
        guide.style.display = 'none';
        tooltip.style.display = 'none';
      }

      svg.addEventListener('mousemove', function (evt) {
        var rect = svg.getBoundingClientRect();
        var vb = svg.viewBox.baseVal;
        if (!rect.width || !vb || !vb.width) return;
        var svgX = (evt.clientX - rect.left) / rect.width * vb.width;
        showAt(nearestIndex(svgX));
      });
      svg.addEventListener('mouseleave', hide);

      hide();
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
</script>"""
