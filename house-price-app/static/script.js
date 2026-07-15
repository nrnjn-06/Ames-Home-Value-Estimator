const form = document.getElementById('form');
const priceEl = document.getElementById('price');
const rangeEl = document.getElementById('range');
const btn = document.getElementById('submit-btn');

const qualSlider = document.getElementById('OverallQual');
const qualOut = document.getElementById('OverallQual-out');

function syncSlider() {
  const min = +qualSlider.min;
  const max = +qualSlider.max;
  const val = +qualSlider.value;
  const pct = ((val - min) / (max - min)) * 100;
  qualSlider.style.setProperty('--fill', pct + '%');
  qualOut.textContent = val;
}
syncSlider();
qualSlider.addEventListener('input', syncSlider);

const fmt = (n) => '$' + Math.round(n).toLocaleString('en-US');

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  btn.disabled = true;
  btn.textContent = 'Estimating…';
  priceEl.style.opacity = '0.4';

  const payload = {
    OverallQual: qualSlider.value,
    GrLivArea: document.getElementById('GrLivArea').value,
    TotalBsmtSF: document.getElementById('TotalBsmtSF').value,
    LotArea: document.getElementById('LotArea').value,
    GarageCars: document.getElementById('GarageCars').value,
    FullBath: document.getElementById('FullBath').value,
    TotRmsAbvGrd: document.getElementById('TotRmsAbvGrd').value,
    YearBuilt: document.getElementById('YearBuilt').value,
    YearRemodAdd: document.getElementById('YearRemodAdd').value,
    Neighborhood: document.getElementById('Neighborhood').value,
  };

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (!res.ok) {
      priceEl.textContent = '$—';
      rangeEl.textContent = data.error || 'Something went wrong. Check your inputs.';
    } else {
      priceEl.textContent = fmt(data.prediction);
      rangeEl.textContent = `Likely range: ${fmt(data.low)} to ${fmt(data.high)}, based on the model's typical error.`;
    }
  } catch (err) {
    priceEl.textContent = '$—';
    rangeEl.textContent = 'Could not reach the server. Make sure app.py is running.';
  } finally {
    priceEl.style.opacity = '1';
    btn.disabled = false;
    btn.textContent = 'Estimate price';
  }
});
