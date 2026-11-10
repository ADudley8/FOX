import numpy as np
from axion_haloscope.simulation import simulate_spectra, AxionParams

def test_simulation_shapes_and_seed():
    spectrum_1 = simulate_spectra(
        n_spectra=4, n_bins=2000, bin_width_hz=100.0,
        f_start_hz=5.70e9, tune_step_bins=80,
        noise_sigma=1.0, rng_seed=1234, injected_axion=None
    )
    spectrum_2 = simulate_spectra(
        n_spectra=4, n_bins=2000, bin_width_hz=100.0,
        f_start_hz=5.70e9, tune_step_bins=80,
        noise_sigma=1.0, rng_seed=1234, injected_axion=None
    )

    assert len(spectrum_1.spectra) == 4 and spectrum_1.spectra[0].shape == (2000,)
    assert spectrum_1.rf_grid.ndim == 1 and spectrum_1.rf_grid.size == 2000 + 3*80  # global grid size
    # deterministic with same seed
    for a,b in zip(spectrum_1.spectra, spectrum_2.spectra):
        assert np.allclose(a, b)
    # RF bookkeeping consistent
    for i, idx in enumerate(spectrum_1.rf_index_map):
        assert np.allclose(spectrum_1.freqs_per_spec[i], spectrum_1.rf_grid[idx])



def test_axion_injection_produces_local_excess():
    n_spectra, n_bins, bw = 6, 4000, 100.0
    f_start, step = 5.70e9, 80
    # Put axion roughly mid-span
    rf_total_bins = n_bins + (n_spectra-1)*step
    f_ax = f_start + 0.5*rf_total_bins*bw

    spectrum = simulate_spectra(
        n_spectra=n_spectra, n_bins=n_bins, bin_width_hz=bw,
        f_start_hz=f_start, tune_step_bins=step, rng_seed=7,
        injected_axion=AxionParams(f_axion_hz=f_ax, sigma_hz=2500.0, total_power=30.0)
    )

    # Naive RF average (no baseline removal), just to "see" the bump
    summed = np.zeros_like(spectrum.rf_grid, dtype=float)
    counts = np.zeros_like(spectrum.rf_grid, dtype=int)
    for s, idx in zip(spectrum.spectra, spectrum.rf_index_map):
        summed[idx] += s
        counts[idx] += 1
    avg = np.divide(summed, counts, where=counts>0)

    # Build an in-bounds signal window W and a background window Wa far away
    i_ax = int(np.argmin(np.abs(spectrum.rf_grid - f_ax)))
    # half-width ~ 15 kHz window -> bins:
    half_w = max(1, int(15_000.0 / bw))
    a = max(0, i_ax - half_w)
    b = min(len(spectrum.rf_grid), i_ax + half_w + 1)
    W = np.zeros_like(spectrum.rf_grid, dtype=bool); W[a:b] = True

    # background window: shift by ~30*half_w bins (far but in-bounds)
    shift = 30 * half_w
    a_bg = a + shift
    b_bg = b + shift
    if b_bg > len(spectrum.rf_grid):  # if off the right edge, move to the left instead
        a_bg = max(0, a - shift)
        b_bg = max(b - shift, a_bg + 1)
    Wa = np.zeros_like(spectrum.rf_grid, dtype=bool); Wa[a_bg:b_bg] = True

    # sanity: both windows non-empty and disjoint
    assert W.any() and Wa.any() and not np.any(W & Wa)

    assert np.mean(avg[W]) > np.mean(avg[Wa])

