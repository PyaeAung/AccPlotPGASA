# pip install obspy numpy matplotlib
from obspy import read, read_inventory
import numpy as np
import matplotlib.pyplot as plt

# -------- USER SETTINGS --------
MSEED = "RF99C.ms"
STAXML = "RF99C.xml"
SAVE_CORRECTED = True
OUT_MSEED = "RF99C_acc_g.mseed"  # saved acceleration in g

# Sa / plotting
DAMPING = 0.05
T_MIN, T_MAX = 0.02, 1.0
NPTS_SA = 400
SMOOTH_SA, SMOOTH_WIN = True, 7
# --------------------------------

g0 = 9.80665

# --- Load ---
st = read(MSEED)
inv = read_inventory(STAXML)
st.sort(keys=["channel"])

# --- Try to ensure IDs match inventory (common for Raspberry Shake) ---
# Many Shakes are AM.RF99C.00.EHZ / EHN / EHE
net_code = inv.networks[0].code
sta_code = inv.networks[0].stations[0].code

# Build a map of desired channel for each component (Z/N/E) from inventory
chan_map = {}
for ch in inv.networks[0].stations[0].channels:
    comp = ch.code[-1]
    chan_map[comp] = (ch.code, ch.location_code)

for tr in st:
    comp = tr.stats.channel[-1] if tr.stats.channel else "Z"
    # replacement to inventory IDs if needed
    if not inv.select(network=tr.stats.network, station=tr.stats.station,
                      location=tr.stats.location, channel=tr.stats.channel):
        # fallback to inventory NET/STA and matching component code
        ch_code, loc_code = chan_map.get(comp, (tr.stats.channel, tr.stats.location))
        tr.stats.network  = net_code
        tr.stats.station  = sta_code
        tr.stats.location = loc_code
        tr.stats.channel  = ch_code

# --- Choose a sensible pre-filter for RShake geophone (4.5 Hz nat. freq) ---
fs = float(st[0].stats.sampling_rate)
nyq = 0.5 * fs
pre_filt = (0.7, 1.0, 0.8 * nyq, 0.95 * nyq)  # (f1, f2, f3, f4) in Hz

# --- Remove response -> ACC (m/s^2) ---
st.attach_response(inv)
st_acc = st.copy()
st_acc.remove_response(inventory=inv, output="ACC",
                       pre_filt=pre_filt, water_level=60,
                       taper=True, taper_fraction=0.05)

# Convert to g (for plotting & saving)
for tr in st_acc:
    tr.data = tr.data / g0
unit = "g"

# Optionally save corrected MiniSEED in g
if SAVE_CORRECTED:
    st_acc.write(OUT_MSEED, format="MSEED")
    print(f"Saved: {OUT_MSEED} (acceleration in g)")

# -------------- Sa computation (Newmark-β, 5% damp) --------------
def response_spectrum(acc, dt, periods, zeta=0.05):
    beta, gamma = 1/4, 1/2
    a = acc; Sa = np.zeros_like(periods, float)
    for i, T in enumerate(periods):
        if T <= 0: continue
        wn = 2*np.pi / T; k = wn*wn; c = 2*zeta*wn
        a0 = 1.0/(beta*dt*dt); a1 = gamma/(beta*dt); a2 = 1.0/(beta*dt)
        a3 = 1.0/(2*beta) - 1.0; a4 = gamma/beta - 1.0; a5 = dt*(gamma/(2*beta) - 1.0)
        keff = k + a0 + c*a1
        u = ud = udd = 0.0; amax = 0.0
        for n in range(a.size):
            peff = (-a[n]) + (a0*u + a2*ud + a3*udd) + c*(a1*u + a4*ud + a5*udd)
            u_new = peff/keff
            ud_new = a1*(u_new - u) - a4*ud - a5*udd
            udd_new = a0*(u_new - u) - a2*ud - a3*udd
            aa = abs(udd_new + a[n])
            if aa > amax: amax = aa
            u, ud, udd = u_new, ud_new, udd_new
        Sa[i] = amax
    return Sa

# Period grid
periods = np.geomspace(T_MIN, T_MAX, NPTS_SA)

# Component labels & order
def comp_label(ch):
    ch = ch.upper()
    if ch.endswith("E"): return "EW"
    if ch.endswith("N"): return "NS"
    return "UD"

labels = [comp_label(tr.stats.channel) for tr in st_acc]
priority = {"EW":0, "NS":1, "UD":2}
order = np.argsort([priority.get(l, 9) for l in labels])
trs  = [st_acc[i] for i in order]
lbls = [labels[i] for i in order]

# Sa for each trace
Sa_list = [response_spectrum(tr.data.astype(np.float64), tr.stats.delta, periods, DAMPING)
           for tr in trs]

# Optional smoothing
def smooth_ma(y, win):
    if not SMOOTH_SA or win <= 1: return y
    if win % 2 == 0: win += 1
    pad = win//2
    ypad = np.pad(y, (pad, pad), mode="edge")
    ker = np.ones(win)/win
    return np.convolve(ypad, ker, mode="valid")
Sa_list = [smooth_ma(Sa, SMOOTH_WIN) for Sa in Sa_list]

# PGA
PGA = [float(np.max(np.abs(tr.data))) for tr in trs]

# -------------- Plot --------------
comp_colors = {"EW":"purple", "NS":"orange", "UD":"black"}
fig = plt.figure(figsize=(14, 6))

# Sa (left)
ax1 = plt.subplot(1, 2, 1)
for Sa, lab in zip(Sa_list, lbls):
    col = comp_colors.get(lab, "blue")
    ax1.plot(periods, Sa, lw=2.0, color=col, label=lab)
    i = int(np.argmax(Sa))
    ax1.plot(periods[i], Sa[i], "o", ms=4, color=col)
    ax1.annotate(f"{Sa[i]:.2f} {unit}", (periods[i], Sa[i]),
                 xytext=(6,6), textcoords="offset points", color=col, fontsize=8)
ax1.set_xscale("linear"); ax1.set_xlim(0.0, 1.0)
ax1.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax1.set_xlabel("T (s)"); ax1.set_ylabel(f"Sa ({unit})")
ax1.grid(True, which="both", alpha=0.35)
ax1.legend(title=MSEED)

# Waveforms (right) — 1/2/3 components ok
n = len(trs)
for i,(tr,lab,pga) in enumerate(zip(trs, lbls, PGA), start=1):
    ax = plt.subplot(3 if n==3 else (2 if n==2 else 1), 2, 2*i)
    t = np.arange(tr.stats.npts) * tr.stats.delta
    col = comp_colors.get(lab, "blue")
    ax.plot(t, tr.data, lw=1.0, color=col)
    ax.set_xlim(0, t[-1]); ax.grid(True, alpha=0.3)
    ax.set_title(lab); ax.set_ylabel(f"Acc ({unit})")
    ax.text(0.01, 0.95, f"PGA = {pga:.4f} {unit}",
            transform=ax.transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, lw=0),
            color=col, fontsize=9)
    if i < n: ax.set_xticklabels([])
    else:     ax.set_xlabel("t (s)")

fig.suptitle("RF99C — Acceleration (g) & 5% Response Spectra", y=0.98)
plt.tight_layout(); plt.show()