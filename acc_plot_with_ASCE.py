# pip install obspy numpy matplotlib
from obspy import read
import numpy as np
import matplotlib.pyplot as plt

# ======== USER INPUTS ========
MSEED = "RF99C_acc_g.mseed"   # already acceleration in g
# Choose ONE mode: A or B
USE_MODE_A = True  # True -> give SDS/SD1/TL directly; False -> give Ss/S1/Fa/Fv

# --- Mode A: direct design params (in g) ---
SDS = 0.60   # <-- fill your value (g)
SD1 = 0.30   # <-- fill your value (g)
TL  = 8.0    # long-period transition (s), city-specific

# --- Mode B: maps + site coefficients (all in g) ---
Ss = 1.00    # short-period mapped MCER
S1 = 0.40    # 1s-period mapped MCER
Fa = 1.0     # site coefficient at Ss
Fv = 1.0     # site coefficient at S1
# =============================

# --- Read data (already in g) ---
st = read(MSEED); st.sort(keys=["channel"])

# helpers
def cosine_taper(n, p=0.05):
    p = np.clip(p, 0, 0.5); w = np.ones(n); m = int(p*n)
    if m>0:
        x = np.linspace(0, np.pi/2, m); w[:m]=np.sin(x)**2; w[-m:]=w[:m][::-1]
    return w

def comp_label(ch):
    ch = ch.upper()
    if ch.endswith("E") or ch[-2:-1]=="E": return "EW"
    if ch.endswith("N") or ch[-2:-1]=="N": return "NS"
    return "UD"

# preprocess (light tidy; units already g)
for tr in st:
    tr.data = tr.data.astype(np.float64)
    tr.detrend("linear"); tr.detrend("demean")
    tr.data *= cosine_taper(tr.stats.npts, 0.05)
    tr.filter("bandpass", freqmin=0.10, freqmax=25.0, corners=4, zerophase=True)

labels = [comp_label(tr.stats.channel) for tr in st]
order = np.argsort([{"EW":0,"NS":1,"UD":2}.get(l,9) for l in labels])
trs = [st[i] for i in order]; lbls = [labels[i] for i in order]

# ---- Newmark-β Sa (5%) in g ----
def response_spectrum_g(acc_g, dt, periods, zeta=0.05):
    beta, gamma = 1/4, 1/2
    a = acc_g; Sa = np.zeros_like(periods, float)
    for i,T in enumerate(periods):
        if T<=0: continue
        wn = 2*np.pi/T; k=wn*wn; c=2*zeta*wn
        a0 = 1.0/(beta*dt*dt); a1=gamma/(beta*dt); a2=1.0/(beta*dt)
        a3 = 1.0/(2*beta)-1.0; a4=gamma/beta-1.0; a5=dt*(gamma/(2*beta)-1.0)
        keff = k + a0 + c*a1
        u=ud=udd=0.0; amax=0.0
        for n in range(a.size):
            peff = (-a[n]) + (a0*u + a2*ud + a3*udd) + c*(a1*u + a4*ud + a5*udd)
            u_new = peff/keff
            ud_new= a1*(u_new-u) - a4*ud - a5*udd
            udd_new= a0*(u_new-u) - a2*ud - a3*udd
            aa = abs(udd_new + a[n])
            if aa>amax: amax=aa
            u,ud,udd = u_new,ud_new,udd_new
        Sa[i]=amax
    return Sa

# period grid
T_MIN, T_MAX = 0.02, 1.0
periods = np.geomspace(T_MIN, T_MAX, 400)
Sa_list = [response_spectrum_g(tr.data, tr.stats.delta, periods, 0.05) for tr in trs]

# ---- ASCE 7-16 design spectrum (5% damping, in g) ----
def asce716_design_spectrum(periods, SDS, SD1, TL):
    Ts = SD1 / SDS
    T0 = 0.2 * Ts
    Sa = np.zeros_like(periods, float)
    for i, T in enumerate(periods):
        if T <= T0:
            Sa[i] = SDS * (0.4 + 0.6 * (T / T0))
        elif T <= Ts:
            Sa[i] = SDS
        elif T <= TL:
            Sa[i] = SD1 / T
        else:
            Sa[i] = SD1 * TL / (T**2)
    return Sa, T0, Ts

if not USE_MODE_A:
    SMS = Fa * Ss; SM1 = Fv * S1
    SDS = (2.0/3.0) * SMS
    SD1 = (2.0/3.0) * SM1
# TL remains as given; if unknown, pick a regional default (e.g., 8–16 s)

Sa_code, T0, Ts = asce716_design_spectrum(periods, SDS, SD1, TL)

# PGA for annotation
PGA = [float(np.max(np.abs(tr.data))) for tr in trs]

# ---- plotting ----
comp_colors = {"EW":"purple","NS":"orange","UD":"black"}
fig = plt.figure(figsize=(14,6))

# Left: spectra (0–1 s)
ax1 = plt.subplot(1,2,1)
for Sa, lab in zip(Sa_list, lbls):
    col = comp_colors.get(lab,"blue")
    ax1.plot(periods, Sa, lw=2.0, color=col, label=lab)
# ASCE 7-16 overlay
ax1.plot(periods, Sa_code, lw=2.5, color="tab:blue", linestyle="--",
         label=f"ASCE 7-16 (SDS={SDS:.2f}g, SD1={SD1:.2f}g)")
# markers
for Sa, lab in zip(Sa_list, lbls):
    col = comp_colors.get(lab,"blue")
    i = int(np.argmax(Sa))
    ax1.plot(periods[i], Sa[i], "o", ms=4, color=col)
# Ts, T0 guides
ax1.axvline(T0, color="gray", ls=":", lw=1)
ax1.axvline(Ts, color="gray", ls=":", lw=1)
ax1.text(T0, ax1.get_ylim()[1]*0.92, "T0", ha="center", va="top", fontsize=8)
ax1.text(Ts, ax1.get_ylim()[1]*0.92, "Ts", ha="center", va="top", fontsize=8)

ax1.set_xscale("linear"); ax1.set_xlim(0.0, 1.0)
ax1.set_xticks([0.0,0.25,0.5,0.75,1.0])
ax1.set_xlabel("T (s)"); ax1.set_ylabel("Sa (g)")
ax1.grid(True, which="both", alpha=0.35)
ax1.legend(title=MSEED, loc="upper right")

# Right: waveforms + PGA
for i,(tr,lab,pga) in enumerate(zip(trs,lbls,PGA), start=1):
    ax = plt.subplot(3,2,2*i)
    t = np.arange(tr.stats.npts) * tr.stats.delta
    col = comp_colors.get(lab,"blue")
    ax.plot(t, tr.data, lw=1.0, color=col)
    ax.set_xlim(0, t[-1]); ax.grid(True, alpha=0.3)
    ax.set_title(lab); ax.set_ylabel("Acc (g)")
    ax.text(0.01,0.95,f"PGA = {pga:.4f} g", transform=ax.transAxes,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, lw=0),
            color=col, fontsize=9)
    if i<3: ax.set_xticklabels([])
    else:   ax.set_xlabel("t (s)")

fig.suptitle("Recorded Sa vs ASCE 7-16 Design Spectrum (5% damping)", y=0.98)
plt.tight_layout(); plt.show()