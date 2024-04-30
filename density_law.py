import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from scipy.interpolate import interp1d
import os

home_directory = os.path.expanduser("~")


stylefile = "%s/.config/matplotlib/stylelib/paper.mplstyle" %home_directory

if os.path.isfile(stylefile):
    plt.style.use("~/.config/matplotlib/stylelib/paper.mplstyle")

pixel_scale = 0.2 * u.arcsec
D = 4.25 * u.Mpc
d = np.tan(pixel_scale.to(u.deg) / (360 * u.deg) * 2 * np.pi * u.rad) * D.to(u.pc).value

def rescale_projection(n0, file="density_profile_long.dat", rmin=212):
    """Rescale a flux projection to match a given density n0 outside
    n0: float,
        Density in the outer regions
    rmin: float,
        Minimum radius to stop the interpolation
    """
    data = np.genfromtxt(file, names=["pixel", "flux"])
    # interpolate 0s
    nonzeros = data["flux"]!=0
    f = interp1d(data["pixel"][nonzeros], data["pixel"][nonzeros])
    data["flux"][~nonzeros] = f(data["pixel"][~nonzeros])
    factor = n0 / data["flux"][-1]
    n = data["flux"] * factor
    r = d * data["pixel"]
    # interpolate back to n0
    rminn = r[np.argmin(np.abs(r - rmin))]
    # make nism arbitrarily small inside rmin
    n[r < rminn] = nism / 5
    # don't go over 150 as it makes cloudy very slow
    newgrid = np.linspace(r[0], r[-1] + 3, 150)
    #f = interp1d(r[(r > rmax) | (r == rmaxx)], n[(r > rmax) | (r == rmaxx)], kind="slinear")
    #n[(r < rmax) & (r > rmaxx)] = f(r[(r < rmax) & (r > rmaxx)])
    f = interp1d(r, n, kind="slinear", fill_value=nism, bounds_error=False)
    n = f(newgrid)
    return newgrid, n

def store_density_law(outfile, r_cm, n):
    """Function to store density law into cloudy format
    outfile:str,
        Output file name
    r_cm: array
        The radius in cm
    n: array
    """
    with open(outfile, "w+") as f:
        f.write("dlaw table\n")
        for n_i, r_i in zip(n, r_cm):
            f.write("%.4f %.4f\n" % (np.log10(r_i), np.log10(n_i)))
        f.write("end of dlaw")


def globule(R, n0=0.45, Rscale=100, alpha=-2):
    return n0*(1 - R/Rscale)**alpha

def powerlaw(R, n0=0.45, r_0=10, alpha=-2):
    return n0*(R/r_0)**alpha

def exponential(R, nism, r_out=10, rwindshock=120, rshock=197.804):
    n = np.ones_like(R) / 50
    n =  nism * np.exp((R/r_out)**2 - 1)
    n[r >  rshock] = np.ones(len(n))[r> rshock]
    return n

def create_density_law(r, nism, R1=120, R2=200, Rout=270):
    n = nism * np.ones_like(r) / 1000
    # outside the bubble ISM
    n[r > R1] = nism * np.ones_like(r)[r > R1] * 0.9 /100
    n[r > R2] = nism * np.ones_like(r)[r > R2] * 100
    n[r > Rout] = nism * np.ones_like(r)[r > Rout]
    return n

# the value calculated by Zhou, revisited downwards by 25% and assuming 90% of it hydrogen
nism = 0.45 * 0.75 * 0.9

## lONG AXIS
rmin = 212
r, n = rescale_projection(nism, "density_profile_long.dat", rmin=rmin)
print("Radius in parsecs", r)
r_cm = (r * u.pc).to(u.cm).value
store_density_law("density_law_long.dat", r_cm, n)

#plt.plot(r, np.log10(n), label="H$\\alpha$ long-axis (rescaled)", color="C1", ls="solid")


#plt.axvspan(120, 200, color="red", alpha=0.2)
#plt.text(120 + 80 / 2, -3, "Shocked \n Wind", fontsize=24, horizontalalignment="center")
fig, axes = plt.subplots(2, 1, sharey=True, sharex=True,
                         gridspec_kw={"hspace":0})

fig.supxlabel("$r$ (pc)", fontsize=24)
for ax in axes:
    ax.axhline(np.log10(nism), color="black", zorder=-10, ls="--")
    ax.set_ylabel("$\log$ $n_\mathrm{H}$ (cm$^{-3}$)")

ax = axes[0]
ax.text(30, -0.4, "Semi-major \n axis", fontsize=22, horizontalalignment="center")
selector = r > (rmin - 10)
ax.plot(r[selector], np.log10(n[selector]), ls="solid", color="C0") # , label="Siwek"
rinbubble = 205
routbubble = 285
ax.axvspan(rinbubble, routbubble, color="green", alpha=0.2)
ax.text(r[np.argmax(n)], -0.4, "Shocked \n ISM", fontsize=24, horizontalalignment="center")
r, n = rescale_projection(nism, "density_profile_long.dat", rmin=0)

ax.plot(r, np.log10(n), label="H$\\alpha$ (rescaled)", color="C0", zorder=-10, ls="--")


## SHORT AXIS
ax = axes[1]
ax.text(30, -0.4, "Semi-minor \n axis", fontsize=22, horizontalalignment="center")
rmin = 110
r, n = rescale_projection(nism, "density_profile_short.dat", rmin=rmin)
print("Radius in parsecs", r)
r_cm = (r * u.pc).to(u.cm).value
store_density_law("density_law_short.dat", r_cm, n)

rinbubble = 105
routbubble = 200
ax.axvspan(rinbubble, routbubble, color="green", alpha=0.2)
ax.plot(r[r > (rmin - 5)], np.log10(n[r> (rmin - 5)]), ls="solid", color="C0")
r, n = rescale_projection(nism, "density_profile_short.dat", rmin=0)
ax.plot(r, np.log10(n), label="H$\\alpha$ (rescaled)", color="C0", ls="--", zorder=-10)
ax.text(250, -0.4, "Ambient \n ISM", fontsize=22, horizontalalignment="center")
plt.legend()
plt.savefig("denstiy_law.pdf")
