"""Head-to-head: our superellipse vs the published exponential flank profile.

Both reduced to ONE free parameter on the same normalised data, so r2 is a fair
comparison (equal degrees of freedom).

  superellipse   v = (1 - u^e)^(1/e)
  exponential    v = (exp(-k u) - exp(-k)) / (1 - exp(-k))
                 the Gee/Karatson/Favalli form a*exp(-b x)+c, with the two
                 anchors v(0)=1 and v(1)=0 absorbing a and c.
"""
import sys, ast, pathlib, glob, os, numpy as np, pandas as pd
sys.path.insert(0,'.')
src = open('make_ideal_overlay_figure.py').read().split('def main()')[0].replace(
    'HERE = pathlib.Path(__file__).resolve().parent','HERE = pathlib.Path(".").resolve()')
exec(src)
ns = load_app(); analyse = ns["analyse_profile"]

K_GRID = np.round(np.arange(0.10, 12.001, 0.02), 3)
def expo(u, k):
    return (np.exp(-k*np.clip(u,0,1)) - np.exp(-k)) / (1.0 - np.exp(-k))

def best(u, v, grid, fn):
    sst = float(np.sum((v-v.mean())**2))
    if sst <= 0: return np.nan, np.nan
    bp, br = np.nan, -np.inf
    for p in grid:
        ssr = float(np.sum((v-fn(u,p))**2)); r2 = 1-ssr/sst
        if r2 > br: bp, br = float(p), r2
    return bp, br

man = pd.read_csv(REAL/"manifest.csv")
stem = man["file"].astype(str).str.split("_az").str[0]
pretty = dict(zip(stem, man["volcano"])); kinds = dict(zip(stem, man["kind"]))

rows=[]
for f in sorted(glob.glob(str(REAL/"*_az*.csv"))):
    key=os.path.basename(f).split("_az")[0]; az=os.path.basename(f).split("_az")[1][:3]
    d=pd.read_csv(f); x=d.iloc[:,0].to_numpy(float); z=d.iloc[:,1].to_numpy(float)
    m,_,_=analyse(x,z)
    if not m or not np.isfinite(m.get("b",np.nan)): continue
    u,v,s,_ = apex_normalise(x,z,m)
    e,r2e = fit_exponent(u,v)
    k,r2k = best(u,v,K_GRID,expo)
    rows.append(dict(key=key,volcano=pretty.get(key,key),kind=kinds.get(key,""),az=az,
                     e=e,r2_super=r2e,k=k,r2_exp=r2k))
df=pd.DataFrame(rows)
df["winner"]=np.where(df.r2_super>df.r2_exp,"superellipse","exponential")
g=df.groupby(["volcano","kind"]).agg(e=("e","median"),r2_super=("r2_super","median"),
     k=("k","median"),r2_exp=("r2_exp","median")).reset_index()
g["diff"]=(g.r2_super-g.r2_exp).round(3)
print(g.sort_values("e").to_string(index=False))
print()
print("transect-level wins:", df.winner.value_counts().to_dict(), "of", len(df))
print("median r2  superellipse %.3f   exponential %.3f" % (df.r2_super.median(), df.r2_exp.median()))
print()
print("does k separate the types as well as e?")
print(df.groupby("kind")[["e","k"]].describe().loc[:,(slice(None),["min","50%","max"])].to_string())
df.to_csv("figures/shape_function_comparison.csv", index=False)
