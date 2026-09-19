"""
DS-MTFNet Full Dataset Forensic Analyzer
Outputs: dataset_analysis.csv + dataset_report.md
Run: python analyze_dataset_full.py
"""
import os, csv, time
import cv2, numpy as np
from PIL import Image

DATASET_ROOT    = "dataset"
SPLITS          = ["train", "val", "test"]
ORIGIN_CLASSES  = ["ai", "ai_edited", "real"]
CONTENT_CLASSES = ["animal", "face", "human"]
ANIMAL_SUBS     = ["cat", "dog", "elephant", "horse", "lion"]
VALID_EXTS      = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CSV_OUT         = "dataset_analysis.csv"
REPORT_OUT      = "dataset_report.md"

def extract(img_path):
    try:
        p = Image.open(img_path).convert("RGB")
        if p.size[0] > 320 or p.size[1] > 320:
            p = p.resize((320, 320), Image.BILINEAR)
        u8  = np.array(p, dtype=np.uint8)
        arr = u8.astype(np.float32)
        H, W = u8.shape[:2]
        # Noise
        bl   = cv2.GaussianBlur(u8,(3,3),1.0).astype(np.float32)
        ng   = 0.299*np.abs(arr-bl)[:,:,0]+0.587*np.abs(arr-bl)[:,:,1]+0.114*np.abs(arr-bl)[:,:,2]
        # Texture
        gray = 0.299*arr[:,:,0]+0.587*arr[:,:,1]+0.114*arr[:,:,2]
        lm   = cv2.blur(gray,(8,8)); lm2 = cv2.blur(gray**2,(8,8))
        lstd = np.sqrt(np.maximum(lm2-lm**2,0))
        # Saturation / Value
        hsv  = cv2.cvtColor(u8, cv2.COLOR_RGB2HSV)
        sat  = hsv[:,:,1]/255.0; val = hsv[:,:,2]/255.0
        # ELA
        bgr  = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
        _, enc = cv2.imencode('.jpg', bgr,[int(cv2.IMWRITE_JPEG_QUALITY),95])
        ela  = np.mean(np.abs(arr-cv2.cvtColor(cv2.imdecode(enc,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB).astype(np.float32)),axis=2)
        # Sharpness
        shp  = float(cv2.Laplacian(cv2.cvtColor(u8,cv2.COLOR_RGB2GRAY),cv2.CV_64F).var())
        # JPEG blocks
        g    = cv2.cvtColor(u8,cv2.COLOR_RGB2GRAY).astype(np.float32)
        hd   = float(np.abs(g[:,8::8]-g[:,7:-1:8]).mean()) if W>8 else 0.0
        vd   = float(np.abs(g[8::8,:]-g[7:-1:8,:]).mean()) if H>8 else 0.0
        return {
            "noise_mean":round(float(ng.mean()),4),"noise_std":round(float(ng.std()),4),
            "texture_mean":round(float(lstd.mean()),4),"texture_std":round(float(lstd.std()),4),
            "sat_mean":round(float(sat.mean()),4),"sat_std":round(float(sat.std()),4),
            "sat_p90":round(float(np.percentile(sat,90)),4),"val_mean":round(float(val.mean()),4),
            "ela_mean":round(float(ela.mean()),4),"ela_std":round(float(ela.std()),4),
            "ela_p95":round(float(np.percentile(ela,95)),4),
            "r_mean":round(float(arr[:,:,0].mean()/255),4),
            "g_mean":round(float(arr[:,:,1].mean()/255),4),
            "b_mean":round(float(arr[:,:,2].mean()/255),4),
            "rgb_std":round(float(arr.std()/255),4),
            "sharpness":round(shp,2),
            "jpeg_block":round((hd+vd)/2,4),
            "width":p.size[0],"height":p.size[1],
            "file_kb":round(os.path.getsize(img_path)/1024,1),"error":""
        }
    except Exception as e:
        blank = {k:"" for k in ["noise_mean","noise_std","texture_mean","texture_std",
            "sat_mean","sat_std","sat_p90","val_mean","ela_mean","ela_std","ela_p95",
            "r_mean","g_mean","b_mean","rgb_std","sharpness","jpeg_block","width","height","file_kb"]}
        blank["error"] = str(e); return blank

def walk():
    for split in SPLITS:
        sp = os.path.join(DATASET_ROOT, split)
        if not os.path.isdir(sp): continue
        for orig in ORIGIN_CLASSES:
            op = os.path.join(sp, orig)
            if not os.path.isdir(op): continue
            for cont in CONTENT_CLASSES:
                cp = os.path.join(op, cont)
                if not os.path.isdir(cp): continue
                subs = ANIMAL_SUBS if cont=="animal" else [cont]
                for sub in subs:
                    pp = os.path.join(cp,sub) if cont=="animal" else cp
                    if not os.path.isdir(pp): continue
                    for f in sorted(os.listdir(pp)):
                        if os.path.splitext(f)[1].lower() in VALID_EXTS:
                            yield split, orig, cont, sub, os.path.join(pp,f)

def agg(rows, k):
    v = [r[k] for r in rows if isinstance(r.get(k),(int,float))]
    if not v: return {"mean":"N/A","std":"N/A","min":"N/A","max":"N/A"}
    return {"mean":round(float(np.mean(v)),4),"std":round(float(np.std(v)),4),
            "min":round(float(np.min(v)),4),"max":round(float(np.max(v)),4)}

def sep(ra,rb,k):
    va=[r[k] for r in ra if isinstance(r.get(k),(int,float))]
    vb=[r[k] for r in rb if isinstance(r.get(k),(int,float))]
    if not va or not vb: return 0.0
    return round(abs(np.mean(va)-np.mean(vb))/(np.std(va)+np.std(vb)+1e-8),3)

def main():
    print("="*70)
    print("  DS-MTFNet Full Dataset Forensic Analyzer")
    print("="*70)
    all_imgs = list(walk())
    total = len(all_imgs)
    print(f"\n  Total images: {total:,}\n")
    fields = ["split","origin_class","content_class","subclass","filename",
              "noise_mean","noise_std","texture_mean","texture_std",
              "sat_mean","sat_std","sat_p90","val_mean",
              "ela_mean","ela_std","ela_p95",
              "r_mean","g_mean","b_mean","rgb_std",
              "sharpness","jpeg_block","width","height","file_kb","error"]
    by_class = {}
    by_combo = {}
    t0 = time.time(); err_count = 0
    with open(CSV_OUT,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i,(split,orig,cont,sub,fp) in enumerate(all_imgs,1):
            if i%200==0 or i==total:
                el=time.time()-t0; eta=(el/i)*(total-i)
                print(f"  [{i:>5}/{total}] {orig:10s} {cont}/{sub}  elapsed={el:.0f}s ETA={eta:.0f}s",end="\r")
            feats = extract(fp)
            if feats["error"]: err_count+=1
            row = {"split":split,"origin_class":orig,"content_class":cont,"subclass":sub,"filename":os.path.basename(fp)}
            row.update(feats); w.writerow(row)
            k=(split,orig)
            by_class.setdefault(k,[])
            if not feats["error"]: by_class[k].append(feats)
            ck=(split,orig,cont,sub)
            by_combo[ck]=by_combo.get(ck,0)+1
    elapsed=time.time()-t0
    print(f"\n\n  CSV saved -> {CSV_OUT}  ({total} rows, {err_count} errors, {elapsed:.1f}s)")
    # ── Report ──────────────────────────────────────────────────────────────
    FEATS = [
        ("noise_mean","Noise Mean"),("noise_std","Noise Std"),
        ("texture_mean","Texture Mean"),("sat_mean","Saturation Mean"),
        ("sat_p90","Saturation P90"),("ela_mean","ELA Mean"),
        ("ela_p95","ELA P95"),("sharpness","Sharpness"),
        ("jpeg_block","JPEG Block Score"),("file_kb","File Size KB"),
    ]
    L=[]
    L.append("# DS-MTFNet — Full Dataset Analysis Report\n\n")
    L.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')} | Total images: {total:,} | Errors: {err_count}\n\n")
    # Section 1
    L.append("## 1. Dataset Size by Split and Class\n\n")
    L.append("| Split | Origin | Count |\n|-------|--------|-------|\n")
    split_tot={}
    for sp in SPLITS:
        for og in ORIGIN_CLASSES:
            n=len(by_class.get((sp,og),[]))
            L.append(f"| {sp} | {og} | {n:,} |\n")
            split_tot[sp]=split_tot.get(sp,0)+n
    for sp in SPLITS:
        L.append(f"| **{sp}** | **TOTAL** | **{split_tot.get(sp,0):,}** |\n")
    L.append("\n")
    # Section 2
    L.append("## 2. Content Sub-Class Breakdown (train)\n\n")
    L.append("| Origin | Content | Subclass | Count |\n|--------|---------|----------|-------|\n")
    for (sp,og,co,su),cnt in sorted(by_combo.items()):
        if sp=="train": L.append(f"| {og} | {co} | {su} | {cnt} |\n")
    L.append("\n")
    # Section 3
    L.append("## 3. Forensic Feature Statistics per Class (train)\n\n")
    for fk,fn in FEATS:
        L.append(f"### {fn}\n\n| Origin | Mean | Std | Min | Max |\n|--------|------|-----|-----|-----|\n")
        for og in ORIGIN_CLASSES:
            a=agg(by_class.get(("train",og),[]),fk)
            L.append(f"| **{og}** | {a['mean']} | {a['std']} | {a['min']} | {a['max']} |\n")
        L.append("\n")
    # Section 4
    L.append("## 4. Class Separation Scores\n\n")
    L.append("> Score = |mean_A - mean_B| / (std_A + std_B).  Score>0.5 = useful | Score<0.2 = weak\n\n")
    L.append("| Feature | AI vs Real | AI vs Edited | Real vs Edited | Best Pair |\n|---------|-----------|-------------|----------------|----------|\n")
    for fk,fn in FEATS:
        ra=by_class.get(("train","ai"),[])
        rr=by_class.get(("train","real"),[])
        re=by_class.get(("train","ai_edited"),[])
        s1=sep(ra,rr,fk); s2=sep(ra,re,fk); s3=sep(rr,re,fk)
        bst=max([("AI vs Real",s1),("AI vs Edited",s2),("Real vs Edited",s3)],key=lambda x:x[1])[0]
        L.append(f"| {fn} | {s1} | {s2} | {s3} | {bst} |\n")
    L.append("\n")
    # Section 5
    L.append("## 5. AI-Edited Class Deep Dive\n\n")
    L.append("| Feature | ai | real | ai_edited | edit-ai gap | edit-real gap |\n")
    L.append("|---------|-----|------|-----------|-------------|---------------|\n")
    ra=by_class.get(("train","ai"),[])
    rr=by_class.get(("train","real"),[])
    re=by_class.get(("train","ai_edited"),[])
    for fk,fn in [("noise_mean","Noise Mean"),("sat_mean","Saturation Mean"),
                   ("sat_p90","Sat P90"),("ela_mean","ELA Mean"),
                   ("sharpness","Sharpness"),("jpeg_block","JPEG Block")]:
        aim=agg(ra,fk)["mean"]; rim=agg(rr,fk)["mean"]; eim=agg(re,fk)["mean"]
        if all(isinstance(x,float) for x in [aim,rim,eim]):
            g1=round(abs(eim-aim),4); g2=round(abs(eim-rim),4)
        else:
            g1=g2="N/A"
        L.append(f"| {fn} | {aim} | {rim} | {eim} | {g1} | {g2} |\n")
    L.append("\n")
    # Section 6
    L.append("## 6. Train vs Val Distribution Shift (ai_edited only)\n\n")
    L.append("| Feature | Train Mean | Val Mean | Shift | Flag |\n|---------|-----------|---------|-------|------|\n")
    ret=by_class.get(("train","ai_edited"),[])
    rev=by_class.get(("val","ai_edited"),[])
    for fk,fn in [("noise_mean","Noise Mean"),("sat_mean","Sat Mean"),
                   ("ela_mean","ELA Mean"),("sharpness","Sharpness"),("jpeg_block","JPEG Block")]:
        tm=agg(ret,fk)["mean"]; vm=agg(rev,fk)["mean"]
        if isinstance(tm,float) and isinstance(vm,float):
            sh=round(vm-tm,4)
            flag="LARGE SHIFT" if abs(sh)>0.05*(abs(tm)+1e-8) else "OK"
        else:
            sh="N/A"; flag=""
        L.append(f"| {fn} | {tm} | {vm} | {sh} | {flag} |\n")
    L.append("\n")
    # Section 7
    L.append("## 7. Recommendations\n\n")
    L.append("| Priority | Action | Expected Val Gain |\n|----------|--------|-------------------|\n")
    L.append("| P1 HIGH | Add 800+ new ai_edited images from diverse tools | +5-10% |\n")
    L.append("| P1 HIGH | Add background-swap and object-replacement edits | +3-6% |\n")
    L.append("| P2 MED  | Test-Time Augmentation (TTA) at inference | +2-4% |\n")
    L.append("| P2 MED  | CutMix instead of Mixup for ai_edited class | +1-3% |\n")
    L.append("| P3 LOW  | Ensemble two models and average predictions | +3-5% |\n\n")
    L.append("### New ai_edited images needed per folder\n\n")
    L.append("| Folder | Current Count | Target | Gap |\n|--------|--------------|--------|-----|\n")
    for co in ["human","face"]:
        cur=by_combo.get(("train","ai_edited",co,co),0)
        tgt=max(by_combo.get(("train","ai",co,co),0),by_combo.get(("train","real",co,co),0))
        L.append(f"| ai_edited/{co} | {cur} | {tgt} | {max(0,tgt-cur)} |\n")
    for su in ANIMAL_SUBS:
        cur=by_combo.get(("train","ai_edited","animal",su),0)
        ai_cnt=by_combo.get(("train","ai","animal",su),0)
        L.append(f"| ai_edited/animal/{su} | {cur} | {ai_cnt} | {max(0,ai_cnt-cur)} |\n")
    L.append("\n")
    with open(REPORT_OUT,"w",encoding="utf-8") as f:
        f.writelines(L)
    print(f"  Report saved -> {REPORT_OUT}")
    print(f"\n{'='*70}\n  DONE!  CSV: {CSV_OUT}  |  Report: {REPORT_OUT}\n{'='*70}\n")

if __name__=="__main__":
    main()
