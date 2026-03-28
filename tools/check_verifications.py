import os
from pathlib import Path

def main():
    print("Checking Verification Results...")
    base_dir = Path("D:/Bob_Skripsi_Do Not Delete/Analysis")
    
    modes = ["Normal", "Demography"]
    pcts = [1, 2, 3, 4, 5, 20]
    thresholds = [1, 5, 10]
    
    for mode in modes:
        print(f"\n--- {mode} Models ---")
        for pct in pcts:
            for k in thresholds:
                fname = base_dir / mode / f"{pct}_percent" / f"verify_{k}.txt"
                if fname.exists():
                    print(f"[{mode}] {pct}% (threshold {k}): OK - Found logging")
                else:
                    print(f"[{mode}] {pct}% (threshold {k}): MISSING")

if __name__ == "__main__":
    main()
