import os
import re
from pathlib import Path

def check_file_contents(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Check for the main success message
            if "All checks passed — results are fully reproducible." in content:
                # Also verify no failures were recorded
                failed_match = re.search(r"Failed\s*:\s*(\d+)", content)
                if failed_match and int(failed_match.group(1)) == 0:
                    return "OK"
                elif failed_match and int(failed_match.group(1)) > 0:
                    return f"FAILED ({failed_match.group(1)} checks failed)"
            
            # If the file exists but didn't finish properly
            return "INCOMPLETE (Success message not found)"
            
    except UnicodeDecodeError:
        try:
            # Fallback for ANSI encoded files
            with open(filepath, 'r', encoding='cp1252') as f:
                content = f.read()
                if "All checks passed" in content:
                    return "OK"
                return "INCOMPLETE / ERR"
        except Exception as e:
            return f"ERROR READING FILE: {e}"
    except Exception as e:
        return f"ERROR: {e}"

def main():
    print("Checking Verification Results...")
    base_dir = Path("D:/Bob_Skripsi_Do Not Delete/Analysis")
    
    modes = ["Normal", "Demography"]
    pcts = [1, 2, 3, 4, 5, 20]
    thresholds = [1.0, 5.0, 10.0]  # Note: The actual files use 1.0, 5.0, 10.0, not 1, 5, 10
    
    for mode in modes:
        print(f"\n--- {mode} Models ---")
        for pct in pcts:
            for k in thresholds:
                if pct == 20 and mode == "Demography":
                    continue
                fname = base_dir / mode / f"{pct}_percent" / f"verify_{k}.txt"
                if fname.exists():
                    status = check_file_contents(fname)
                    print(f"[{mode}] {pct}% (threshold {k}): {status}")
                else:
                    print(f"[{mode}] {pct}% (threshold {k}): MISSING ({fname})")

if __name__ == "__main__":
    main()
