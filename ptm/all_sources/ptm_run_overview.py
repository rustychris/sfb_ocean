import glob, os
from stompy.model.fish_ptm import ptm_tools

## 
run_dir='all_source_select_w_const'

bin_files=glob.glob(os.path.join(run_dir,'*_bin.out'))

bin_files.sort()
##

bins=[ptm_tools.PtmBin(bf) for bf in bin_files]

##

for bf,b in zip(bin_files,bins):
    end_count=len(b.read_timestep(-1)[1])
    print(f"{os.path.basename(bf):30s}: {end_count}")

