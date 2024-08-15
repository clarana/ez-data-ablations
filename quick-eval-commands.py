import os
import sys

window_size=3
version="v2"

fos = sys.argv[1]
i = int(sys.argv[2]) # can be negative, start with -1 if window_size = 5
gpu_ind = int(sys.argv[3]) if len(sys.argv) > 3 else 0
partition_years = sorted([part.split(',')[1] for part in os.listdir("/net/nfs.cirrascale/allennlp/claran/xppt-sources-branched/s2orc-recombined") if fos in part])

min_model_i, max_model_i = max(0, i-(window_size//2)), min(len(partition_years)-1, i+(window_size//2)+1)
#min_model_i, max_model_i = max(0, i-(window_size//2)), min(len(partition_years), i+(window_size//2)+1)
min_data_i, max_data_i = max(0, min_model_i-1), max_model_i+1

models = [f"branched-train-{fos}{years}-{version}" for years in partition_years[min_model_i:max_model_i]]
data = [f"{fos},{years}" for years in partition_years[min_data_i:max_data_i]]

models_str = " ".join(models)
data_str = " ".join(data)

s = f"""python eval_model_data.py \\
            --model_eval_mode uniform_avg_eval \\
            --data_eval_mode each \\
            --gpu_ind {gpu_ind} \\
            --model_names {models_str} \\
            --partition_names {data_str} \\
                              > >(tee -a "/net/nfs.cirrascale/allennlp/claran/quick-eval/{fos}-rolling{window_size}-{version}.txt")"""


print(s)




"""
e.g.:

python eval_model_data.py \
            --model_eval_mode uniform_avg_eval \
            --data_eval_mode each \
            --model_names branched-train-Mathematics1970-2002 branched-train-Mathematics2003-2006 \
            --partition_names Mathematics,1970-2002 Mathematics,2003-2006 Mathematics,2007-2009 \
                              > >(tee -a "/net/nfs.cirrascale/allennlp/claran/quick-eval/Mathematics-rolling3.txt")

python eval_model_data.py \
            --model_eval_mode uniform_avg_eval \
            --data_eval_mode each \
            --model_names branched-train-Mathematics1970-2002 branched-train-Mathematics2003-2006 branched-train-Mathematics2007-2009 \
            --partition_names Mathematics,1970-2002 Mathematics,2003-2006 Mathematics,2007-2009 Mathematics,2010-2011 \
                              > >(tee -a "/net/nfs.cirrascale/allennlp/claran/quick-eval/Mathematics-rolling3.txt")




"""

