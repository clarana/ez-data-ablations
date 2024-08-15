# Converting to OLMo to use olmo-eval

## converting XP-PT weights to OLMo

Given some XP-PT checkpoint (state dict) like customgpt-110m-tokens-41943040000-rank-0, choose a config for the appropreate model size in `xp-pt/xp-pt-as-olmo/configs`

```
python xp-pt/xp-pt-as-olmo/convert_checkpoint.py --checkpoint <path to checkpoint> --config xp-pt/xp-pt-as-olmo/configs/<config for your model size> --output_dir <dir to output weights and config>
```

## evaluating model in olmo-eval branch of catwalk

We're going to walk you through the olmo evaluation pipeline now with some tips and tricks not (yet) in [their readme](https://github.com/allenai/LLM/blob/main/evaluation/README.md)

```
git clone git@github.com:allenai/LLM.git
cd LLM
git reset --hard 8dd276ca4a16ffb74cdce90b2928fc78d5beafff
cd ../

pip install -e "LLM[dev]"
pip install -r LLM/evaluation/requirements.txt

```

Get a github token from https://github.com/settings/tokens/new with “repo” scope access and nothing else. DONT FORGET TO CONFIGURE SSO on the page after it takes you too after you generate the token. Finally export the token:
```
export GITHUB_TOKEN=<my new token>
```

The readme says you have to gcloud auth as well, tho this might only be needed if you're using models in gs which I never got to work.
```
gcloud auth login
```

Make a google sheet spreadsheet and share it with `olmo-eval@ai2-allennlp.iam.gserviceaccount.com`. This sheet will be used later to recieve eval results.

Also get something called an API json from [here](https://console.cloud.google.com/iam-admin/serviceaccounts/details/101308414346962828659;edit=true/keys?project=ai2-allennlp) and then add it like this:

```python
from tango.integrations.beaker.common import get_client
beaker = get_client("<beaker_workspace>")

with open("credentials_file.json") as f:
    beaker.secret.write("GDRIVE_SERVICE_ACCOUNT_JSON", f.read())
```

Next you'll get the ppl eval data from `<the olmo bucket>/eval-data/perplexity/v3/` and put it on NFS.

Likewise put the converted xp-pt-as-olmo model checkpoint dir (containing `model.pt` and `config.yaml`) on NFS too.

Next you'll need to update settings to work in beaker in `LLM/evaluation/tango-in-beaker.yml`.
```
- name: EVAL_DATA_PATH
    # ❗Change this to the common location 👇
      value: <change this to a NFS location where you put ppl data>
```
```
- name: GLOBAL_MODEL_DIR
    # ❗Change this to the common location 👇
      value: <change this to a NFS location that contains the eval model checkpoints>
```
Make a new gcs bucket, `gsutil mb gs://my_new_bucket`, and set it as your workspace
```
workspace:
  type: "gs"
  # ❗Change this to the workspace you want to use 👇
  workspace: "my_new_bucket"
  project: "ai2-allennlp"
```
```
workspace:
  type: "gs"
  # ❗Change this to the workspace you want to use 👇
  workspace: "<my_beaker_workspace>"
  project: "ai2-allennlp"
```

Then you'll want to make changes to the template `xp-pt/xp-pt-as-olmo/xp-pt-catwalk.jsonnet`:
```
//❗Set gsheet to the name of your google sheet.
// Set it to null if you do not want your results to be uploaded to a google sheet (they will still be saved as an object).
local gsheet = "my-new-gsheet-CHANGE-ME";
```

```
        model_path: "/net/nfs/allennlp/path/to/model/dir/", # ❗ Provide path to dir containing model weights and config
```

Now finally you are ready to run the evaluation!
```
tango --settings LLM/evaluation/tango-in-beaker.yml run xp-pt/xp-pt-as-olmo/xp-pt-catwalk.jsonnet
```