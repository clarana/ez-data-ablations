/*--------------------------------------- Configurations -----------------------------------------*/

local utils = import 'utils.libsonnet';

// These are using smaller test sets.
local rc20_tasks = import 'task_sets/test_rc20_tasks.libsonnet';
local gen_tasks = import 'task_sets/test_gen_tasks.libsonnet';
local ppl_suite = import 'task_sets/eval_suite_ppl_val_v3.libsonnet';


//❗Set gsheet to the name of your google sheet.
// Set it to null if you do not want your results to be uploaded to a google sheet (they will still be saved as an object).
local gsheet = "my-new-gsheet-CHANGE-ME";

// Models to evaluate

local models = [
    {
        model_path: "/net/nfs/allennlp/path/to/model/dir/", # ❗ Provide path to dir containing model weights and config
        gpus_needed: 1,
        trust_remote_code: true,
        //❗Task sets contain default values for prediction_kwargs. These can be overriden for each model here.
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },
    //❗Add more models here if you need,
    /*{
        model_path: "EleutherAI/pythia-1b", // path can be a huggingface model id
        revision: "step140000", //❗Specify checkpoint if needed
        gpus_needed: 1
    }*/
];

local task_sets = [
    rc20_tasks.task_set,
    gen_tasks.task_set,
    ppl_suite.task_set
];


{
    steps: utils.create_pipeline(models, task_sets, gsheet)
}
