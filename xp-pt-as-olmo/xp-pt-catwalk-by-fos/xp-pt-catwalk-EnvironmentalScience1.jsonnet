/*--------------------------------------- Configurations -----------------------------------------*/


local utils = import '../../../ai2-llm-eval/configs/utils.libsonnet';

// These are using smaller test sets.
local rc20_tasks_test = import '../../../ai2-llm-eval/test_fixtures/task_sets/test_rc20_tasks.libsonnet';
local rc20_tasks = import '../../../ai2-llm-eval/configs/task_sets/rc20_tasks.libsonnet';
local gen_tasks_test = import '../../../ai2-llm-eval/test_fixtures/task_sets/test_gen_tasks.libsonnet';
local gen_tasks = import '../../../ai2-llm-eval/configs/task_sets/gen_tasks.libsonnet';
local ppl_suite = import '../../../ai2-llm-eval/configs/task_sets/eval_suite_ppl_val_v3_not_deconned.libsonnet';

local ppl_suite_s2orc_val = import '../../../ai2-llm-eval/configs/task_sets/eval_suite_ppl_val_xppt.libsonnet';
local ppl_suite_s2orc_test = import '../../../ai2-llm-eval/test_fixtures/task_sets/test_eval_suite_ppl_val_xppt.libsonnet';


//❗Set gsheet to the name of your google sheet.
// Set it to null if you do not want your results to be uploaded to a google sheet (they will still be saved as an object).
 local gsheet = "catwalk-xppt-environmentalscience"; //"my-new-gsheet-CHANGE-ME";

// Models to evaluate

local models = [
    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-EnvironmentalScience1970-2013-decon-110m",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },
    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-EnvironmentalScience2014-2016-decon-110m",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },
    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-EnvironmentalScience2017-2018-decon-110m",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },
    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-EnvironmentalScience2019-2019-decon-110m",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },
];

local task_sets = [
//    rc20_tasks.task_set,
//    gen_tasks.task_set,
//    ppl_suite.task_set,
    ppl_suite_s2orc_val.task_set,
//    ppl_suite_s2orc_test.task_set,
];


{
    steps: utils.create_pipeline(models, task_sets, gsheet)
}