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

local ppl_suite_m2d2_wiki_val = import '../../../ai2-llm-eval/configs/task_sets/eval_suite_ppl_val_m2d2_wiki.libsonnet';
local ppl_suite_m2d2_wiki_test = import '../../../ai2-llm-eval/test_fixtures/task_sets/test_eval_suite_ppl_val_m2d2_wiki.libsonnet';


//❗Set gsheet to the name of your google sheet.
// Set it to null if you do not want your results to be uploaded to a google sheet (they will still be saved as an object).
 local gsheet = "seq-fos-upsamp-3"; //"my-new-gsheet-CHANGE-ME";

// Models to evaluate

local models = [
    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/seq-fos-decon-110m/seq-fos-Physics-upsamp",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },
    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/seq-fos-decon-110m/seq-fos-PoliticalScience-upsamp",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },
    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/seq-fos-decon-110m/seq-fos-Psychology-upsamp",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },
    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/seq-fos-decon-110m/seq-fos-Sociology-upsamp",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },
];

local task_sets = [
//    rc20_tasks_test.task_set,
//    rc20_tasks.task_set,
//    gen_tasks_test.task_set,
//    gen_tasks.task_set,
//    ppl_suite.task_set,
    ppl_suite_s2orc_val.task_set,
//    ppl_suite_s2orc_test.task_set,
//    ppl_suite_m2d2_wiki_val.task_set,
//    ppl_suite_m2d2_wiki_val.task_set_L1,
//    ppl_suite_m2d2_wiki_test.task_set,
//    ppl_suite_m2d2_wiki_test.task_set_L1,
];


{
    steps: utils.create_pipeline(models, task_sets, gsheet)
}