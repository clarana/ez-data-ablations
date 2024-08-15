/*--------------------------------------- Configurations -----------------------------------------*/

/*
local utils = import '../../ai2-llm-eval/configs/utils.libsonnet';

// These are using smaller test sets.
local rc20_tasks = import '../../ai2-llm-eval/test_fixtures/task_sets/test_rc20_tasks.libsonnet';
local gen_tasks = import '../../ai2-llm-eval/test_fixtures/task_sets/test_gen_tasks.libsonnet';
local ppl_suite = import '../../ai2-llm-eval/configs/task_sets/eval_suite_ppl_val_v3.libsonnet';

local ppl_suite_s2orc_val = import '../../ai2-llm-eval/test_fixtures/task_sets/eval_suite_ppl_val_xppt.libsonnet';
local ppl_suite_s2orc_test = import '../../ai2-llm-eval/test_fixtures/task_sets/test_eval_suite_ppl_val_xppt.libsonnet';
*/

local utils = import '../../../LLM/evaluation/experiments/utils.libsonnet';

// These are using smaller test sets.
local rc20_tasks = import '../../../LLM/evaluation/experiments/task_sets/test_sets/test_rc20_tasks.libsonnet';
local gen_tasks = import '../../../LLM/evaluation/experiments/task_sets/test_sets/test_gen_tasks.libsonnet';
local ppl_suite = import '../../../LLM/evaluation/experiments/task_sets/eval_suite_ppl_val_v3_not_deconned.libsonnet';

local ppl_suite_s2orc_val = import '../../../LLM/evaluation/experiments/task_sets/eval_suite_ppl_val_xppt.libsonnet';
//local ppl_suite_s2orc_test = import '../../../LLM/evaluation/experiments/task_sets/test_sets/test_eval_suite_ppl_val_xppt.libsonnet';


//❗Set gsheet to the name of your google sheet.
// Set it to null if you do not want your results to be uploaded to a google sheet (they will still be saved as an object).
 local gsheet = "catwalk-xppt-computerscience"; //"my-new-gsheet-CHANGE-ME";

// Models to evaluate

local models = [
    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-ComputerScience1970-2008",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },
    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-ComputerScience2009-2011",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },
    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-ComputerScience2012-2013",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },
    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-ComputerScience2014-2015",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },
    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-ComputerScience2016-2016",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },
    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-ComputerScience2017-2017",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },
    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-ComputerScience2018-2018",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },
    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-ComputerScience2019-2019",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },
    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-ComputerScience2020-2020",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },
    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-ComputerScience2021-2021",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },
    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-ComputerScience2022-2022",
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
    ppl_suite.task_set,
    ppl_suite_s2orc_val.task_set,
//    ppl_suite_s2orc_test.task_set,
];


{
    steps: utils.create_pipeline(models, task_sets, gsheet)
}
