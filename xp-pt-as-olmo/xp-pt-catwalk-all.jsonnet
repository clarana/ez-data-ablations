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

local utils = import '../../LLM/evaluation/experiments/utils.libsonnet';

// These are using smaller test sets.
local rc20_tasks = import '../../LLM/evaluation/experiments/task_sets/test_sets/test_rc20_tasks.libsonnet';
local gen_tasks = import '../../LLM/evaluation/experiments/task_sets/test_sets/test_gen_tasks.libsonnet';
local ppl_suite = import '../../LLM/evaluation/experiments/task_sets/eval_suite_ppl_val_v3.libsonnet';

local ppl_suite_s2orc_val = import '../../LLM/evaluation/experiments/task_sets/eval_suite_ppl_val_xppt.libsonnet';
//local ppl_suite_s2orc_test = import '../../LLM/evaluation/experiments/task_sets/test_sets/test_eval_suite_ppl_val_xppt.libsonnet';


//❗Set gsheet to the name of your google sheet.
// Set it to null if you do not want your results to be uploaded to a google sheet (they will still be saved as an object).
 local gsheet = "catwalk-results-xppt"; //"my-new-gsheet-CHANGE-ME";

// Models to evaluate

local models = [
    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/sequential-train-s2orc-cont", # ❗ Provide path to dir containing model weights and config
        gpus_needed: 1,
        trust_remote_code: true,
        //❗Task sets contain default values for prediction_kwargs. These can be overriden for each model here.
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },
    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/seed-xppt",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },
    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-AgriculturalAndFoodSciences2016-2018",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-AgriculturalAndFoodSciences2019-2020",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-AgriculturalAndFoodSciences2021-2021",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-AgriculturalAndFoodSciences2022-2022",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Art1970-2022",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Biology1970-1993",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Biology1994-1999",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Biology2000-2003",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Biology2004-2006",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Biology2007-2008",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Biology2009-2010",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Biology2011-2011",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Biology2012-2012",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Biology2013-2013",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Biology2014-2014",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Biology2015-2015",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Biology2016-2016",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Biology2017-2017",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Biology2018-2018",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Biology2019-2019",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Biology2020-2020",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Biology2021-2021",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Biology2022-2022",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Business1970-2017",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Business2018-2019",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Business2020-2020",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Business2021-2022",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Chemistry1970-2014",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Chemistry2015-2018",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Chemistry2019-2020",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Chemistry2021-2022",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-ComputerScience1970-2008",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-ComputerScience2009-2011",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-ComputerScience2012-2013",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-ComputerScience2014-2015",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-ComputerScience2016-2016",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-ComputerScience2017-2017",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-ComputerScience2018-2018",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-ComputerScience2019-2019",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-ComputerScience2020-2020",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-ComputerScience2021-2021",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-ComputerScience2022-2022",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Economics1970-2016",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Economics2017-2019",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Economics2020-2020",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Economics2021-2022",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Education1970-2018",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Education2019-2020",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Education2021-2022",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Engineering1970-2016",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Engineering2017-2019",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Engineering2020-2020",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Engineering2021-2021",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Engineering2022-2022",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-EnvironmentalScience1970-2013",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-EnvironmentalScience2014-2016",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-EnvironmentalScience2017-2018",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-EnvironmentalScience2019-2019",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-EnvironmentalScience2020-2020",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-EnvironmentalScience2021-2021",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-EnvironmentalScience2022-2022",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Geology1970-2022",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-History1970-2022",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Linguistics1970-2022",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-MaterialsScience1970-2015",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-MaterialsScience2016-2018",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-MaterialsScience2019-2020",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-MaterialsScience2021-2021",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-MaterialsScience2022-2022",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Mathematics1970-2002",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Mathematics2003-2006",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Mathematics2007-2009",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Mathematics2010-2011",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Mathematics2012-2013",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Mathematics2014-2015",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Mathematics2016-2017",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Mathematics2018-2018",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Mathematics2019-2019",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Mathematics2020-2020",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Mathematics2021-2021",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Mathematics2022-2022",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Medicine2010-2011",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Medicine2012-2012",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Medicine2013-2013",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Medicine2014-2014",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Medicine2019-2019",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Medicine2020-2020",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Medicine2021-2021",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Medicine2022-2022",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-NA1970-2019",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-NA2020-2022",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Philosophy1970-2022",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Physics1970-1997",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Physics1998-2000",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Physics2001-2002",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Physics2003-2004",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Physics2005-2006",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Physics2007-2008",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Physics2009-2009",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Physics2010-2010",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Physics2011-2011",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Physics2012-2012",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Physics2013-2013",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Physics2014-2014",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Physics2015-2015",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Physics2016-2016",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Physics2017-2017",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Physics2018-2018",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Physics2019-2019",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Physics2020-2020",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Physics2021-2021",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Physics2022-2022",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-PoliticalScience1970-2019",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-PoliticalScience2020-2022",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Psychology1970-2013",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Psychology2014-2016",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Psychology2017-2018",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Psychology2019-2020",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Psychology2021-2021",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Psychology2022-2022",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Sociology1970-2022",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/sequential-train-s2orc-cont",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-AgriculturalAndFoodSciences1970-2015",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/seed-xppt",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Medicine2016-2016",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Medicine2017-2017",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Medicine2018-2018",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Medicine2007-2009",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Medicine2015-2015",
        gpus_needed: 1,
        trust_remote_code: true,
        prediction_kwargs: {
            model_max_length: 2048,
            limit: null, //etc.
        }
    },    {
        model_path: "/net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-Medicine1970-2006",
        gpus_needed: 1,
        trust_remote_code: true,
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
    ppl_suite.task_set,
    ppl_suite_s2orc_val.task_set,
//    ppl_suite_s2orc_test.task_set,
];


{
    steps: utils.create_pipeline(models, task_sets, gsheet)
}
