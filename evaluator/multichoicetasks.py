import abc
import datasets
import re
import torch
import transformers

from torch.utils.data import DataLoader, Dataset


class ICLMultiChoiceTaskDataset(metaclass=abc.ABCMeta):
    """Only supports zero-shot for now.
    """
    def __init__(self, tokenizer, dataset_path, dataset_name=None, model_ctx_len=2048):
        super().__init__()

        self.tokenizer = tokenizer
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.model_ctx_len = model_ctx_len

        self.samples = []
        self.dataset = datasets.load_dataset(
            path=self.dataset_path,
            name=self.dataset_name,
            split='validation',
        )

        # prep examples
        self.prep_examples()

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)

    def prep_examples(self):
        """Append doc_ids to each example so that they are processed together in the metric
        """
        doc_id = 0
        for doc in self.dataset:
            # from EAI harness
            # how this all works:
            #          CTX      CONT
            # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
            # gpt2    \               \
            # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
            # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

            ctx = self.doc_to_text(doc)
            continuations = self.doc_to_continuations(doc)
            label_id = self.doc_to_label(doc)
            dc = self.doc_to_domain_conditional(doc)

            # tokenize
            ctx = self.token_encode(ctx)
            dc = self.token_encode(dc)

            for cont_id, continuation in enumerate(continuations):
                cont_str_len = len(continuation) - 1    # continuation contain leading blank
                continuation = self.token_encode(continuation)

                # query, remove last token from continuation, truncate from left is longer than model ctx length
                query = ctx + continuation[:-1]
                query = query[-self.model_ctx_len:]

                # get domain conditional query
                # we don't expect this to be longer than self.model_ctx_len and it won't make sense to truncate from left
                dc_query = dc + continuation[:-1]

                # form a sample
                self.samples.append(
                    {
                        'doc_id': doc_id,
                        'cont_id': cont_id,
                        'ctx': ctx,
                        'continuation': continuation,
                        'ctx_len': len(ctx),
                        'dc_len': len(dc),
                        'cont_len': len(continuation),  # even if query has last token removed, LM will output same cont len
                        'cont_str_len': cont_str_len,
                        'query': query,    # remove last token from continuation
                        'dc_query': dc_query,
                        'label_id': label_id,
                    }
                )

            doc_id += 1

    def pad_tokens_until_max(self, tokens, max_len=2048):
        """truncate from left if len(tokens) > model_ctx_len, max_len is not considered then
            queries are already truncated at max length of model_ctx_len
            this acts as additional check for all types of sequences in the batch
        """
        if len(tokens) > self.model_ctx_len:
            return tokens[-self.model_ctx_len:]
        else:
            # pad to max_len, but check again if this padding exceeded self.model_ctx_len
            # this time truncate from right side of the sequence because additional padding caused len(tokens) > self.model_ctx_len
            tokens = tokens + [self.tokenizer.pad_token_id] * (max_len - len(tokens))

            if len(tokens) > self.model_ctx_len:
                tokens = tokens[:self.model_ctx_len]

            return tokens

    def collate_fn(self, data):
        # pad to max length
        # 'ctx', 'continuation', 'query' can all have variable length
        max_ctx_len = 0
        max_cont_len = 0
        max_query_len = 0
        max_dc_query_len = 0

        for sample in data:
            if len(sample['ctx']) > max_ctx_len:
                max_ctx_len = len(sample['ctx'])

            if len(sample['continuation']) > max_cont_len:
                max_cont_len = len(sample['continuation'])

            if len(sample['query']) > max_query_len:
                max_query_len = len(sample['query'])

            if len(sample['dc_query']) > max_dc_query_len:
                max_dc_query_len = len(sample['dc_query'])

        doc_ids = []
        cont_ids = []
        ctxs = []
        continuations = []
        ctx_lens = []
        dc_lens = []
        cont_lens = []
        cont_str_lens = []
        queries = []
        dc_queries = []
        label_ids = []

        # pad according to max_lengths
        for sample in data:
            doc_ids.append(sample['doc_id'])
            cont_ids.append(sample['cont_id'])

            ctxs.append(torch.LongTensor(self.pad_tokens_until_max(sample['ctx'], max_len=max_ctx_len)))
#             print(sample['continuation'], len(sample['continuation']), max_cont_len)
#             print(self.pad_tokens_until_max(sample['continuation'], max_len=max_cont_len))
            continuations.append(torch.LongTensor(self.pad_tokens_until_max(sample['continuation'], max_len=max_cont_len)))

            ctx_lens.append(sample['ctx_len'])
            dc_lens.append(sample['dc_len'])
            cont_lens.append(sample['cont_len'])
            cont_str_lens.append(sample['cont_str_len'])

            queries.append(torch.LongTensor(self.pad_tokens_until_max(sample['query'], max_len=max_query_len)))
            dc_queries.append(torch.LongTensor(self.pad_tokens_until_max(sample['dc_query'], max_len=max_dc_query_len)))

            label_ids.append(sample['label_id'])

        batch = {
            'doc_id': torch.LongTensor(doc_ids),
            'cont_id': torch.LongTensor(cont_ids),
            'ctx': torch.stack(ctxs),
            'continuation': torch.stack(continuations),
            'ctx_len': torch.LongTensor(ctx_lens),
            'dc_len': torch.LongTensor(dc_lens),
            'cont_len': torch.LongTensor(cont_lens),  # since query has last token removed from continuation
            'cont_str_len': torch.LongTensor(cont_str_lens),
            'input_ids': torch.stack(queries),
            'dc_input_ids': torch.stack(dc_queries),
            'label_id': torch.LongTensor(label_ids),
        }

        return batch

    def token_encode(self, string):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def token_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    @abc.abstractmethod
    def doc_to_text(self, doc):
        """Match EAI eval harness
            returns a single context string
        """
        pass

    @abc.abstractmethod
    def doc_to_continuations(self, doc):
        """Match EAI eval harness
            returns a list of continuations
        """
        pass

    @abc.abstractmethod
    def doc_to_label(self, doc):
        """Match EAI eval harness
            returns continuation id which corresponds to true label
        """
        pass

    def doc_to_domain_conditional(self, doc):
        """Provide string for domain conditional normalization
            by default its blank string, continuation normalized by prob conditioned on a blank
        """
        return " "


class PIQA(ICLMultiChoiceTaskDataset):
    """PIQA sends context in the following fashion: "Question: GOAL\nAnswer:"
        space added as prefix to each continuation

        implement PMI_DC

        {
            'goal': "How do I ready a guinea pig cage for it's new occupants?",
            'sol1': 'Provide the guinea pig with a cage full of a few inches of bedding made of ripped paper strips, you will also need to supply it with a water bottle and a food dish.',
            'sol2': 'Provide the guinea pig with a cage full of a few inches of bedding made of ripped jeans material, you will also need to supply it with a water bottle and a food dish.',
            'label': 0
        }
    """
    def __init__(self, tokenizer, dataset_path='piqa', dataset_name=None):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )

    def doc_to_text(self, doc):
        return "Question: " + doc["goal"] + "\nAnswer:"

    def doc_to_continuations(self, doc):
        # add spaces in front of continuation
        return [" " + doc['sol1'], " " + doc['sol2']]

    def doc_to_label(self, doc):
        return doc['label']

    def doc_to_domain_conditional(self, doc):
        return "Answer:"


class HellaSwag(ICLMultiChoiceTaskDataset):
    """HellaSwag concats "ACTIVITY_LABEL: CTX_A CTX_B.capitalize()" to form context and then sends endings as continuations
        space added as prefix to each continuation

    {
        'activity_label': 'Roof shingle removal',
        'ctx_a': 'A man is sitting on a roof.',
        'ctx_b': 'he',
        'ctx': 'A man is sitting on a roof. he',
        'endings': ['is using wrap to wrap a pair of skis.', 'is ripping level tiles off.', "is holding a rubik's cube.", 'starts pulling up roofing on a roof.'],
        'label': '3'
    }
    """
    def __init__(self, tokenizer, dataset_path='hellaswag', dataset_name=None):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )

    @classmethod
    def preprocess(cls, text):
        text = text.strip()
        # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")

        return text

    def doc_to_text(self, doc):
        return self.preprocess(doc["activity_label"] + ": " + doc["ctx_a"] + " " + doc["ctx_b"].capitalize())

    def doc_to_continuations(self, doc):
        # add spaces in front of continuation
        return [" " + self.preprocess(ending) for ending in doc['endings']]

    def doc_to_label(self, doc):
        return int(doc['label'])

    def doc_to_domain_conditional(self, doc):
        domain_conditional = self.preprocess(doc["ctx_b"].capitalize())

        # ensure non 0 len domain conditional
        if len(domain_conditional) == 0:
            return self.preprocess(doc["ctx_a"]).split(' ')[-1]

        return domain_conditional


class WinoGrande(ICLMultiChoiceTaskDataset):
    """Prompt: split sentence at _ "SENTENCE[:idx] + OPTION1/OPTION2", where idx = SENTENCE.index("_")
        implement PMI_DC
        acc, random at 50%
        continuation is everything in setnence after '_' (" SENTENCE[idx:].strip()")

        Req_loglikelihood('People think Samantha', ' is embarassed, because Samantha made snide comments about the shirt Rebecca was wearing.')
        Req_loglikelihood('People think Rebecca', ' is embarassed, because Samantha made snide comments about the shirt Rebecca was wearing.')

    {
        'sentence': 'People think _ is embarassed, because Samantha made snide comments about the shirt Rebecca was wearing.',
        'option1': 'Samantha',
        'option2': 'Rebecca',
        'answer': '2'
    }

    TODO: might need to write custom metric for Winogrande
    """
    def __init__(self, tokenizer, dataset_path='winogrande', dataset_name='winogrande_xl'):
        # all winogrande datasets have same val set
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )

    def prep_examples(self):
        """Overwrite for WinoGrande as multiple ctx, single continuation
        """
        doc_id = 0
        for doc in self.dataset:
            # here ctx is a list
            ctxs = self.doc_to_text(doc)
            dcs = self.doc_to_domain_conditional(doc)

            continuation = self.doc_to_continuations(doc)
            label_id = self.doc_to_label(doc)
            cont_str_len = len(continuation) - 1    # continuations contain leading blank space

            # tokenize
            continuation = self.token_encode(continuation)

            for cont_id, (ctx, dc) in enumerate(zip(ctxs, dcs)):
                ctx = self.token_encode(ctx)
                dc = self.token_encode(dc)

                # query, remove last token from continuation, truncate from left is longer than model ctx length
                query = ctx + continuation[:-1]
                query = query[-self.model_ctx_len:]

                # get domain conditional query
                # we don't expect this to be longer than self.model_ctx_len and it won't make sense to truncate from left
                dc_query = dc + continuation[:-1]

                # form a sample
                self.samples.append(
                    {
                        'doc_id': doc_id,
                        'cont_id': cont_id,
                        'ctx': ctx,
                        'continuation': continuation,
                        'ctx_len': len(ctx),
                        'dc_len': len(dc),
                        'cont_len': len(continuation),  # even if query has last token removed, LM will output same cont len
                        'cont_str_len': cont_str_len,
                        'query': query,    # remove last token from continuation
                        'dc_query': dc_query,
                        'label_id': label_id,
                    }
                )

            doc_id += 1

    def doc_to_text(self, doc):
        # special case where there are multiple ctx and single continuation
        pronoun_loc = doc["sentence"].index("_")

        ctx = []
        for option in [doc['option1'], doc['option2']]:
            ctx.append(doc["sentence"][:pronoun_loc] + option)

        return ctx

    def doc_to_continuations(self, doc):
        # add spaces in front of continuation
        pronoun_loc = doc["sentence"].index("_") + 1
        return " " + doc["sentence"][pronoun_loc:].strip()

    def doc_to_label(self, doc):
        return int(doc['answer']) - 1

    def doc_to_domain_conditional(self, doc):
        """same number of domain conditionals as context
        """
        return [doc['option1'], doc['option2']]


class OpenBookQA(ICLMultiChoiceTaskDataset):
    """OBQA: question_stem is sent as context (no special prompt format) and choices are sent as continuation
        space added as prefix to each continuation

        implement PMI_DC

    {
        'question_stem': 'Frilled sharks and angler fish live far beneath the surface of the ocean, which is why they are known as',
        'choices': {'text': ['Deep sea animals', 'fish', 'Long Sea Fish', 'Far Sea Animals'],
        'label': ['A', 'B', 'C', 'D']},
        'answerKey': 'A'
    }
    """
    def __init__(self, tokenizer, dataset_path='openbookqa', dataset_name=None):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )

    def doc_to_text(self, doc):
        return doc["question_stem"]

    def doc_to_continuations(self, doc):
        # add spaces in front of continuation
        return [" " + choice for choice in doc["choices"]["text"]]

    def doc_to_label(self, doc):
        return ["A", "B", "C", "D"].index(doc["answerKey"].strip())

    def doc_to_domain_conditional(self, doc):
        return doc["question_stem"].strip().split(' ')[-1]


class BoolQ(ICLMultiChoiceTaskDataset):
    """Prompt: "PASSAGE\nQuestion: QUESTION?\nAnswer:"
        acc, random at 50% (SuperGLUE)
        continuation: yes, no

        {
            'question': 'is ncis new orleans over for the season',
            'passage': 'NCIS: New Orleans (season 4) -- The fourth season of NCIS: New Orleans premiered on September 26, 2017 on CBS. The series continues to air following Bull, Tuesday at 10:00 p.m. (ET) and contained 24 episodes. The season concluded on May 15, 2018.',
            'label': 1
        }
    """
    def __init__(self, tokenizer, dataset_path='boolq', dataset_name=None):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )

    def doc_to_text(self, doc):
        return doc['passage'] + "\nQuestion: " + doc['question'] + "?\nAnswer:"

    def doc_to_continuations(self, doc):
        # add spaces in front of continuation
        return [" yes", " no"]

    def doc_to_label(self, doc):
        # if doc['answer'] is True, return index of " yes" which is 0
        if doc['answer']:
            return 0
        else:
            return 1

    def doc_to_domain_conditional(self, doc):
        return "Answer:"


class SciQ(ICLMultiChoiceTaskDataset):
    """SciQ sends context as "SUPPORT\nQuestion: QUESTION\nAnswer:" and then distractors + correct_answer as continuations
        space added as prefix to each continuation

        implement PMI_DC

    {
        'question': 'Who proposed the theory of evolution by natural selection?',
        'distractor3': 'Scopes',
        'distractor1': 'Linnaeus',
        'distractor2': 'shaw',
        'correct_answer': 'darwin',
        'support': ''
    }
    """
    def __init__(self, tokenizer, dataset_path='sciq', dataset_name=None):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )

    def doc_to_text(self, doc):
        return doc["support"] + "\nQuestion: " + doc["question"] + "\nAnswer:".strip()

    def doc_to_continuations(self, doc):
        # add spaces in front of continuation
        return [" " + doc["distractor1"], " " + doc["distractor2"], " " + doc["distractor3"], " " + doc["correct_answer"]]

    def doc_to_label(self, doc):
        return 3

    def doc_to_domain_conditional(self, doc):
        return "Answer:"


class ArcEasy(ICLMultiChoiceTaskDataset):
    """ArcEasy creates context with "Question: QUESTION\nAnswer:" and sends the choices as continuations
        space added as prefix to each continuation

    {
        'question': 'Which technology was developed most recently?',
        'choices': {'text': ['cellular telephone', 'television', 'refrigerator', 'airplane'],
        'label': ['A', 'B', 'C', 'D']},
        'answerKey': 'A'
    }
    """
    def __init__(self, tokenizer, dataset_path='ai2_arc', dataset_name='ARC-Easy'):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )

    def doc_to_text(self, doc):
        return "Question: " + doc["question"] + "\nAnswer:"

    def doc_to_continuations(self, doc):
        # add spaces in front of continuation
        return [" " + choice for choice in doc["choices"]["text"]]

    def doc_to_label(self, doc):
        # some doc["answerKey"] are stored as numbers
        num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}

        if doc["answerKey"] in num_to_letter:
            doc["answerKey"] = num_to_letter[doc["answerKey"]]

        return ["A", "B", "C", "D", "E"].index(doc["answerKey"])

    def doc_to_domain_conditional(self, doc):
        return "Answer:"


class ArcChallenge(ArcEasy):
    """ArcChallenge follows the same prompt format as ArcEasy.
        implement PMI_DC
    """
    def __init__(self, tokenizer, dataset_path='ai2_arc', dataset_name='ARC-Challenge'):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )


class COPA(ICLMultiChoiceTaskDataset):
    """Prompt: "PREMISE.strip()[:-1] because/therefore"
        Req_loglikelihood('The pair of students came under scrutiny by the teacher because', ' the students both received excellent grades.'
        continuations: CHOICE1/CHOICE2

        "cause": "because",
        "effect": "therefore",

        implement PMI_DC
        acc, random at 50%

        {
            'premise': 'The pair of students came under scrutiny by the teacher.',
            'choice1': 'The students both received excellent grades.',
            'choice2': 'Their responses on the assignment were identical.',
            'question': 'cause',
            'label': 1
        }
    """
    def __init__(self, tokenizer, dataset_path='super_glue', dataset_name='copa'):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )

    def doc_to_text(self, doc):
        connector = "because" if doc["question"] == "cause" else "therefore"

        # remove the period
        return doc["premise"].strip()[:-1] + " " + connector

    def doc_to_continuations(self, doc):
        # add spaces in front of continuation
        def convert_choice(choice):
            return choice[0].lower() + choice[1:]

        return [" " + convert_choice(doc["choice1"]), " " + convert_choice(doc["choice2"])]

    def doc_to_label(self, doc):
        return doc["label"]

    def doc_to_domain_conditional(self, doc):
        return "because" if doc["question"] == "cause" else "therefore"


class RTE(ICLMultiChoiceTaskDataset):
    """Prompt: "SENTENCE1\nQuestion: SENTENCE2 True or False?\nAnswer:"
        implement PMI_DC
        acc, random at 50% (GLUE)
        continuations: True, False

        {
            'sentence1': 'The number of Danes opposed to swapping the krone for the euro has increased slightly to 35.3 percent, up from 34.6 percent in April, according to a poll published on Thursday by Danske Bank.',
            'sentence2': 'The introduction of the euro has been opposed.',
            'label': 0,
        }
    """
    def __init__(self, tokenizer, dataset_path='nyu-mll/glue', dataset_name='rte'):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )

    def doc_to_text(self, doc):
        return doc["sentence1"] + "\nQuestion: " + doc["sentence2"] + " True or False?\nAnswer:"

    def doc_to_continuations(self, doc):
        # add spaces in front of continuation
        return [" True", " False"]

    def doc_to_label(self, doc):
        return doc["label"]

    def doc_to_domain_conditional(self, doc):
        return "Answer:"


class CommitmentBank(ICLMultiChoiceTaskDataset):
    """Prompt: "PREMISE\nQuestion: HYPOTHESIS. True, False or Neither?\nAnswer:"
    continuations: True, False, Neither

        implement PMI_DC
        acc/F1, random at 33% acc. (SuperGLUE)

    {
        'premise': 'Then they would awake, terrified and sweating, to find themselves in white starched linen, in a comfortable bed, in peaceful England. And all would be well. It may be said that although he survived it the siege nevertheless had a bad effect on the Collector.',
        'hypothesis': 'the siege nevertheless had a bad effect on the Collector',
        'label': 0
    }
    """
    def __init__(self, tokenizer, dataset_path='super_glue', dataset_name='cb'):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )

    def doc_to_text(self, doc):
        return doc["premise"] + "\nQuestion: " + doc["hypothesis"] + ". True, False or Neither?\nAnswer:"

    def doc_to_continuations(self, doc):
        # add spaces in front of continuation
        return [" True", " False", " Neither"]

    def doc_to_label(self, doc):
        return doc["label"]

    def doc_to_domain_conditional(self, doc):
        return "Answer:"


class MRPC(ICLMultiChoiceTaskDataset):
    """Prompt for MRPC is formed using "Sentence 1: SENTENCE1\nSentence 2: SENTENCE2\nQuestion: Do both sentences mean the same thing?\nAnswer:"
        acc/F1, random at 50% acc. (GLUE)
        continuations: yes and no

        {
            'sentence1': 'In fiction : Edward P. Jones ( " The Known World " ) and Scott Spencer ( " A Ship Made of Paper " ) .',
            'sentence2': 'The fifth nominee for fiction is Scott Spencer , for A Ship Made of Paper .',
            'label': 0
        }
    """
    def __init__(self, tokenizer, dataset_path='nyu-mll/glue', dataset_name='mrpc'):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )

    @classmethod
    def preprocess(self, string):
        string = string.replace(" n't", "n't")
        string = string.replace(" )", ")")
        string = string.replace("( ", "(")
        string = string.replace('" ', '"')
        string = string.replace(' "', '"')

        string = re.sub(r" (['.,])", r"\1", string)

        return string

    def doc_to_text(self, doc):
        return "Sentence 1: " + self.preprocess(doc["sentence1"]) + "\nSentence 2: " + self.preprocess(doc["sentence2"]) + "\nQuestion: Do both sentences mean the same thing?\nAnswer:"

    def doc_to_continuations(self, doc):
        # add spaces in front of continuation
        return [" yes", " no"]

    def doc_to_label(self, doc):
        # if doc['label'] is True, return index of " yes" which is 0
        if doc['label']:
            return 0
        else:
            return 1

    def doc_to_domain_conditional(self, doc):
        return "Answer:"


class SST2(ICLMultiChoiceTaskDataset):
    """SST2 task formats prompts as "SENTENCE\nQuestion: Is this sentence positive or negative?\nAnswer:"
        some preprocessing done on sentence

        constructs 2 requests, 1 for positive and another for negative
        positive and negative have just 1 token in tokenizer
        positive: 1313
        negative: 2430

        implement PMI_DC
        acc, random at 50% (GLUE)

        {
            'sentence': "harrison 's flowers puts its heart in the right place , but its brains are in no particular place at all . ",
            'label': 1,
        }
    """
    def __init__(self, tokenizer, dataset_path='nyu-mll/glue', dataset_name='sst2'):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )

    @classmethod
    def preprocess(self, string):
        string = string.replace(" n't", "n't")
        string = string.replace(" )", ")")
        string = string.replace("( ", "(")
        string = string.replace('" ', '"')
        string = string.replace(' "', '"')

        string = re.sub(r" (['.,])", r"\1", string)

        return string

    def doc_to_text(self, doc):
        return self.preprocess(doc["sentence"]) + "\nQuestion: Is this sentence positive or negative?\nAnswer:"

    def doc_to_continuations(self, doc):
        # add spaces in front of continuation
        # # {1: "positive", 0: "negative"}
        return [" negative", " positive"]

    def doc_to_label(self, doc):
        # {1: "positive", 0: "negative"}
        return doc["label"]

    def doc_to_domain_conditional(self, doc):
        return "Answer:"
