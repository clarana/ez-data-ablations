from composer.utils import dist
from composer.core import Evaluator
from evaluator.multichoicetasks import (
    PIQA,
    HellaSwag,
    WinoGrande,
    OpenBookQA,
    BoolQ,
    SciQ,
    ArcEasy,
    ArcChallenge,
    COPA,
    RTE,
    CommitmentBank,
    MRPC,
    SST2,
)

from torch.utils.data import DataLoader


def create_eval_suite(tokenizer, eval_batch_size):
    # add dataloaders here
    # piqa
    piqa_dataset = PIQA(tokenizer=tokenizer)
    piqa_sampler = dist.get_sampler(piqa_dataset, shuffle=False)
    piqa_dataloader = DataLoader(piqa_dataset, batch_size=eval_batch_size, collate_fn=piqa_dataset.collate_fn, sampler=piqa_sampler)

    # hellaswag
    hellaswag_dataset = HellaSwag(tokenizer=tokenizer)
    hellaswag_sampler = dist.get_sampler(hellaswag_dataset, shuffle=False)
    hellaswag_dataloader = DataLoader(hellaswag_dataset, batch_size=eval_batch_size, collate_fn=hellaswag_dataset.collate_fn, sampler=hellaswag_sampler)

    # winogrande
    winogrande_dataset = WinoGrande(tokenizer=tokenizer)
    winogrande_sampler = dist.get_sampler(winogrande_dataset, shuffle=False)
    winogrande_dataloader = DataLoader(winogrande_dataset, batch_size=eval_batch_size, collate_fn=winogrande_dataset.collate_fn, sampler=winogrande_sampler)

    # OpenBookQA
    openbookqa_dataset = OpenBookQA(tokenizer=tokenizer)
    openbookqa_sampler = dist.get_sampler(openbookqa_dataset, shuffle=False)
    openbookqa_dataloader = DataLoader(openbookqa_dataset, batch_size=eval_batch_size, collate_fn=openbookqa_dataset.collate_fn, sampler=openbookqa_sampler)

    # BoolQ
    boolq_dataset = BoolQ(tokenizer=tokenizer)
    boolq_sampler = dist.get_sampler(boolq_dataset, shuffle=False)
    boolq_dataloader = DataLoader(boolq_dataset, batch_size=eval_batch_size, collate_fn=boolq_dataset.collate_fn, sampler=boolq_sampler)

    # SciQ
    sciq_dataset = SciQ(tokenizer=tokenizer)
    sciq_sampler = dist.get_sampler(sciq_dataset, shuffle=False)
    sciq_dataloader = DataLoader(sciq_dataset, batch_size=eval_batch_size, collate_fn=sciq_dataset.collate_fn, sampler=sciq_sampler)

    # Arc-Easy
    arceasy_dataset = ArcEasy(tokenizer=tokenizer)
    arceasy_sampler = dist.get_sampler(arceasy_dataset, shuffle=False)
    arceasy_dataloader = DataLoader(arceasy_dataset, batch_size=eval_batch_size, collate_fn=arceasy_dataset.collate_fn, sampler=arceasy_sampler)

    # Arc-Challenge
    arcchallenge_dataset = ArcChallenge(tokenizer=tokenizer)
    arcchallenge_sampler = dist.get_sampler(arcchallenge_dataset, shuffle=False)
    arcchallenge_dataloader = DataLoader(arcchallenge_dataset, batch_size=eval_batch_size, collate_fn=arcchallenge_dataset.collate_fn, sampler=arcchallenge_sampler)

    # COPA
    copa_dataset = COPA(tokenizer=tokenizer)
    copa_sampler = dist.get_sampler(copa_dataset, shuffle=False)
    copa_dataloader = DataLoader(copa_dataset, batch_size=eval_batch_size, collate_fn=copa_dataset.collate_fn, sampler=copa_sampler)

    # RTE
    rte_dataset = RTE(tokenizer=tokenizer)
    rte_sampler = dist.get_sampler(rte_dataset, shuffle=False)
    rte_dataloader = DataLoader(rte_dataset, batch_size=eval_batch_size, collate_fn=rte_dataset.collate_fn, sampler=rte_sampler)

    # CB
    cb_dataset = CommitmentBank(tokenizer=tokenizer)
    cb_sampler = dist.get_sampler(cb_dataset, shuffle=False)
    cb_dataloader = DataLoader(cb_dataset, batch_size=eval_batch_size, collate_fn=cb_dataset.collate_fn, sampler=cb_sampler)

    # MRPC
    mrpc_dataset = MRPC(tokenizer=tokenizer)
    mrpc_sampler = dist.get_sampler(mrpc_dataset, shuffle=False)
    mrpc_dataloader = DataLoader(mrpc_dataset, batch_size=eval_batch_size, collate_fn=mrpc_dataset.collate_fn, sampler=mrpc_sampler)

    # SST2
    sst2_dataset = SST2(tokenizer=tokenizer)
    sst2_sampler = dist.get_sampler(sst2_dataset, shuffle=False)
    sst2_dataloader = DataLoader(sst2_dataset, batch_size=eval_batch_size, collate_fn=sst2_dataset.collate_fn, sampler=sst2_sampler)

    # add tasks here
    # piqa
    piqa_task = Evaluator(
        label='piqa',
        dataloader=piqa_dataloader,
        metric_names=["len_norm"],
    )

    # hellaswag
    hellaswag_task = Evaluator(
        label='hellaswag',
        dataloader=hellaswag_dataloader,
        metric_names=["len_norm"],
    )

    # winogrande
    winogrande_task = Evaluator(
        label='winogrande',
        dataloader=winogrande_dataloader,
        metric_names=["acc"],
    )

    # OpenBookQA
    openbookqa_task = Evaluator(
        label='openbook_qa',
        dataloader=openbookqa_dataloader,
        metric_names=["len_norm"],
    )

    # BoolQ
    boolq_task = Evaluator(
        label='boolq',
        dataloader=boolq_dataloader,
        metric_names=["len_norm"],
    )

    # SciQ
    sciq_task = Evaluator(
        label='sciq',
        dataloader=sciq_dataloader,
        metric_names=["acc"],
    )

    # Arc-Easy
    arceasy_task = Evaluator(
        label='arc_easy',
        dataloader=arceasy_dataloader,
        metric_names=["acc"],
    )

    # Arc-Challenge
    arcchallenge_task = Evaluator(
        label='arc_challenge',
        dataloader=arcchallenge_dataloader,
        metric_names=["pmi_dc"],
    )

    # COPA
    copa_task = Evaluator(
        label='copa',
        dataloader=copa_dataloader,
        metric_names=["acc"],
    )

    # RTE
    rte_task = Evaluator(
        label='rte',
        dataloader=rte_dataloader,
        metric_names=["len_norm"],
    )

    # CB
    cb_task = Evaluator(
        label='commitment_bank',
        dataloader=cb_dataloader,
        metric_names=["acc"],
    )

    # MRPC
    mrpc_task = Evaluator(
        label='mrpc',
        dataloader=mrpc_dataloader,
        metric_names=["f1"],
    )

    # SST2
    sst2_task = Evaluator(
        label='sst2',
        dataloader=sst2_dataloader,
        metric_names=["acc"],
    )

    return [
        piqa_task,
        hellaswag_task,
        winogrande_task,
        openbookqa_task,
        boolq_task,
        sciq_task,
        arceasy_task,
        arcchallenge_task,
        copa_task,
        rte_task,
        cb_task,
        mrpc_task,
        sst2_task,
    ]
